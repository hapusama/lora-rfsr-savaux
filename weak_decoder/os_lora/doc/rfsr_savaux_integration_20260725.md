# RF-SR + Savaux repository analysis and minimal integration

Date: 2026-07-25

## 1. Upstream snapshot

The official repository was cloned next to `gr-lora_sdr`:

```text
D:\Desktop\proj\RFSuperResolution
commit 00c135947f855790f458fdc25ae9533c70d77849
```

Relevant upstream files:

```text
RFSuperResolution/
  example/example.py          minimal inference example
  rfsr/interp.py              polyphase interpolation implementation
  rfsr/nn/nn.py               model architecture, training, FP32 loader
  rfsr/nn/nn_quant8.py        post-training quantization and INT8 inference
  rfsr/nn/dataset.py          synthetic and OTA dataset adapters
  rfsr/per.py                 author PER experiment
  rfsr/PHY.py                 LoRa encoder/decoder used by evaluation
  checkpoints/                two FP32 checkpoints and one INT8 checkpoint
```

The checkout is kept separate and unmodified. Weak-decoder uses it through an
adapter so the upstream revision and checkpoint hash remain explicit.

## 2. Data contract

### Synthetic training data

`SyntheticLoRaDataset` generates clean SF12, BW 125 kHz, CR 4/5 packets with a
16-byte payload, CRC, and eight preamble symbols. The target `y` is clean
high-rate IQ. The input is `x = y[::4]` followed by complex AWGN. PyTorch sees:

```text
x: float32 [batch, 2, L]       channel 0 = I, channel 1 = Q
y: float32 [batch, 2, 4L]
snr: float32 [batch, 1]
```

### OTA data

The published loader expects complex64 `.cfile` files. Raw captures are
nominally 2 MSPS. With `DSF=8`, input IQ is 250 kSPS. The clean/reference path
is downsampled by `DSF/OSF = 2`, producing the 1 MSPS target. Therefore the
pretrained contract is exactly:

```text
complex64 250 kSPS -> float32 [B, 2, L]
RF-SR              -> float32 [B, 2, 4L] -> complex64 1 MSPS
```

The 10,000-packet OTA dataset is not included in Git. It is published
separately under dataset DOI `10.21979/N9/C6ABM3`. The current upstream OTA
dataset class also contains hard-coded local paths and unresolved helper names,
so downloading the data alone is not yet a drop-in training path. This does not
block pretrained inference.

## 3. Public checkpoints

```text
Synthetic FP32
  model_model0v0_bs5_osf4_ds250_lr0.001_wd1e-05.pth
  SHA256 e355c8279d77e150d762e3ff1052606cc9bcc6a9b97db0e2b6adbdffbabeeaa0

Synthetic INT8 TorchScript
  model_model0v0_bs5_osf4_ds250_lr0.001_wd1e-05_int8.pt

OTA-fine-tuned FP32
  model_model0v0lopenaltyhl_bs1_osf4_ds250_lr0.0001_wd1e-05_ota_dsf8.pth
  SHA256 a4de5311d70a9c37618a89632d023abe46f847ba18b3d15fb439a513fcd0c398
```

Both FP32 models have 12,802 trainable parameters and map `[1, 2, 64]` to
`[1, 2, 256]`. The minimal integration defaults to the OTA-fine-tuned FP32
checkpoint for real captures. Synthetic experiments should also run the
synthetic checkpoint as a checkpoint-domain ablation.

## 4. Exact inference path

The author model performs the following operations:

1. Insert three zeros between adjacent 250 kSPS samples.
2. Filter I and Q independently with the Kaiser-windowed sinc implementation
   in `rfsr/interp.py` (`width=511`, beta approximately 14.769).
3. Pass the interpolated two-channel tensor through four 1-D convolutional
   layers. In complex-channel notation the widths are `1 -> 16 -> 32 -> 16 -> 1`.
   Every layer uses kernel size 3 and padding 1; the first three use ReLU.
4. Add the CNN output as a residual to the polyphase interpolation output.

The published `model0v0` accepts an SNR argument, but the non-gated network does
not use it. The adapter still records and passes SNR to preserve the upstream
inference signature.

The most important ablation is exact rather than approximate:

```text
frontend.interpolate(low_iq) = author's polyphase tensor
frontend.enhance(low_iq)     = same tensor + author's CNN residual
```

Using SciPy interpolation for the first arm would change the filter and make
the comparison ambiguous.

## 5. Added integration files

```text
weak_decoder/rf_super_resolution/
  frontend.py       lazy upstream loader, provenance, exact interpolation,
                    CNN inference, overlap-cropped chunking
  __init__.py       public frontend API

weak_decoder/os_lora/experiments/
  probe_rfsr_savaux.py
                    one-symbol five-path smoke probe

weak_decoder/os_lora/tests/
  test_rf_super_resolution_frontend.py
                    optional author-checkpoint integration test
```

No file in `baselines/savaux_oversampled` or `os_lora/system/oversampled_glrt.py`
was changed. The probe calls `paper_oversampled_spectrum`,
`estimate_branch_noise_model`, and `branch_gls_scores` directly.

The adapter loads PyTorch only on construction. Existing weak-decoder imports
and tests therefore remain usable in environments without PyTorch. Its default
65,536-sample chunks use 68 low-rate context samples on both sides. This covers
the interpolation FIR support and the four-layer CNN receptive field; the
context is cropped after inference.

## 6. Minimal smoke command

Run from `gr-lora_sdr/weakPacket_decoding` with the isolated environment:

```powershell
D:\Desktop\proj\.venvs\rfsr\Scripts\python.exe -B -m `
  weak_decoder.os_lora.experiments.probe_rfsr_savaux `
  --input-low-iq <250ksps-complex64.cfile> `
  --start-low <synchronized-symbol-start> `
  --sf 12 --bw 125000 `
  --gt-bin <clean-ground-truth-bin> `
  --rfsr-repo D:\Desktop\proj\RFSuperResolution `
  --noise-low-iq <250ksps-noise-only-complex64.cfile> `
  --native-high-iq <independent-1msps-complex64.cfile> `
  --start-high <same-symbol-start-at-1msps> `
  --noise-high-iq <1msps-noise-only-complex64.cfile> `
  --output data\experiments\rfsr_savaux\smoke.json
```

The output contains the selected bin and true-peak/strongest-false-peak ratio
for all available paths, plus mean/max branch correlation. If noise-only IQ is
not supplied, the JSON marks the covariance source as `identity`. If native
high-rate IQ is absent, arm 5 is marked `not_run` rather than synthesized.

## 7. Full five-arm experiment protocol

The formal experiment must start from an independently acquired, clean 1 MSPS
complex64 capture. The current Branch4 dataset is 500 kSPS and cannot serve as
the native-high arm.

For each packet, SNR, and random seed:

1. Freeze packet synchronization, CFO, STO, header fields, and clean payload
   bins from the clean 1 MSPS decode. Do not rerun synchronization per arm.
2. Add one complex-noise realization at 1 MSPS.
3. Construct the 250 kSPS input with one fixed decimation phase. Record the
   mapping and any fractional start quantization. Both interpolation and RF-SR
   must receive this exact same low-rate array.
4. Run the five arms:

```text
A  native 250 kSPS -> ordinary LoRa
B  native 250 kSPS -> exact author interpolation -> Savaux
C  native 250 kSPS -> exact author interpolation + CNN -> ordinary LoRa
D  native 250 kSPS -> exact author interpolation + CNN -> Savaux + branch GLS
E  native ADC 1 MSPS -> Savaux
```

5. Estimate each arm's branch covariance from the corresponding transformation
   of matched off-packet windows. Do not reuse native-high covariance for RF-SR.
6. Pass hard payload bins through the existing codec/CRC path. Report both
   actual payload PER and bin-level packet error (`any payload symbol wrong`).

Per symbol, store:

```text
selected_bin
symbol_error
10 log10(S[ground_truth] / max(S[all false bins]))
Savaux and GLS top-L candidate membership
```

Per packet/SNR/method, aggregate:

```text
PER, bin-PER, SER
mean/median true-to-false ratio and lower quantiles
branch covariance diagonal CV
mean/max absolute off-diagonal correlation
covariance eigenvalues, effective rank, and GLS information
```

## 8. Decision logic for the scientific question

The key contrast is:

```text
CNN gain under Savaux = D - B
Savaux gain after CNN = D - C
native observation gap = E - D
```

Interpretation:

- If `D > B` but `D` is approximately equal to `C`, the gain is primarily CNN
  denoising; Savaux is not extracting additional observations.
- If `D > C` consistently and RF-SR branch covariance has useful additional
  rank/information, the generated offsets help Savaux beyond ordinary decoding.
- If RF-SR branch correlations approach one and `D` is approximately equal to
  `C`, the four offsets are deterministic replicas and GLS should not count
  them as four independent samples.
- `E - D` measures the remaining value of genuinely independent ADC samples.

The formal claim should be based on paired packet/seed outcomes and confidence
intervals, not only mean SER curves.

## 9. Current blockers before a formal result

1. Existing synchronized Branch4 captures are SF10/BW125/500 kSPS (OSR 4), not
   native 1 MSPS. Upsampling these files cannot create arm E.
2. Public weights were trained for SF12/BW125. Applying them to current SF10
   captures is a domain-transfer experiment and must be labeled as such.
3. The older SF12 files in `gr-lora_sdr/data/USRP_IQ` do not carry sufficient
   explicit sampling metadata to treat them as the matched 1 MSPS reference.

The next collection should therefore use SF12/BW125 at 1 MSPS first, with the
same fixed payload and a noise-only recording. SF10 can then be added as a
separate generalization experiment.
