# RF Super Resolution: A Deep Learning Approach to Spatial Enhancement for LoRa

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19451778.svg)](https://doi.org/10.5281/zenodo.19451778)
[![Dataset DOI](https://img.shields.io/badge/Dataset%20DOI-10.21979%2FN9%2FC6ABM3-blue)](https://doi.org/10.21979/N9/C6ABM3)

---

## Abstract

The analog-to-digital converter (ADC) in a radio frequency (RF) front-end and digital signal processing (DSP) are significant sources of energy consumption, particularly in low-power systems like LoRa. To reliably demodulate weak signals, current systems rely on heavy oversampling — often 8× the signal bandwidth — which imposes a substantial and persistent oversampling tax on the analog front-end and DSP. This paper investigates if this tax can be mitigated by adapting techniques from image and video super resolution. We propose *RF Super Resolution*, a lightweight, real-time neural upscaler for RF signals. Our approach pairs an efficient digital interpolation algorithm with a shallow four-layer CNN. The neural network is trained to learn and correct the residual artifacts introduced by the digital upsampling and noise, effectively mimicking the output of a high-rate analog ADC and denoising filter. We validate our system on a large-scale, over-the-air LoRa study. Our results show that RF-SR, given a 2× Nyquist input (250 kHz), restores demodulation performance of a native 8× oversampled (2 MHz) system at half its sampling rate (1 MHz). This effectively removes the analog oversampling requirement, and provides an additional 1.25 dB SNR gain over the oversampled baseline, making it an efficient and effective signal enhancer suitable for gateway integration or post training quantized deployment at the end-node.

---

## What RF-SR Achieves

- **Removes the analog oversampling requirement**: RF-SR at 250 kHz input (2× Nyquist) restores the demodulation performance of a native 2 MHz (8× oversampled) system.
- **Halves the effective sampling rate**: Full native-quality demodulation at 1 MHz instead of 2 MHz.
- **+1.25 dB SNR gain** over the 8× oversampled baseline on synthetic data.
- **Lightweight**: 4-layer CNN (~25 kB weights).
- **Quantization-friendly**: Supports INT8 post-training quantization (PTQ) for efficient deployment on NPUs and ASICs.

---

## Installation

Requires Python 3.10+. A GPU is recommended for training; inference can run on CPU.

```bash
python3 -m venv venv/
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.
pip install -r requirements.txt
```

> **PyTorch & CUDA**: The default `requirements.txt` targets CUDA 12.8. Adjust the `--extra-index-url` line to match your CUDA version — see [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/). Run `nvidia-smi` to check your CUDA version.

---

## Quick Example

```bash
python3 example/example.py
```

---

## Dataset (RFSR-OTA)

The 10,000-packet over-the-air LoRa IQ dataset used to train and evaluate RF-SR is openly available on NTU's institutional research-data repository, DR-NTU (Data):

- **DOI:** [10.21979/N9/C6ABM3](https://doi.org/10.21979/N9/C6ABM3)
- **Contents:** 10,000 over-the-air LoRa packet recordings (baseband IQ samples; SF12, BW = 125 kHz, CR = 4/5, 16-byte random payload, 8 preamble symbols) collected on the NTU campus, Singapore — spanning indoor LoS, short/mid-range NLoS, and an 800 m over-the-hill link, with varied transmit power and receiver-side RF attenuation for a wide SNR dynamic range. The recordings preserve real hardware impairments (CFO, clock skew) and non-Gaussian urban noise.
- **License:** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
- **Documentation:** the `README.pdf` on the dataset record describes the collection setup, file format, naming convention, and directory layout.

If you use the dataset, please cite it alongside the paper (see [Citation](#citation)):

```bibtex
@data{N9/C6ABM3_2026,
  author    = {Kuster, Andreas},
  publisher = {DR-NTU (Data)},
  title     = {{An Over-the-Air LoRa IQ Dataset for RF Super Resolution (RFSR-OTA)}},
  year      = {2026},
  doi       = {10.21979/N9/C6ABM3},
  url       = {https://doi.org/10.21979/N9/C6ABM3}
}
```

---

## Training

Train the RF-SR model on synthetically generated LoRa SF12 packets (BW = 125 kHz). Training takes approximately 15 minutes on an RTX 3090.

```bash
python3 rfsr/nn/nn.py \
    --model model0v0 \
    --batch_size 5 \
    --osf 4 \
    --dataset_size 250 \
    --learning_rate 0.001 \
    --weight_decay 1e-5 \
    --optimizer adam \
    --dsf 8
```



**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--model` | `model0v0` | Architecture (`model0v0` = 4-layer CNN; the paper's recommended model) |
| `--osf` | `4` | Output oversampling factor (4× → 0.25 MHz in, 1 MHz out) |
| `--dsf` | `8` | Input downsampling factor (8× → 0.25 MHz from 2 MHz reference) |
| `--dataset_size` | `250` | Number of LoRa packets per epoch |
| `--batch_size` | `5` | Batch size |
| `--learning_rate` | `0.001` | Adam learning rate |
| `--num_epochs` | `100` | Training epochs |

The trained model is saved to `checkpoints/model_<name>.pth`. Pre-trained weights are already included in this repository.

### Optional: INT8 Post-Training Quantization

Convert the trained model to INT8 for efficient edge deployment:

```bash
python3 rfsr/nn/nn_quant8.py
```

Output: `checkpoints/model_model0v0_bs5_osf4_ds250_lr0.001_wd1e-05_int8.pt`

This takes approximately 15 minutes (calibration on 500 packets).

---

## Evaluation: RF-SR vs. Standard LoRa

We measure **Packet Error Rate (PER) vs. SNR** for standard LoRa at different sampling rates against RF-SR.

### Step 1 — Run the evaluation

This generates baseline PER curves for standard LoRa at 2, 1, 0.5, and 0.25 MSPS, plus RF-SR, and RF-SR quantization sweeps. Results are appended to `results/per_vs_fs.json`.

```bash
python3 rfsr/per.py --synth_nn
```

> Runtime: 2 h for 1000 samples. Pre-collected results are already in `results/per_vs_fs.json`.

### Step 2 — Plot the comparison

```bash
python3 rfsr/per.py --plot --set synth_nn
```

Output: `results/per_vs_snr_synth_nn.png` — shows RF-SR at 0.25 MSPS vs. standard LoRa at 0.25 / 0.5 / 1 / 2 MSPS.


---

## Citation

If you use RF-SR in your research, please cite:

```bibtex
@inproceedings{rfsr,
  title     = {{RF Super Resolution}: A Deep Learning Approach to Spatial Enhancement for {LoRa}},
  author    = {Kuster, Andreas and Xu, Huatao and Tan, Rui and Li, Mo},
  booktitle = {Proceedings of the 24th Annual International Conference on Mobile Systems, Applications and Services (MobiSys '26)},
  year      = {2026},
  pages     = {464--477},
  doi       = {10.1145/3745756.3809216}
}
```

---

## Credits

The LoRa physical layer implementation (`rfsr/PHY.py`) is based on **SDR-LoRa**, originally written by [Fabio Busacca](https://github.com/fabio-busacca/sdr-lora) and was subsequently extended for this work.

---

## License

This project (except `rfsr/PHY.py`) is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
