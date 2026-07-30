# RFSR reference 标签训练交接（2026-07-30）

## 当前结论

现有的 g100 checkpoint 不是「真实输入 -> 合成标签」模型，而是两阶段得到的
received-to-received 模型：

```text
阶段 1 synthetic pretrain:
  x = 合成 LoRa 标签 y 的 250 kS/s 抽取 + 合成 AWGN
  y = 干净的合成 1 MS/s LoRa IQ

阶段 2 g100 OTA fine-tune:
  x = 真实 USRP 接收 IQ 的 250 kS/s polyphase view
  y = 同一真实 USRP 接收包的 1 MS/s IQ
```

因此，已有 g100 的输入和输出都是真实接收 IQ；数据集里的 synthetic reference
没有参与其训练 loss。

当前没有 RFSR 训练进程运行。一次 `--ota-target reference` smoke training 只看到
第 1 个 epoch，随后已停止；不能继续使用它的结果。

## 当前 checkpoint

```text
# 历史 g100，received-to-received，仍可复现已有评估结果
/root/autodl-tmp/rfsr-run/finetune_g100_seed42_bs3/checkpoints/
  model_model0v0hl_bs3_osf4_ds250_lr0.0001_wd1e-05_ota_received_g100_dsf8.pth
SHA256: f1cd95ece0a42e5fee66ea61eabe40ea91796db80a9248fac8d85e2fe219b0bd

# 历史 g24，received-to-received
third_party/rfsr/checkpoints/
  model_model0v0hl_bs1_osf4_ds250_lr0.0001_wd1e-05_ota_received_g24_dsf8.pth
```

g100 使用 100 个物理包的 deterministic 6:2:2 split：60 train、20 validation、
20 test。每个物理包有 2 个 ADC phase x 4 个低率 q phase，即 480 个训练 view；
它们不是 480 个独立无线信道样本。

## 哪些数据是真实、哪些是合成

```text
UART 已知 frame / PHY
  ├─ 本地生成 ideal TX complex baseband reference                     [合成]
  └─ STM32/SX1276 实际发射 -> USRP 2 MS/s capture -> 切包/裁剪/相位拆分 [真实]
```

| 数据或步骤 | 来源 | 是否真实 |
|---|---|---|
| `data/raw/ota/*.cfile` | STM32/SX1276 实发、USRP 采集 | 是 |
| `rfsr_db/ota/*_fulltrim.cfile` | 从真实 capture 切出的单包 1 MS/s IQ | 是 |
| 250 kS/s q0..q3 view | 真实 1 MS/s OTA 的确定性抽取 | 是，不是新增合成噪声 |
| `data/reference_phy/reference/signalout_*.cfile` | 按 UART frame 和 PHY 生成 | 否，合成 ideal baseband |
| `reference_phy` synthetic pretrain | 随机 frame + AWGN 的程序生成 | 否，完全合成 |
| Savaux SER 的 symbol 真值 | reference metadata / 已知 frame | 合成/已知真值；只用于评分 |
| 评估中的额外 AWGN | 评估脚本随机生成 | 合成，只在测试时加入 |

reference 与 OTA 使用相同的真实发射 PHY：SF12、BW 125 kHz、CR 4/8、preamble 16、
explicit header、CRC、sync word `0x12`、33-byte PHY frame。每个配对记录都校验
`expected_frame_hex` 与已接收 CRC frame 一致。

reference 是理想发射端基带：`artificial_cfo_hz=0`、`awgn_added=false`。真实 OTA
含 CFO、SFO/STO、通道/接收增益和噪声。

## 对齐状态：已有与没有的部分

`tools/build_rfsr_ota_dataset.py` 已做包级时间窗口对齐：同一 packet、相同长度，
并在 OTA/reference 前面放置相同的 10,000 个 1 MS/s leading-zero 样点。因此
preamble 位于近似相同的位置。

它没有做 CFO 校正、SFO/STO 精细重采样、复增益估计或幅度归一化。metadata 中的
典型字段为：

```text
reference.alignment_status = not_aligned_to_ota
ota.alignment.status       = packet_boundary_aligned
cfo_correction_applied     = false
complex_gain_correction_applied = false
amplitude_normalization_applied = false
```

用户当前明确选择的目标是 **ideal output**：synthetic `y` 不应加回 CFO、SFO、
接收增益或噪声。在线链路仍为：

```text
raw 250 kS/s received IQ -> RFSR -> packet detection / FrameSync -> Savaux
```

离线切包/构造 reference 不等于在线在 RFSR 前调用 FrameSync。

## reference 标签训练失败的直接原因

在同一训练 view 上实测：

```text
received x RMS             = 0.0003647
synthetic reference y RMS  = 0.9982
幅度比                     = 2737 倍
功率差                     = 68.7 dB
零延迟复相关              ≈ 0.007
```

`model0v0hl` 的 loss 是：

```text
time-domain MAE + 0.1 * unnormalised FFT-magnitude MAE
```

对该 reference 标签假设预测全零，可复现：

```text
time MAE = 0.6341
FFT magnitude MAE = 525.3499
hybrid loss = 53.1691
```

reference smoke run 的首轮结果：

```text
Epoch 1: train_loss = 53.0862547, validation_loss = 53.0532605
```

它几乎等于全零预测的 loss，说明未归一化的频域 MAE 被 synthetic reference 的
单位幅度主导。直接继续训练没有比较价值：模型会先尝试学习约 2700 倍的增益，且仍
面对理想 y 与带 CFO/SFO 的真实 x 的逐采样相位差。

## 已改但尚未提交的代码

1. `third_party/rfsr/rfsr/nn/nn.py`
   - `--ota-target` 默认从 `received` 改为 `reference`。
   - 新训练文件名会包含 `_ota_reference_g100_`，不会覆盖已有 g100。
   - `--ota-target received` 仍可完整复现历史 checkpoint。

2. `README.md`
   - OTA 微调命令改为显式 `--ota-target reference`。
   - 明确 reference 只用于离线监督，运行时 RFSR 仍在 FrameSync 前。

3. `tools/evaluate_rfsr_savaux_ser.py`
   - split sidecar 不再硬编码要求 `target_source=received`；reference checkpoint
     可在相同 held-out 物理包上评估。
   - native/RFSR 评估波形仍固定取 received OTA，不能改成 synthetic reference。

4. `tools/evaluate_rfsr_ota_decode.py`
   - 输出 JSON 的 `target` 字段改为 `evaluation_waveform`，避免把评估用真实
     received 波形误写成 checkpoint 的训练标签。

当前工作区还有其它未提交改动（CPU 并行评估、Savaux 注释/测试等）；不得用 reset、
checkout 或覆盖式操作清除它们。

## 下一个窗口应先做什么

在重新跑 reference-target 微调前，先实现并测试 **每包同尺度归一化**。不要把 CFO
或 SFO 加回 ideal `y`；这是用户已否决的另一种 received-coordinate 目标。

建议的最小实现:

```text
对每个训练 item：
  sx = RMS(x 的有效样本区间)
  sy = RMS(y 的有效样本区间)
  x_train = x / sx
  y_train = y / sy

推理：
  y_hat = RFSR(x / sx) * sx
```

reference 的 `sy` 当前约为 1，received 的 `sx` 约为 `3.6e-4`；这样能先消除
68.7 dB 数值尺度错配。必须在单测中验证：received/reference 两种 target 均有限、
归一化后 RMS 接近 1、inference 反归一化后的长度和有限性不变。

这只能解决幅度问题；ideal y 与真实 x 的 CFO/SFO/相位差是该实验刻意要求模型消除的
监督目标。四层局部 CNN 是否足以完成此任务需要重新训练后用 FFT 裕量和 SER 判断，
不能只看 loss。

完成归一化后，在一个全新的目录运行下列对照训练。请保留显式 `--ota-target reference`：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate grlora
export PYTHONPATH=/root/lora-rfsr-savaux/third_party/rfsr

RUN_DIR=/root/autodl-tmp/rfsr-run/finetune_g100_reference_norm_seed42_bs3
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

python -B /root/lora-rfsr-savaux/third_party/rfsr/rfsr/nn/nn.py \
  --model model0v0hl \
  --batch_size 3 \
  --osf 4 --dsf 8 --dataset_size 250 \
  --num_epochs 100 \
  --learning_rate 0.0001 --weight_decay 1e-5 --optimizer adam \
  --ota \
  --ota-root /root/autodl-tmp/lora-rfsr-savaux/data/reference_phy/rfsr_db \
  --ota-target reference \
  --ota-max-groups 100 --ota-split-seed 42 \
  --early-stop-patience 10 \
  --pretrained /root/autodl-tmp/rfsr-run/pretrain/checkpoints/model_model0v0_bs5_osf4_ds250_lr0.001_wd1e-05_synthref_random_snr-22to10_cfo0_stonone.pth
```

该命令目前只是归一化实现后的目标命令；在归一化代码落地前不要直接运行。

## 现有评估链与历史结果

当前主链只保留 RFSR + Savaux，GLS 已移出主流程：

```text
tools/evaluate_rfsr_savaux_ser.py
  -> weak_decoder/synchronization/single_packet.py
       -> clean RFSR/native received waveform 的 FrameSync
  -> weak_decoder/os_lora/system/synchronized_savaux.py
       -> Savaux header/payload 硬判决
  -> weak_decoder/baselines/savaux_oversampled/paper_oversampled_demod.py
       -> Eq.36 branch 谱、Eq.37 branch 合并
  -> 只统计 clean FrameSync 成功包的 SER
```

reference metadata 在 Savaux 硬判决结束后才用于 SER 评分，不会提供 packet 边界、
CFO 或 symbol 真值给 RFSR/FrameSync/Savaux。

共同 held-out 的 5 个物理包上，历史模型与原生的普通 FFT 峰值裕量中位数：

| 指标 | g24 received | g100 received | native 1 MS/s |
|---|---:|---:|---:|
| header | 12.62 dB | 11.00 dB | 23.61 dB |
| payload | 11.70 dB | 10.29 dB | 18.69 dB |

对应 Savaux 峰值裕量中位数：

| 指标 | g24 received | g100 received | native 1 MS/s |
|---|---:|---:|---:|
| header | 12.40 dB | 10.76 dB | 23.77 dB |
| payload | 11.35 dB | 9.81 dB | 16.83 dB |

普通 FFT 与 Savaux 都显示历史 received-to-received RFSR 输出的主峰不如原生
1 MS/s 集中；这不是 Savaux branch 合并单独造成的。

## 上游 RFSR 对照边界

- 上游公开 synthetic dataset：`x`、`y` 都是合成；`y` 是干净高率 LoRa，`x` 是
  `y` 降采样后加 AWGN。
- 上游 legacy OTA loader：`x` 是真实接收 OTA，`y` 是配对的 `signalout` reference。
  上游没有公开完整的 OTA pair 生成/精细对齐流程，因此不能声称当前 reference
  训练已经严格复现论文的 OTA 数据契约。
- 网络主干一致：polyphase 插值 + 四层 `model0v0` CNN；`model0v0hl` 的 `hl` 是
  hybrid MAE loss 选择，不增加网络层。
- 论文正文的 hybrid loss 记录为 frequency weight `0.5`；当前代码默认 `0.1`。
  README quickstart 的 `model0v0` 路径会落到 MSE，不能把该 quickstart 当作论文
  OTA 训练配方。
