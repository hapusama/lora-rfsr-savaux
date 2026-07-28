# RFSR OTA Handoff (2026-07-28)

## 状态

本轮已完成 received-to-received OTA 微调模型的存档、端到端 LoRa 解码评估入口，以及 RFSR 后接 Savaux/branch-GLS 的逐符号诊断入口。

最终微调权重已复制到仓库内，供其他机器直接使用：

```text
third_party/rfsr/checkpoints/model_model0v0hl_bs1_osf4_ds250_lr0.0001_wd1e-05_ota_received_g24_dsf8.pth
SHA256: d28caf49a62f9ea5d27684e8dc24ebd07f0920fb5a1f3ee9d14dacb285dc4b90
```

源训练副本仍在：

```text
/root/autodl-tmp/rfsr-run/finetune/checkpoints/model_model0v0hl_bs1_osf4_ds250_lr0.0001_wd1e-05_ota_received_g24_dsf8.pth
```

两者已做 SHA256 一致性核对。

## 代码入口

- `tools/evaluate_rfsr_ota_decode.py`
  - 已补充中文说明和中文代码注释。
  - 完整链：250 kS/s low rate、4x 插值、RFSR 1 MS/s、原生 1 MS/s 共四条支路，分别送入 `noisy_iq.detector` 的 GNU Radio/gr-lora_sdr 完整链，以 CRC 和完整帧内容评分。
  - 可用 `--extra-snr-db` 重复给多个离散点，或使用 `--extra-snr-start-db`、`--extra-snr-stop-db`、`--extra-snr-step-db` 做细步长网格。
  - 每个物理包、每个 SNR 点只生成一条高率复 AWGN，再由所有支路共享；RFSR 和插值均从这条相同高率带噪 IQ 下采样得到输入。
  - 默认 `--rfsr-snr-conditioning manifest`，即保持微调时使用的 manifest detector SNR 条件输入。`minimum` 仅用于把该值和人工噪声 SNR 取较小值的消融，不应作为默认可比结果。
  - `--savaux-symbol-count > 0` 时，额外运行 `weak_decoder/os_lora` 已有的 `paper_oversampled_spectrum` 与 `branch_gls_scores`。GLS 协方差只从同一 held-out test split 的包前 off-packet 噪声估计。

- `weak_decoder/rf_super_resolution/frontend.py`
  - 仍是唯一的 RFSR 适配层，加载精确 `model0v0hl` 状态字典并以 `eval()/inference_mode()` 推理。

## 评估环境

必须先载入 GNU Radio，再导入 PyTorch。脚本本身已保证导入顺序；运行环境为：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate grlora
cd /root/lora-rfsr-savaux
export PYTHONPATH=/root/lora-rfsr-savaux/third_party/rfsr
```

公共参数：

```bash
CHECKPOINT=/root/lora-rfsr-savaux/third_party/rfsr/checkpoints/model_model0v0hl_bs1_osf4_ds250_lr0.0001_wd1e-05_ota_received_g24_dsf8.pth
OTA_ROOT=/root/autodl-tmp/lora-rfsr-savaux/data/reference_phy/rfsr_db
```

## 可直接复现的命令

原始 OTA、五个 held-out 物理包的完整链 smoke test：

```bash
python -B tools/evaluate_rfsr_ota_decode.py \
  --ota-root "$OTA_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --output /root/autodl-tmp/rfsr-run/finetune/ota_decode_raw.json \
  --ota-max-groups 24 \
  --ota-split-seed 42 \
  --limit 5 \
  --device cuda
```

低 SNR 门限区间的半 dB 网格：

```bash
python -B tools/evaluate_rfsr_ota_decode.py \
  --ota-root "$OTA_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --output /root/autodl-tmp/rfsr-run/finetune/ota_decode_snr_grid.json \
  --ota-max-groups 24 \
  --ota-split-seed 42 \
  --limit 5 \
  --device cuda \
  --noise-seed 20260728 \
  --extra-snr-start-db -16.5 \
  --extra-snr-stop-db -20 \
  --extra-snr-step-db -0.5
```

在一个指定点追加 Savaux/branch-GLS 诊断。SF12 的 Savaux 论文 DFT 很重，先使用每包一个 payload symbol：

```bash
python -B tools/evaluate_rfsr_ota_decode.py \
  --ota-root "$OTA_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --output /root/autodl-tmp/rfsr-run/finetune/ota_decode_savaux_neg15.json \
  --ota-max-groups 24 \
  --ota-split-seed 42 \
  --limit 5 \
  --device cuda \
  --extra-snr-db -15 \
  --noise-seed 20260728 \
  --savaux-symbol-count 1 \
  --savaux-symbol-kind payload \
  --savaux-gls-extra-snr-db -15
```

结果 JSON 的 `conditions[*].summary` 是完整 CRC 解码结果；`conditions[*].savaux_branch_gls` 是逐符号诊断结果。

## 当前结果

评估集固定为 `ota-max-groups=24`、split seed `42` 的 held-out test，其中 q0/ADC0 有 5 个独立物理包。额外 AWGN 的功率相对当前收到 OTA 波形总功率定义。

在三个独立 AWGN 种子上，每个 SNR 为 15 次 packet 尝试（5 包 x 3 种子）：

| 额外 SNR | low 250 kS/s | 插值 1 MS/s | RFSR 1 MS/s | 原生 1 MS/s |
| --- | ---: | ---: | ---: | ---: |
| -18 dB | 3/15 | 3/15 | 10/15 | 3/15 |
| -18.5 dB | 1/15 | 2/15 | 6/15 | 2/15 |
| -19 dB | 0/15 | 0/15 | 4/15 | 0/15 |

对应原始结果文件：

```text
/root/autodl-tmp/rfsr-run/finetune/ota_decode_snr_grid_manifest_neg16p5_to_neg20_step0p5.json
/root/autodl-tmp/rfsr-run/finetune/ota_decode_snr_seed20260729_neg18_to_neg19_step0p5.json
/root/autodl-tmp/rfsr-run/finetune/ota_decode_snr_seed20260730_neg18_to_neg19_step0p5.json
```

结论：在这批很小的 held-out 集和三条固定噪声实现上，RFSR 在 `-18` 到 `-19 dB` 的门限区间有可重复的完整 LoRa CRC 解码增益。样本数仍只有 15 次 packet/点，不能把上述比例当成最终 PER 曲线；下一轮应增加 held-out 物理包和 AWGN seeds。

`-14` 到 `-17.5 dB` 的首个种子中四条链均为 `4/5`，`-19.5`、`-20 dB` 时四条链均为 `0/5`，符合增益只在门限区间可见的现象。

## Savaux/GLS 诊断的边界

当前 Savaux/branch-GLS 是“简单拼接”而不是第二套完整 LoRa 接收机：

- RFSR 前没有做时间、CFO、SFO、增益或幅度校正。
- 诊断使用 reference metadata 中的已知包边界和 raw FFT-bin 真值，仅在硬判决之后计算正确率。
- CFO 参数来自 OTA manifest 中下游 detector 的历史估计；没有在 RFSR 前旋转 IQ。
- 当前诊断刻意没有实现 Savaux 自己的 SFO 细同步。对 `-15 dB` 的 5 个首 payload symbol，原生、插值、RFSR 都出现一致的 2 到 4 bin 偏移，Savaux 和 branch-GLS 均为 `0/5`。这是下游细同步缺失导致的共同误差，不能据此判断 RFSR 没有增益。
- 完整 GNU Radio 链负责真实时间/CFO/SFO 同步，因此上表的 CRC 结果才是当前可用的端到端结论。

若要把 Savaux/GLS 作为主接收机，下一步需要从 RFSR 输出中接入正常的 packet detector/frame sync，再将其同步后的 symbol 起点、CFO 和 SFO 估计交给 Savaux。该处理必须放在 RFSR 之后，不能回写到 OTA 训练样本或作为 RFSR 前置校正。
