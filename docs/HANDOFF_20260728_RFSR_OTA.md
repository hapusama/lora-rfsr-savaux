# RFSR OTA Handoff (2026-07-28)

## 状态

本轮已完成 received-to-received OTA 微调模型的存档、端到端 LoRa 解码评估入口，
以及 RFSR 后接 FrameSync + Savaux 的逐符号评测入口。早期 branch-GLS 诊断仅作
历史记录，当前联合主链已经移除 GLS。

当前 100 包、batch 3 微调权重：

```text
/root/autodl-tmp/rfsr-run/finetune_g100_seed42_bs3/checkpoints/
  model_model0v0hl_bs3_osf4_ds250_lr0.0001_wd1e-05_ota_received_g100_dsf8.pth
SHA256: f1cd95ece0a42e5fee66ea61eabe40ea91796db80a9248fac8d85e2fe219b0bd
```

仓库内仍保留早期 24 包微调权重，供历史结果复现：

```text
third_party/rfsr/checkpoints/model_model0v0hl_bs1_osf4_ds250_lr0.0001_wd1e-05_ota_received_g24_dsf8.pth
SHA256: d28caf49a62f9ea5d27684e8dc24ebd07f0920fb5a1f3ee9d14dacb285dc4b90
```

源训练副本仍在：

```text
/root/autodl-tmp/rfsr-run/finetune/checkpoints/model_model0v0hl_bs1_osf4_ds250_lr0.0001_wd1e-05_ota_received_g24_dsf8.pth
```

两者已做 SHA256 一致性核对。

## 历史完整链入口

> `evaluate_rfsr_ota_decode.py` 的人工噪声位于 RFSR 之前，并会在带噪波形上重新
> 同步，只用于此前 CRC 完整链结果。它不符合当前“干净 RFSR 输出先 FrameSync，
> 再加噪并固定同步做 Savaux”的实验口径。当前 SER 入口见下文
> `RFSR 后同步 Savaux/SER 链`。

- `tools/evaluate_rfsr_ota_decode.py`
  - 已补充中文说明和中文代码注释。
  - 完整链：250 kS/s low rate、4x 插值、RFSR 1 MS/s、原生 1 MS/s 共四条支路，分别送入 `noisy_iq.detector` 的 GNU Radio/gr-lora_sdr 完整链，以 CRC 和完整帧内容评分。
  - 可用 `--extra-snr-db` 重复给多个离散点，或使用 `--extra-snr-start-db`、`--extra-snr-stop-db`、`--extra-snr-step-db` 做细步长网格。
  - 每个物理包、每个 SNR 点只生成一条高率复 AWGN，再由所有支路共享；RFSR 和插值均从这条相同高率带噪 IQ 下采样得到输入。
  - 默认 `--rfsr-snr-conditioning manifest`，即保持微调时使用的 manifest detector SNR 条件输入。`minimum` 仅用于把该值和人工噪声 SNR 取较小值的消融，不应作为默认可比结果。

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
CHECKPOINT=/root/autodl-tmp/rfsr-run/finetune_g100_seed42_bs3/checkpoints/model_model0v0hl_bs3_osf4_ds250_lr0.0001_wd1e-05_ota_received_g100_dsf8.pth
LEGACY_CHECKPOINT=/root/lora-rfsr-savaux/third_party/rfsr/checkpoints/model_model0v0hl_bs1_osf4_ds250_lr0.0001_wd1e-05_ota_received_g24_dsf8.pth
OTA_ROOT=/root/autodl-tmp/lora-rfsr-savaux/data/reference_phy/rfsr_db
```

## 可直接复现的命令

原始 OTA、五个 held-out 物理包的完整链 smoke test：

```bash
python -B tools/evaluate_rfsr_ota_decode.py \
  --ota-root "$OTA_ROOT" \
  --checkpoint "$LEGACY_CHECKPOINT" \
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
  --checkpoint "$LEGACY_CHECKPOINT" \
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

结果 JSON 的 `conditions[*].summary` 是该历史入口的完整 CRC 解码结果，不能与
下文固定 clean FrameSync 的条件 SER 直接比较。

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

## RFSR 后同步 Savaux/SER 链

旧版 `evaluate_rfsr_ota_decode.py --savaux-symbol-count` 仍只是使用 manifest
边界的逐符号诊断，不应作为 Savaux 端到端结果。新的
`tools/evaluate_rfsr_savaux_ser.py` 已完成独立主链：

1. 从 held-out test 物理包未额外加噪的 1 MS/s received IQ 抽取 250 kS/s；
2. 先运行一次 RFSR，得到干净的 1 MS/s RFSR 输出；
3. 只在这条干净 RFSR 输出上执行 packet detection、frame location 和
   CFO/STO/SFO FrameSync，并冻结同步结果；
4. 对干净 RFSR 输出加入各档复 AWGN，所有 SNR 都复用第 3 步的 FrameSync；
5. 在带噪波形上做 8 个 explicit-header Savaux 判决，解析长度/CR/CRC/LDRO，
   再按冻结的 SFO 游标推进完成 payload Savaux；
6. 全部 SNR 的硬判决结束后才读取 reference metadata 并统计 SER。

SER 采用干净输出成功同步包上的条件口径：未通过 clean FrameSync 的包不进入
SER 分子或分母；固定同步后，加噪 Savaux 发生的头部无效、payload 截断或判决缺失仍按
符号错误统计。评分只覆盖能由 frame bytes 唯一确定的 payload symbol 前缀；
最后一个不完整 interleaver block 的 padding 取决于实际发射机实现，没有可靠
真值，因此不计入 SER 分母。SF12 的 Savaux Eq.36 分支谱
已改为等价的 chirp-convolution FFT 实现，避免构造每个 polyphase branch 的
致密 `N x N` 矩阵。

固定 `ota-max-groups=100`、seed `42` 时，物理包严格划分为 60/20/20；完整 UID
列表保存在 g100 checkpoint 同名 `_split_manifest.json` 中。

同一物理包的 ADC/polyphase views 不会跨 split。每次结果 JSON 都保存完整 UID、
三组数量、两两 overlap 和 `disjoint=true`。

微调入口会在 checkpoint 旁写入同名 `_split_manifest.json`。同名 checkpoint
续训前必须与 sidecar 的 seed、target 和三组 UID 一致，否则训练直接中止；
SER 评估也默认要求该绑定存在并匹配。当前 received-to-received checkpoint 的
sidecar 已补齐并通过验证。

可复现的 SER 网格命令：

```bash
python -B tools/evaluate_rfsr_savaux_ser.py \
  --ota-root "$OTA_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --output /root/autodl-tmp/rfsr-run/finetune/rfsr_savaux_ser_grid.json \
  --ota-max-groups 100 --ota-split-seed 42 \
  --device cuda --include-clean-output \
  --method rfsr_1msps --method native_1msps \
  --extra-snr-start-db -20 --extra-snr-stop-db -34 --extra-snr-step-db -0.25 \
  --noise-seed 20260728 --noise-seed-count 5
```

默认统计 RFSR payload SER；追加 `--ser-section all` 可统计 header+payload，重复
传入 `--method interpolation_1msps`、`--method native_1msps` 可加入配对对照。
旧结果文件
`/root/autodl-tmp/rfsr-run/finetune/rfsr_savaux_ser_heldout5_grid.json`
使用的是把同步失败包按全错计入的 v1 口径，不能与现在的条件 SER 混用。
当前输出 schema 为 `lora-rfsr-clean-sync-then-noisy-savaux-ser-v4`，同时保存
`packet_count`、`clean_synchronized_packets`、`clean_sync_success_rate`、
`ser_packet_count`、`symbol_errors/symbol_count` 和条件 `ser`。两支路使用原生
1 MS/s OTA 包功率定义同一噪声功率，并叠加相同复 AWGN。`aggregate_by_snr`
跨 seed 汇总，`paired_rfsr_vs_native` 只使用两边共同 clean-sync 成功的相同尝试。
正式曲线使用 100 包划分留下的 20 个 test 物理包和多个 AWGN seed。
