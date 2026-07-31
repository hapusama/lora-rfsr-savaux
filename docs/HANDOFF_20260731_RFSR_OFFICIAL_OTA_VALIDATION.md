# RFSR 官方 OTA 数据验证交接（2026-07-31）

## 下一阶段目标

下一步不再扩大当前合成实验，而是在磁盘允许、DR-NTU 数据站恢复后获取作者公开的
RFSR-OTA 数据集，完成一次真正的数据域匹配验证：

```text
官方 OTA IQ
  -> 官方 checkpoint 与官方/可审计预处理
  -> 普通 LoRa / 普通 FFT margin
  -> FrameSync
  -> Savaux
  -> branch GLS
```

最终要回答两个不同问题：

1. RF-SR 是否改善真实 OTA 弱包的同步、普通解码和普通 FFT；
2. RF-SR 是否恢复了可由 Savaux/GLS 利用的 polyphase branch 信息，而不只是去噪。

## 必须先纠正的实验表述

截至本 handoff，**没有运行官方 OTA 数据集测试**。

已经完成的是：

```text
官方 synthetic 生成器
  + 官方 synthetic checkpoint
  + 我们本地的 FrameSync/Savaux/GLS 评估器
```

旧结果中名为 `official_ota_rfsr` 的列，准确含义是：

> 把官方 OTA checkpoint 用在本地合成 250 kS/s 输入上的域外诊断。

它没有使用官方 OTA IQ，因此不能说明 OTA checkpoint 在真实 OTA 数据上的效果。

## 官方数据在哪里

上游 README 指向独立的 DR-NTU 数据记录，而不是 Git 仓库：

```text
Dataset: RFSR-OTA
DOI: 10.21979/N9/C6ABM3
License: CC BY-NC 4.0
Contents: 10,000 个 OTA LoRa IQ packet
PHY: SF12, BW 125 kHz, 16-byte payload, 8-symbol preamble
```

本地依据：

```text
third_party/rfsr/README.md:47-65
```

Git 仓库只跟踪代码和公开 checkpoint。项目 `.gitignore` 明确排除：

```text
*.cfile *.npy *.npz *.bin
data/raw/* data/processed/* data/results/*
```

所以“只有 checkpoint、没有数据集”是当前仓库的预期状态，不是数据已经隐藏在
`third_party/rfsr/checkpoints` 中。

当前仓库里找到的 5 个 `data/_file_source_staging/*.cfile` 是本地已有语料，不是
官方 10,000 包 RFSR-OTA 数据集，不能替代官方数据。

## 下载前的磁盘门槛

2026-07-31 当前空间：

```text
/root             可用约 18 GB
/root/autodl-tmp  可用约 28 GB
repo data/        约 61 MB
```

官方 archive 的压缩和解压体积尚未确认。下载前必须先读取数据记录和 `README.pdf`，
确认：

1. archive 数量、单文件大小、总压缩大小；
2. 解压后总大小；
3. 是否支持逐文件或分卷下载；
4. 文件 checksum；
5. IQ dtype、采样率、端序、每包/连续文件布局；
6. payload/packet ID/SNR/场景 metadata；
7. 作者使用的训练、验证和测试划分。

只有预计下载文件、解压副本、转换产物和结果总和不会逼近 28 GB 上限时才继续。
不要先完整下载再检查体积，也不要把 archive 放到 `/root` 根盘。

建议数据根目录：

```text
/root/autodl-tmp/rfsr-official-ota/
  archive/
  extracted/
  imported/
  results/
```

下载完成后至少记录：

```bash
df -h /root/autodl-tmp
du -sh /root/autodl-tmp/rfsr-official-ota/*
find /root/autodl-tmp/rfsr-official-ota/archive -type f -print0 | sort -z | xargs -0 sha256sum
```

## 官方代码中 synthetic 与 OTA 的区别

### Synthetic test

下列入口不需要离线数据集：

```text
third_party/rfsr/example/example.py
third_party/rfsr/rfsr/per.py --synth_nn
```

`per.py` 每轮调用 `encode()` 现场生成随机包、加入 AWGN、直接抽取到 250 kS/s，
因此只需要 synthetic checkpoint。

### OTA test/train

真实 OTA loader 需要本地存在：

```text
manifests/views.csv
OTA IQ
reference/target IQ
每包 metadata
```

本地 manifest loader 位于：

```text
third_party/rfsr/rfsr/nn/ota_dataset.py
```

它会在缺少 `views.csv`、OTA IQ、reference IQ 或 metadata 时直接报错。当前
vendored `per.py` 没有完整 OTA evaluation loop；`ota_nn` 只出现在绘图选项，
实际绘图实现仍只接受 `synth_nn`。不能假定下载完数据后运行
`python rfsr/per.py --ota_nn` 就能闭环。

## 当前公开 checkpoint

```text
# synthetic checkpoint
third_party/rfsr/checkpoints/
  model_model0v0_bs5_osf4_ds250_lr0.001_wd1e-05.pth
SHA256:
  e355c8279d77e150d762e3ff1052606cc9bcc6a9b97db0e2b6adbdffbabeeaa0

# 上游公开 OTA checkpoint
third_party/rfsr/checkpoints/
  model_model0v0lopenaltyhl_bs1_osf4_ds250_lr0.0001_wd1e-05_ota_dsf8.pth
SHA256:
  a4de5311d70a9c37618a89632d023abe46f847ba18b3d15fb439a513fcd0c398

# 本地历史 g24，不是上述公开 OTA checkpoint
third_party/rfsr/checkpoints/
  model_model0v0hl_bs1_osf4_ds250_lr0.0001_wd1e-05_ota_received_g24_dsf8.pth
SHA256:
  d28caf49a62f9ea5d27684e8dc24ebd07f0920fb5a1f3ee9d14dacb285dc4b90
```

上游 provenance：

```text
third_party/rfsr/UPSTREAM.md
base commit: 00c135947f855790f458fdc25ae9533c70d77849
```

公开 OTA checkpoint 在 unit-amplitude synthetic 输入上残差接近零，输出几乎等于
插值。这只能说明域不匹配；真正结论必须等待官方 OTA IQ。

## checkpoint 划分泄漏风险

公开 OTA checkpoint 没有本地 split sidecar。即使下载了 10,000 包，也不能随意
随机切出 20% 后称为“held-out test”，因为 checkpoint 可能已经在这些包上训练过。

数据到手后优先寻找：

1. `README.pdf` 中的官方 train/test 规则；
2. packet ID、场景或采集 session 的官方划分；
3. checkpoint 对应的训练清单或随机 seed；
4. 论文/代码中是否只报告整数据集重建而没有严格 held-out 划分。

如果找不到官方划分，允许使用 checkpoint 做**域匹配诊断**，但结果必须标注：

```text
unbound checkpoint diagnostic; training/test overlap unknown
```

此时不能声称泛化性能。`evaluate_rfsr_savaux_ser.py` 的
`--allow-unbound-checkpoint` 只负责允许机械运行，不会消除数据泄漏风险。

## 数据到手后的第一阶段：只验收，不评估

不要立即跑模型。先生成一份 inventory JSON/CSV，至少包含：

```text
relative_path
file_size
sha256
dtype
complex_sample_count
sample_rate_hz
SF/BW/CR/preamble
packet ID 或 payload
场景/位置/链路条件
官方 SNR 或可用噪声窗口
```

随机抽 2 到 5 个文件做只读检查：

```text
文件长度与文档一致
complex IQ 有限且非全零
功率尺度合理
能看到 LoRa preamble
普通 decoder 至少能处理强包
metadata 能关联 payload 或 packet ID
```

官方 archive 可能已经逐包切片，也可能不是本地 `views.csv` 合同。不要直接把
官方文件塞进 `tools/build_rfsr_ota_dataset.py`：该工具面向本项目的 UART 真值和
连续 USRP capture。应先检查官方布局，再写一个最小 import adapter，把官方字段
映射到项目所需 manifest，避免重新检测已经切好的包。

## 第二阶段：最小 smoke test

先只选 2 个强包和 2 个弱包，验证以下四条 1 MS/s 波形长度与同步：

```text
native_1msps
official_interpolation
official_synthetic_rfsr
official_ota_rfsr
```

输入合同必须是：

```text
官方 received 2 MS/s IQ
  -> 选择明确的低率抽取相位，得到 received 250 kS/s
  -> 不做 CFO/SFO/复增益/幅度归一化
  -> RF-SR 输出 1 MS/s
  -> FrameSync
```

不要把 ideal reference 或 synthetic clean 波形作为 RFSR 测试输入。

smoke test 必查：

```text
checkpoint hash
输入/输出长度
输入和输出 RMS
FrameSync 成功状态
CFO/STO/SFO
普通 FFT bin/margin
Savaux bin/margin
```

## 第三阶段：正式配对矩阵

第一轮只用官方 OTA 中原有噪声，不额外加 AWGN。按同一个物理包配对比较：

| 前端 | 普通 FFT | Savaux | Savaux+GLS | 完整 CRC |
|---|---:|---:|---:|---:|
| native 1 MS/s | 必须 | 必须 | 必须 | 必须 |
| 官方插值 | 必须 | 必须 | 必须 | 必须 |
| synthetic checkpoint | 必须 | 必须 | 必须 | 必须 |
| OTA checkpoint | 必须 | 必须 | 必须 | 必须 |

同时报告：

```text
packet 数与独立物理 packet 数
FrameSync 成功率
端到端 SER（同步失败整包计错）
同步成功条件下 SER
普通 FFT/Savaux/GLS peak margin
输出功率与相对增益
branch 协方差与平均相关性
完整 header/CRC 成功率
```

低 SNR 结论应优先按官方 OTA 的真实 SNR 分层，而不是先人工加噪。

## OTA 的 250 kS/s SNR 口径

当前 synthetic 审计已新增：

```text
clean_250k 与 noisy_250k 已知时：
SNR = 10 log10(Pclean / P(noisy-clean))
```

但真实 OTA 没有同一 channel realization 的 clean received 波形，不能使用上述相减
公式。官方 OTA 应优先使用作者 metadata 的 SNR；若需要本地复核，则必须固定一个
接收机估计器，例如：

```text
包前纯噪声窗估计 Pnoise
已同步 preamble/packet 有效窗估计 Psignal+noise
SNR = 10 log10(max(Psignal+noise-Pnoise, eps) / Pnoise)
```

或使用统一的 dechirped-preamble estimator。所有方法分层时必须基于同一条原始
250 kS/s received 输入的 SNR，不得按 RFSR 输出功率重新分组。

## 额外人工噪声的顺序

只有真实 OTA 原始条件跑通后，才考虑额外 AWGN：

```text
received 2 MS/s OTA
  -> 加额外噪声
  -> 抽取到 250 kS/s
  -> 在 250 kS/s 重新估计/记录输入 SNR
  -> RFSR
  -> 重新 FrameSync
  -> FFT/Savaux/GLS
```

`RFSR -> clean FrameSync -> 1 MS/s 后置噪声` 只保留为诊断。它会人为引入 8 条
独立高率噪声样本，且受不同前端输出增益影响，不能作为 branch diversity 主证据。

## 可复用评估入口

```text
tools/evaluate_rfsr_ota_decode.py
  完整 decode + 可选 Savaux/branch-GLS 诊断

tools/evaluate_rfsr_savaux_ser.py
  clean FrameSync 后置噪声诊断；不是 OTA 主结论入口

tools/evaluate_official_rfsr_synthetic_chain.py
  当前 synthetic go/no-go 工具；不能直接用于官方 OTA archive
```

转换成项目 manifest 合同后，可先用类似命令做 2 包机械 smoke test：

```bash
conda run -n grlora python tools/evaluate_rfsr_savaux_ser.py \
  --ota-root /root/autodl-tmp/rfsr-official-ota/imported \
  --checkpoint third_party/rfsr/checkpoints/model_model0v0lopenaltyhl_bs1_osf4_ds250_lr0.0001_wd1e-05_ota_dsf8.pth \
  --output /root/autodl-tmp/rfsr-official-ota/results/smoke_unbound.json \
  --allow-unbound-checkpoint \
  --limit 2 --device cuda --workers 1 \
  --method rfsr_1msps --method interpolation_1msps --method native_1msps \
  --include-clean-output --ser-section all
```

这条命令只验证 importer/loader/模型/同步能运行。因为 checkpoint 未绑定 split，
输出必须叫 `unbound`，不能作为最终性能数字。

正式主入口应优先使用 `evaluate_rfsr_ota_decode.py`，但在 importer 和官方 split
确认前不要编造最终命令。

## 当前 synthetic 审计状态

主脚本：

```text
tools/evaluate_official_rfsr_synthetic_chain.py
```

已完成的代码改动：

```text
模块说明、docstring 和算法注释已改为中文
pre_rfsr 在抽取到 250 kS/s 后计算实测 SNR
packet row 保存 signal/noise power 与 measured SNR
summary 保存 measured SNR 的 median/min/max
schema_version = 2
```

快速单测：5 项通过。

按用户要求，SNR 修改后没有重跑正式矩阵。因此当前旧结果：

```text
data/results/official_rfsr_synthetic_chain_20260730.json
SHA256:
  ba489a40c92bc557b87a03d53d3140524a73d3f715ffa017ee706f1fb1b948a9
```

仍是 schema v1，只包含目标注入 SNR，不包含新的 250 kS/s 实测字段。不要把旧 JSON
中的 `snr_db` 描述成实测 250 kS/s SNR。

旧 synthetic 结果的定性结论：

```text
官方 synthetic checkpoint 在合成域可能改善同步/去噪
尚未证明恢复 native polyphase branch diversity
后置同功率噪声结果被前端输出增益严重混杂
OTA checkpoint 在合成输入上近似退化为插值，不能外推到真实 OTA
```

## 正式 go/no-go 标准

### 支持“同步/去噪前端”

在无额外噪声的官方 OTA 弱包上，OTA checkpoint 相对插值稳定提高：

```text
FrameSync 成功率
完整 header/CRC 成功率
普通 FFT margin 或 SER
```

### 支持“恢复可合并 branch 信息”

必须进一步看到：

```text
Savaux 相对同一 RFSR 输出普通 FFT 的增益稳定存在
GLS 相对 Savaux 的增益跨 packet/SNR 可重复
RFSR branch 相关性和 margin 不只是后置白噪声制造的结果
结果不能只由同步成功包集合变化解释
```

如果官方 OTA checkpoint 能改善普通解码，但 Savaux/GLS 没有额外增益，结论应是：

```text
RF-SR 是去噪/同步前端，但没有恢复 Savaux/MRC 所需的原生 branch 多样性。
```

如果官方 OTA 上连普通解码和同步也无法改善，则停止
`RFSR -> Savaux -> GLS` 主线。

## 工作区注意事项

当前新增但未提交：

```text
docs/ANALYSIS_20260730_OFFICIAL_RFSR_SYNTHETIC_CHAIN.md
tools/evaluate_official_rfsr_synthetic_chain.py
weak_decoder/os_lora/tests/test_evaluate_official_rfsr_synthetic_chain.py
docs/HANDOFF_20260731_RFSR_OFFICIAL_OTA_VALIDATION.md
```

不要 reset、checkout 或覆盖这些文件。官方数据和转换产物应留在
`/root/autodl-tmp/rfsr-official-ota/`，不要加入 Git。
