# LoRa RF-SR + Savaux

本仓库是当前论文主线的统一工作区：

> 低采样率 LoRa IQ（默认 250 ksample/s）
> → RF Super Resolution 提升到 1 Msample/s
> → Savaux / branch-GLS 弱包解调
> → 与原生高采样率接收链做公平对照

现在优先做数据采集、RFSR 复现和五臂实验，不提前改神经网络结构。遇到可复现的失败模式后，再决定是否需要训练或改模型。

## 仓库地图

| 路径 | 内容 |
| --- | --- |
| `weak_decoder/` | Savaux、OS-LoRa、同步、GLS 和实验代码 |
| `third_party/rfsr/` | RFSR 上游源码、公开 checkpoint 和原始结果 |
| `acquisition/` | USRP B210 IQ 采集脚本与 GRC 流图 |
| `noisy_iq/` | IQ 检测和功率分析工具 |
| `scripts/` | 弱包处理与同步入口 |
| `data/` | 仅存数据规范和 manifest；大文件不进入 Git |
| `docs/HANDOFF_20260727.md` | 当前真实数据采集配置、预训练结果、阻塞点和下一步 |
| `docs/research/` | 从旧工作区收拢的研究记录 |

原目录 `gr-lora_sdr/weakPacket_decoding` 和 `RFSuperResolution` 均未修改或删除。

## 1. 环境

Python 3.10 或 3.11 均可。GPU 服务器建议直接选已经配置好 CUDA/PyTorch 的镜像，再安装本项目：

```bash
git clone <your-repository-url>
cd lora-rfsr-savaux
python -m pip install -U pip
python -m pip install -e ".[rfsr]"
```

本机 Windows：

```powershell
Set-Location D:\Desktop\proj\lora-rfsr-savaux
python -m pip install -e ".[rfsr]"
```

PyTorch/CUDA 版本应以服务器镜像和显卡驱动为准；不要为了本仓库强行覆盖一个已经可用的 CUDA 环境。

## 2. 先验证 RFSR 与 Savaux 串联

公开 checkpoint 已随上游源码保留在 `third_party/rfsr/checkpoints/`。运行集成测试：

```bash
python -m unittest \
  weak_decoder.os_lora.tests.test_rf_super_resolution_frontend \
  weak_decoder.os_lora.tests.test_litenap_error_modes -v
```

五臂单符号探针：

```bash
python -m weak_decoder.os_lora.experiments.probe_rfsr_savaux \
  --input-low-iq data/raw/example_low_250k.bin \
  --start-low 0 \
  --sf 12 \
  --bw 125000 \
  --input-rate 250000 \
  --output-rate 1000000 \
  --snr-db -15 \
  --device cuda \
  --native-high-iq data/raw/example_high_1m.bin \
  --start-high 0 \
  --output data/results/probe.json
```

`--rfsr-repo` 现在是可选项；默认使用仓库内的 `third_party/rfsr`。探针比较：

1. 低采样率 Savaux；
2. 传统插值 + Savaux；
3. RFSR + Savaux；
4. 原生高采样率 Savaux；
5. 原生高采样率 branch-GLS。

这只是串联 smoke test，不是最终 PER 曲线。正式实验还需要可靠的低/高采样配对和真值。

## 3. 数据不要上传进 Git

默认目录：

```text
data/
├── raw/          # 原始 complex64 IQ，只读
├── processed/    # 对齐、切帧或缓存后的数据
├── results/      # 可再生成的大型 CSV/图
└── manifests/    # 可提交的小型元数据清单
```

`.bin`、`.npy`、`.npz` 以及上述三个数据目录都已忽略。每次采集至少保留：

- IQ 文件；
- 同名 `.bin.json` 采集元数据；
- `data/manifests/` 中的一行记录；
- 文件大小和 SHA-256。

本机到服务器优先用断点续传，不要经 Git：

```bash
rsync -avP data/raw/ user@server:/workspace/lora-rfsr-savaux/data/raw/
```

Windows 没有 `rsync` 时可用 `scp`；数据更大时建议对象存储或租用平台的数据盘。先按采集批次压缩/上传，服务器端校验 SHA-256 后再训练，不必一次搬完整数据湖。

详细约定见 [data/README.md](data/README.md)。

## 4. 当前实验顺序

1. 固定 PHY 参数和采集增益，采集高 SNR、低 SNR、noise-only。
2. 得到能够对齐的 250 ksample/s 与 1 Msample/s 数据及符号真值。
3. 在未改模型的情况下复现 RFSR 官方插值和 checkpoint。
4. 跑通五臂探针，再扩成 packet/SER/PER 评估。
5. 只有在数据证明 RFSR 的输入契约或误差模式不适配时，才进入窗口训练、微调或模型改造。

当前研究交接以 `docs/HANDOFF_20260727.md` 为准；更早的 RF-SR 代码分析和
LiteNap 失败机制见 `docs/HANDOFF_20260726.md`。

## 许可证与来源

本仓库包含不同来源的代码：

- `weak_decoder/`、`noisy_iq/`、`scripts/`、`acquisition/` 及本仓库整合代码按 GPL-3.0 管理；
- `third_party/rfsr/` 保留其上游 MIT 许可证；
- RFSR 上游明确说明 `rfsr/PHY.py` 基于 SDR-LoRa，不属于其 MIT 授权范围。

详情见 `LICENSE`、`LICENSES/README.md` 和 `third_party/rfsr/LICENSE`。
