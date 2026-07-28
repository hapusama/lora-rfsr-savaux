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
| `data/` | 项目内唯一数据根目录；大 IQ 不进入 Git，但服务器目录副本必须包含它们 |
| `docs/HANDOFF_20260727.md` | 当前真实数据采集配置、预训练结果、阻塞点和下一步 |
| `docs/research/` | 从旧工作区收拢的研究记录 |

运行主线不依赖仓库旁边的 `gr-lora_sdr/` 或 `RFSuperResolution/`；
RF-SR 源码和 checkpoint 已放在 `third_party/rfsr/`。

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
Set-Location .\lora-rfsr-savaux
python -m pip install -e ".[rfsr]"
```

PyTorch/CUDA 版本应以服务器镜像和显卡驱动为准；不要为了本仓库强行覆盖一个已经可用的 CUDA 环境。

### 只上传这个目录时的边界

“无外部依赖”在这里指没有兄弟源码仓库、兄弟数据目录或作者机器绝对路径依赖。
服务器仍需安装 Python、NumPy/SciPy 和与其 CUDA 匹配的 PyTorch。GNU Radio、
UHD 和 `gnuradio.lora_sdr` 只属于采集机上的 `detect` 阶段，不属于服务器
associate/trim/train 阶段。

大 IQ 被 `.gitignore` 排除，所以只执行 `git clone` 不会带上它。应直接复制
或 `rsync` 整个 `lora-rfsr-savaux/` 目录；复制前在采集机运行：

```bash
python tools/check_server_bundle.py --mode preprocess
```

trim 完成后、开始训练前运行：

```bash
python tools/check_server_bundle.py --mode train
```

预检会拒绝项目外 capture、绝对运行时 manifest 路径、缺失 reference、
缺失 `detections.csv`、缺失 OTA 文件或缺失 Python 依赖。

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

### 用 raw 33-byte frame 做合成预训练

先在项目根目录生成一次 clean reference 语料：

```powershell
python tools\generate_reference_phy.py `
  --uart-log data\raw\packet_reference.txt `
  --output-root data\reference_phy
```

raw-frame 的 symbol 编码和 IQ 调制入口位于
`third_party/rfsr/rfsr/PHY.py::encode_raw_phy()`；生成脚本和预训练共用这一
实现。训练数据集位于 `third_party/rfsr/rfsr/nn/dataset.py`。默认只读取
metadata 中来自 `packet_reference.txt` 的 PHY 参数和 33-byte 长度，然后
在每次取样时随机生成新的 raw payload 和 1 MSPS clean 标签，再按 4 倍抽取
成 250 kSPS 输入，并只给输入依次添加 STO、CFO 和 AWGN。

### 从连续 USRP cfile 构建 OTA 数据集

`tools/build_rfsr_ota_dataset.py` 实现
`catalog -> detect -> associate -> trim -> validate` 流程。所有持久化输出
都被限制在 `data/` 下，默认数据集根目录为
`data/reference_phy/rfsr_db/`。

新采集先生成规范的原始文件名和 JSON sidecar：

```powershell
python tools\build_rfsr_ota_dataset.py init-capture `
  --experiment-id 1 `
  --location-id lab1 `
  --condition lowsnr
```

正式连续文件名包含 experiment/session/location/condition、SF/BW/采样率、
preamble、sync word、CR/CRC、中心频率和 RX gain。采集机先只运行需要
GNU Radio 的检测：

```powershell
python tools\build_rfsr_ota_dataset.py detect `
  --capture data\raw\ota\rxcap_exp001_sess000_loclab1_condlowsnr_run000_sf12_bw125000_fs2000000_pre16_sw12_cr48_crc1_fc487700000_rxg20.cfile
```

结果写入项目内
`data/reference_phy/rfsr_db/manifests/captures/<capture_uid>/detections.csv`。
随后复制整个目录到服务器，运行明确不会导入 GNU Radio 的入口：

```bash
python tools/build_rfsr_ota_dataset.py server \
  --capture data/raw/ota/rxcap_exp001_sess000_loclab1_condlowsnr_run000_sf12_bw125000_fs2000000_pre16_sw12_cr48_crc1_fc487700000_rxg20.cfile \
  --usrp-log data/raw/ota/rxcap_exp001_sess000_loclab1_condlowsnr_run000_sf12_bw125000_fs2000000_pre16_sw12_cr48_crc1_fc487700000_rxg20.usrp.txt
```

每个物理 packet 只落盘两个 1 MS/s OTA phase；每个 phase 的四个
250 kS/s 输入由 `manifests/views.csv` 枚举并由 Dataset 动态 `[q::4]`
生成。完整设计、旧文件导入命令和字段说明见
[`tools/trimm.md`](tools/trimm.md)。

合成预训练命令：

```powershell
Set-Location .\third_party\rfsr
$env:PYTHONPATH = (Get-Location).Path

python -B rfsr\nn\nn.py `
  --model model0v0 `
  --batch_size 1 `
  --osf 4 `
  --dsf 8 `
  --dataset_size 250 `
  --num_epochs 100 `
  --synthetic-source reference_phy `
  --reference-root ..\..\data\reference_phy `
  --snr-min-db -22 `
  --snr-max-db 10 `
  --cfo-min-hz -35000 `
  --cfo-max-hz 35000 `
  --sto-initial-min-chips -0.5 `
  --sto-initial-max-chips 0.5 `
  --sto-slope-min-chips-per-symbol -0.05 `
  --sto-slope-max-chips-per-symbol 0.05
```

这里的 `dataset_size` 是每个 epoch 的随机 payload/噪声样本数；同一个 index
在下一次访问时也会重新生成 payload。新权重文件名带
`_synthref_random`，不会与官方 `PHY.py` 私有头路径的 checkpoint 混用。
需要复跑官方合成基线时传 `--synthetic-source upstream`。

需要把已有 120 条固定 cfile 作为消融对照时传：

```powershell
--fixed-reference-payloads
```

固定 SNR 训练时使用单个参数，它会覆盖区间参数：

```powershell
--snr-db -15
```

同样可以用 `--cfo-hz 31110` 固定 CFO；不传时默认每个输入样本在
`-35 kHz～+35 kHz` 内独立抽取 CFO。CFO 只加入低采样率输入 `x`，
高采样率标签 `y` 和离线 reference cfile 始终保持 `cfo_hz=0`。

STO 也只加入输入 `x`。每条样本独立抽取初始 `τ₀` 和逐符号斜率，
默认范围分别是 `[-0.5, 0.5] chip` 与
`[-0.05, 0.05] chip/symbol`。实现按 Hi²LoRa 的底层采样模型
`i -> i - τ_s` 对完整波形做分数采样，因此会保留 LoRa chirp 回绕点
前后的相位变化。消融时传：

```powershell
--no-sto
```

无论是否启用 STO，高采样率标签 `y` 都保持零 STO；checkpoint 文件名会
加入 STO 范围或 `_stonone`，避免不同实验互相续训。

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
