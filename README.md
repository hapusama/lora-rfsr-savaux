# LoRa RF-SR + Savaux

本仓库是当前论文主线的统一工作区：

> 低采样率 LoRa IQ（默认 250 ksample/s）
> → RF Super Resolution 提升到 1 Msample/s
> → packet detection 与 CFO/STO/SFO FrameSync
> → Savaux 弱包解调
> → 与原生高采样率接收链做公平对照

当前联合主链暂不加入 GLS。GLS 代码只作为历史研究算法保留，不参与本文的
RFSR + Savaux 判决或 SER 统计。

## 仓库地图

| 路径 | 内容 |
| --- | --- |
| `weak_decoder/` | 当前同步/Savaux 实现，以及保留的 OS-LoRa 研究算法 |
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

## 2. RFSR、FrameSync 与 Savaux 串联

公开 checkpoint 已随上游源码保留在 `third_party/rfsr/checkpoints/`。运行集成测试：

```bash
python -m unittest \
  weak_decoder.os_lora.tests.test_rf_super_resolution_frontend \
  weak_decoder.os_lora.tests.test_litenap_error_modes -v
```

当前端到端入口直接从 held-out OTA 物理包开始：

```bash
python -B tools/evaluate_rfsr_savaux_ser.py \
  --ota-root data/reference_phy/rfsr_db \
  --ota-max-groups 100 --ota-split-seed 42 \
  --checkpoint "$CHECKPOINT" \
  --output data/results/rfsr_savaux_ser.json \
  --device cuda --include-clean-output \
  --method rfsr_1msps --method native_1msps \
  --extra-snr-start-db -20 --extra-snr-stop-db -34 --extra-snr-step-db -0.25 \
  --noise-seed 20260728 --noise-seed-count 5 --workers 0
```

入口先在未额外加噪的 RFSR 输出上调用
`weak_decoder/synchronization/single_packet.py` 完成包检测和 FrameSync，并冻结
该包的起点及 CFO/STO/SFO。随后才对 RFSR 的 1 MS/s 输出加入各档 AWGN，并调用
`weak_decoder/os_lora/system/synchronized_savaux.py` 复用同一份 FrameSync 完成
header-first Savaux 解调；加噪后不会重新同步。只有干净输出 FrameSync 成功的包
进入 SER 分母，参考 metadata 只在全部 SNR 的硬判决完成后用于评分。
`--workers 0` 会按容器实际可用 CPU 自动并行干净 FrameSync 和随后独立的
包级 Savaux；RFSR 仍保持单 GPU 进程。默认 JSON 只保存汇总，需逐 symbol 审计时
追加 `--save-symbol-details`。

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
在启动时一次性生成并缓存随机 raw payload、1 MSPS clean 标签和带扰动输入，
再按 4 倍抽取成 250 kSPS 输入，并只给输入依次添加 STO、CFO 和 AWGN。
后续每个 epoch 只改变这批固定样本的读取顺序。

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

这里的 `dataset_size` 是启动时一次性生成并缓存的随机 payload/噪声样本数；
后续每个 epoch 只会打乱同一批样本的读取顺序。新权重文件名带
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

### OTA 严格 RF-SR 微调与解码测试

OTA 微调默认使用 `--ota-target reference`：输入是低采样率接收 IQ，标签是与
该包配对的合成 PHY reference。reference 只在离线训练时提供监督；部署和评估时
仍是原始 250 kS/s IQ 先经过 RFSR，再由 FrameSync 估计同步参数，不会在 RFSR 前
调用 FrameSync 或向其提供 reference。`--ota-target received` 保留为复现历史
received-to-received checkpoint 的显式选项。

以下命令用固定 seed 从物理包中选择 100 个，并按物理包严格划成
60/20/20（训练/验证/测试）的 6:2:2；同一包的两个 ADC phase 与四个 q phase
绝不会跨 split。每个物理包共有 8 个降采样 view，因此真正参与梯度更新的是
`60 x 8 = 480` 个训练 view；validation 只用于早停，test 完全留给最终评测。

```bash
cd /root/autodl-tmp/rfsr-run/finetune
export PYTHONPATH=/root/lora-rfsr-savaux/third_party/rfsr

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

要使用全部 OTA 物理包，去掉 `--ota-max-groups 100`。训练完成后，使用
采集机的 GNU Radio/gr-lora 环境运行以下测试。它会从完整 LoRa 接收链的
packet detector 开始，比较 250 kSPS 原始输入、作者插值、RFSR 输出与原生
1 MSPS OTA，并以 metadata 中的完整 frame 和 CRC 统计成功数：

```bash
conda activate grlora
cd /root/lora-rfsr-savaux
export PYTHONPATH=/root/lora-rfsr-savaux/third_party/rfsr

python -B tools/evaluate_rfsr_ota_decode.py \
  --ota-root /root/autodl-tmp/lora-rfsr-savaux/data/reference_phy/rfsr_db \
  --ota-max-groups 100 --ota-split-seed 42 \
  --checkpoint "$CHECKPOINT" \
  --output /root/autodl-tmp/rfsr-run/finetune/ota_decode_test.json \
  --device cuda
```

可在上述测试命令加入 `--extra-snr-db -10`，在每个 held-out 高采样包上先加入
一个固定 seed 的公共 AWGN 实现，再由它抽取低采样输入。该噪声只用于测试，
不改变训练数据。注意这是旧的完整 CRC 链压力测试，噪声位于 RFSR 之前；不要
用它代替下面“clean FrameSync 后再加噪”的 Savaux SER 实验。

要统计“干净 RFSR 输出先同步、随后固定同步信息加噪”的 Savaux payload SER，
使用下面的专用入口。参考 metadata 只在全部 SNR 的硬判决完成后读取，不会向
同步或解调提供包边界、频偏或符号真值：

```bash
python -B tools/evaluate_rfsr_savaux_ser.py \
  --ota-root /root/autodl-tmp/lora-rfsr-savaux/data/reference_phy/rfsr_db \
  --ota-max-groups 100 --ota-split-seed 42 \
  --checkpoint "$CHECKPOINT" \
  --output /root/autodl-tmp/rfsr-run/finetune/rfsr_savaux_ser.json \
  --device cuda --include-clean-output \
  --method rfsr_1msps --method native_1msps \
  --extra-snr-start-db -20 --extra-snr-stop-db -34 --extra-snr-step-db -0.25 \
  --noise-seed 20260728 --noise-seed-count 5 --workers 0
```

输出 JSON 的
`split_manifest` 会保存三组完整 UID、数量、两两 overlap 列表及 `disjoint=true`，
因此每次测试结果都能独立审计数据泄漏。默认只跑 `rfsr_1msps`；可重复添加
`--method interpolation_1msps` 和 `--method native_1msps` 作为配对对照。
每个物理包只在不额外加噪的 RFSR 输出上运行一次 FrameSync；随后所有 SNR
条件都复用该结果。SER 只统计干净 FrameSync 成功包中、能由 frame bytes 唯一
确定的 payload symbol；干净同步失败包不进入 SER 分子或分母。最后一个不完整
interleaver block 的发射机私有 padding symbols 没有可靠真值，明确排除在分母外。
RFSR 和 native 使用原生 1 MS/s OTA 的同一个包级功率作为噪声标尺，并叠加完全
相同的复 AWGN 实现。`aggregate_by_snr` 汇总所有 seed；其中
`paired_rfsr_vs_native` 只比较两条支路共同 clean-sync 成功的相同包/seed 尝试。

OTA 微调现在会在 checkpoint 旁写入同名 `_split_manifest.json`，并在续训前
校验 seed、target 和三组物理包 UID；不一致时直接拒绝续训。SER 工具默认也
要求该 sidecar 与本次 `--ota-max-groups/--ota-split-seed` 完全匹配。只有确实
无法补齐来源的旧模型才使用 `--allow-unbound-checkpoint`，此时结果会明确标为
`verified=false`，不能当作严格 held-out 结论。

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
4. 跑通 held-out OTA 上的 RFSR + FrameSync + Savaux 条件 SER，再扩充物理包和噪声种子。
5. 只有在数据证明 RFSR 的输入契约或误差模式不适配时，才进入窗口训练、微调或模型改造。

当前研究交接以 `docs/HANDOFF_20260727.md` 为准；更早的 RF-SR 代码分析和
LiteNap 失败机制见 `docs/HANDOFF_20260726.md`。

## 许可证与来源

本仓库包含不同来源的代码：

- `weak_decoder/`、`noisy_iq/`、`scripts/`、`acquisition/` 及本仓库整合代码按 GPL-3.0 管理；
- `third_party/rfsr/` 保留其上游 MIT 许可证；
- RFSR 上游明确说明 `rfsr/PHY.py` 基于 SDR-LoRa，不属于其 MIT 授权范围。

详情见 `LICENSE`、`LICENSES/README.md` 和 `third_party/rfsr/LICENSE`。
