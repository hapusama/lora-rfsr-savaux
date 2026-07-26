# weakPacket_decoding

这个目录用于真实 LoRa 弱包的离线检测、同步和过采样解调。当前主线是：

```text
raw complex64 IQ
  -> 弱前导码检测
  -> sync word + SFD 帧定界
  -> gr-lora 风格 CFO/STO/SFO frame sync
  -> 多个过采样 branch 的符号观测
  -> os_lora 非均匀采样 / GLS 合并
  -> 标准 LoRa PHY hard decision 与 SER/CRC 评估
```

当前重点是把 FFT demod 之前的前端同步做稳定。早期 phase-guided、
phase-consistency、candidate-pruning、symbol-phase two-stage 和 codec/CRC beam
search 已退出活动代码；相关研究记录仍可在 `doc/history/`、`notes/` 和 Git 历史中查看。

## 目录结构

```text
weak_decoder/
  branch4_profile.py       当前 STM32 固定帧实验参数与文件名生成
  chirp.py                 同步、解调与 baseline 共用的 chirp/FFT 工具
  run_iq_frontend.py       标准 raw IQ 前端入口
  synchronization/         检测、帧定位与 CFO/STO/SFO 同步
    preamble_detector.py
    frame_locator.py
    grlora_frame_sync.py
  decoding/                传统 codec 与诊断解调实现
    header_first_demod.py
    payload_codec.py
    adaptive_path_demod.py
    structured_path_demod.py
    timing_path_demod.py
  os_lora/                 当前 OS-LoRa/GLS 主线
  baselines/               保留的论文 baseline

scripts/
  run_weak_sync_chain.py   完整同步链实现与详细诊断输出
  run_header_first_demod.py
  detect_weak_preamble.py

USRP_collector/
  collect_usrp_iq.py
  usrp_iq_collector.grc
  data/
```

## Branch4 固定帧参数

参数来自：

```text
LoraSTMacL1_2019.03.28_修改main函数_实现classA_通用版(Branch4)/apps/main.c
```

```text
RF              487.7 MHz
SF              10
BW              125 kHz
CR              4/7
preamble        32 symbols
sync word       0x34
header          explicit
PHY CRC         enabled
App payload     20 bytes
PHY payload     33 bytes
FCnt            1（固定）
TX power        2 dBm
TX period       3 s
IQ sample rate  500 ksample/s（OSR=4）
```

生成推荐文件名：

```powershell
python -m weak_decoder.branch4_profile --condition high_snr --run 1
```

```text
high_snr/sf10_bw125_fs500_pre32_sw34_r001.bin
```

采集条件建议使用：

```text
high_snr      高 SNR 固定帧，用于建立 FFT-bin ground truth
low_snr       低 SNR 固定帧
noise_only    发射机关闭，估计真实有色噪声协方差
interference  发射机与干扰源同时开启
```

文件名只保留解码前端需要的 SF、带宽、采样率、前导码长度和 sync word；
频率、CR、payload、FCnt、TX power、RX gain 等实验条件记录在
`USRP_collector/data/branch4_fixed/README.md` 和采集脚本生成的 `.bin.json` 中。

## 1. 采集 IQ

在 RadioConda 中运行：

```powershell
Set-Location "D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"

python USRP_collector\collect_usrp_iq.py `
  --output USRP_collector\data\branch4_fixed\high_snr\sf10_bw125_fs500_pre32_sw34_r001.bin `
  --duration 120 `
  --gain 20 `
  --device-args "serial=YOUR_USRP_SERIAL"
```

脚本同时生成 `.bin.json`，保存实际 UHD 参数。正式实验中不要只依赖文件名。

## 2. 前导码检测与 frame sync

在 `gr-lora` 环境中运行：

```powershell
Set-Location "D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"

python -m weak_decoder.run_iq_frontend `
  --input "USRP_collector\data\branch4_fixed\high_snr\sf10_bw125_fs500_pre32_sw34_r001.bin" `
  --max-packets 20
```

Branch4 参数已经内置，不再从文件名位置猜 SF 和 preamble。默认输出：

```text
data/frontend/<capture>_sync.csv
```

关键字段包括：

```text
grlora_fine_payload_start_sample
grlora_cfo_int_est / grlora_cfo_frac_est
grlora_payload_sto_frac_est
grlora_sfo_hat
grlora_branch_sample_phases
grlora_branch_valid
```

这些字段定义了 FFT/OS-LoRa demod 的输入边界和各 branch 同步状态。

## 3. 传统 FFT 参考链

在接入 GLS 前，可以先验证传统 header-first FFT：

```powershell
python scripts\run_header_first_demod.py `
  --input "USRP_collector\data\<capture>.bin" `
  --sync-csv "data\frontend\<capture>_sync.csv" `
  --output "data\frontend\<capture>_symbols.csv" `
  --frames-output "data\frontend\<capture>_frames.csv" `
  --sf 10 --bw 125000 --samp-rate 500000
```

## 4. OS-LoRa / GLS

当前实现已经按职责拆分：

```text
weak_decoder/os_lora/system/       可复用的在线解码算法
weak_decoder/os_lora/experiment_support/  实验共享基础设施
weak_decoder/os_lora/experiments/  离线评估、诊断、标定与绘图
weak_decoder/os_lora/doc/          算法与实验文档
```

详细的模块职责、导入方式和实验入口见
`weak_decoder/os_lora/README.md`。实验入口之间不得互相导入，删除任意实验脚本
不会影响其他入口的导入。

算法文档：

```text
weak_decoder/os_lora/doc/非均匀采样.md
weak_decoder/os_lora/doc/GLS.MD
```

## Baselines

以下内容保留，不属于清理范围：

```text
weak_decoder/baselines/loratrimmer/
weak_decoder/baselines/savaux_oversampled/
weak_decoder/baselines/symfec/
weak_decoder/baselines/unichirp/
weak_decoder/decoding/adaptive_path_demod.py
weak_decoder/decoding/structured_path_demod.py
weak_decoder/decoding/timing_path_demod.py
```

它们用于论文对比、负结果或消融实验，不应混入 GLS 权重估计本身。
