# scripts 解码链入口说明

本目录放置可直接运行的命令行入口。具体算法实现主要位于
`../weak_decoder/`，因此不是每个 Python 模块都要单独运行；通常由这里的编排脚本
按顺序调用多个算法模块。

当前默认链路分为两个命令级阶段：

```text
原始 complex64 IQ
  │
  ├─ 第一阶段：检测与同步
  │    preamble detection
  │      -> 起点细化
  │      -> sync word + SFD 帧定位
  │      -> CFO/STO/SFO frame sync
  │      -> sync.csv
  │
  └─ 第二阶段：普通 LoRa FFT demod
       读取 IQ + sync.csv
         -> 8 个 explicit-header symbols FFT
         -> 解析 header，确定 payload symbol 数量
         -> header + payload 逐 symbol dechirp + FFT
         -> fft_symbols.csv
```

第一阶段与第二阶段目前通过 CSV 交接，不会由一个命令自动连续执行。这样便于复用
耗时的同步结果，后续比较普通 FFT、OS-LoRa、GLS 和不同 baseline 时无需反复扫描整份 IQ。

## 推荐入口

### 1. 原始 IQ 到 frame sync

推荐从 `weakPacket_decoding` 目录运行包内入口：

```powershell
python -m weak_decoder.run_iq_frontend `
  --input "USRP_collector\data\branch4_fixed\high_snr\sf10_bw125_fs500_pre32_sw34_r001.bin" `
  --output "data\frontend\sf10_bw125_fs500_pre32_sw34_r001_sync.csv" `
  --sf 10 `
  --bw 125000 `
  --samp-rate 500000 `
  --center-freq 487700000 `
  --sync-word 0x34 `
  --preamble-len 32 `
  --win-chirps 4
```

这里的 `--preamble-len 32` 表示发射包中共有 32 个前导 upchirp；
`--win-chirps 4` 表示前导码检测的每个滑动窗口包含 4 个 chirp。检测器会对这
4 个 chirp 分别解调和 FFT，然后按 bin 做非相干功率累加：

```text
E[k] = sum_m |FFT_m[k]|^2,  m = 0, 1, 2, 3
```

在当前 `SF=10、BW=125 kHz、Fs=500 kS/s` 配置下，每个 chirp 为 4096 个采样点，
所以检测窗口长度为 `4 × 4096 = 16384` 个采样点。滑窗默认每次前进 1 个 chirp；
未显式指定连续窗口门限时，同步链会采用
`preamble_len - win_chirps + 1 = 29` 个峰值 bin 稳定的连续窗口作为粗检测条件。
`run_iq_frontend` 的 `--win-chirps` 默认值也是 4，但推荐在实验命令中显式写出，
便于仅凭命令和记录复现实验参数。

`weak_decoder.run_iq_frontend` 是一层 Branch4 参数封装，内部调用本目录的
`run_weak_sync_chain.py`：

```text
weak_decoder/run_iq_frontend.py
  -> scripts/run_weak_sync_chain.py
       -> weak_decoder.synchronization.preamble_detector.detect_preamble_runs()
       -> align_event_start()
       -> weak_decoder.synchronization.frame_locator.locate_frame_from_event()
       -> weak_decoder.synchronization.grlora_frame_sync.run_grlora_frame_sync_validation()
       -> 写出 sync CSV
```

同步 CSV 中供后续 FFT、OS-LoRa 和 GLS 使用的关键字段包括：

```text
grlora_framesync_valid
grlora_fine_payload_start_sample
grlora_cfo_int_est
grlora_cfo_frac_est
grlora_payload_sto_frac_est
grlora_sfo_hat
grlora_branch_sample_phases
grlora_branch_valid
```

历史字段名中的 `payload_start` 在 explicit-header 模式下实际表示 LoRa data region
起点，即 PHY header 第 0 个 symbol 的起点，不是 MAC payload 第 0 个 symbol。

### 2. frame sync 到普通 FFT demod

第二阶段需要显式运行：

```powershell
python scripts\run_header_first_demod.py `
  --input "USRP_collector\data\branch4_fixed\high_snr\sf10_bw125_fs500_pre32_sw34_r001.bin" `
  --sync-csv "data\frontend\sf10_bw125_fs500_pre32_sw34_r001_sync.csv" `
  --output "data\frontend\sf10_bw125_fs500_pre32_sw34_r001_fft_symbols.csv" `
  --frames-output "data\frontend\sf10_bw125_fs500_pre32_sw34_r001_fft_frames.csv" `
  --consistency-output "data\frontend\sf10_bw125_fs500_pre32_sw34_r001_payload_consistency.csv" `
  --sf 10 `
  --bw 125000 `
  --samp-rate 500000 `
  --frame-filter framesync-valid
```

内部调用关系：

```text
scripts/run_header_first_demod.py
  -> 读取 IQ 和 sync CSV
  -> weak_decoder.decoding.header_first_demod.demod_symbol_sequence()
       -> weak_decoder.chirp.build_downchirp()
       -> weak_decoder.chirp.dechirp_fft()
       -> FFT power argmax
  -> weak_decoder.decoding.header_first_demod.decode_explicit_header()
  -> 根据 header 确定 payload symbol 数量
  -> 再解调完整 header + payload symbol 序列
  -> 写出逐 symbol、逐 frame 和一致性 CSV
```

`fft_symbols.csv` 的核心字段：

```text
frame_index
stage                    header 或 payload
frame_symbol_index
stage_symbol_index
start_sample
raw_fft_bin              1024 点 FFT 数组下标
signed_fft_bin
symbol_value             gr-lora_sdr 映射后的 hard symbol
peak_power
peak_margin_db
```

做 FFT-bin SER 时比较 `raw_fft_bin` 或正式真值表中的 `groundtruth_fft_bin`，不要与
`symbol_value` 混用。payload 的映射约定为：

```text
symbol_value = (raw_fft_bin - 1) mod 2^SF
```

header 和 LDRO symbol 还会按 LoRa 规则进一步除以 4。

## 本目录脚本职责

| 脚本 | 用途 | 是否属于默认主链 |
|---|---|---|
| `run_weak_sync_chain.py` | 前导码检测、帧定位、CFO/STO/SFO 同步 | 是，第一阶段核心入口 |
| `run_header_first_demod.py` | explicit header 与 payload 的普通 FFT demod | 是，第二阶段入口 |
| `detect_weak_preamble.py` | 只运行前导码检测，适合独立调试门限 | 否，完整同步链已包含该步骤 |
| `plot_payload_peak_trends.py` | 绘制已导出 payload FFT peak 的幅度/相位 | 诊断工具 |
| `verify_payload_codec_alignment.py` | 用已知 symbol CSV 检查 payload codec 对齐 | 验证工具，不自动解码新 IQ |
| `experiments/` | ground truth、baseline、消融和历史实验入口 | 不属于默认主链 |

## weak_decoder 根模块与脚本的关系

`../weak_decoder/` 下的实现按同步与解调职责分组：

```text
chirp.py
  ├─ synchronization/
  │    preamble_detector.py
  │      -> frame_locator.py
  │           -> grlora_frame_sync.py
  └─ decoding/
       header_first_demod.py
         -> payload_codec.py
```

其中 `payload_codec.py` 已实现 payload 反交织、Hamming、去白化和 CRC 等功能，
但当前默认 `run_header_first_demod.py` 只借助显式 header 确定 payload symbol 数，
不会自动把 payload 一直解到 bytes/CRC。当前可信主链边界是逐 symbol FFT-bin 导出。

以下 `decoding/` 模块是保留的 baseline/消融实现，不会被普通 FFT 或 GLS 主线静默调用：

```text
adaptive_path_demod.py
structured_path_demod.py
timing_path_demod.py
```

## 固定帧 ground truth

原生 `gr-lora_sdr::fft_demod` 的逐 symbol Top-K peak 导出入口为：

```text
experiments/export_peak_groundtruth.py
```

它可以使用 `--consensus-output` 将多次重复发送汇总成一行一个 symbol 位置的真值表。
当前 Branch4 高 SNR 数据的正式结果位于：

```text
../data/groundtruth/branch4_fixed/high_snr/
  sf10_bw125_fs500_pre32_sw34_r001_fft_bin_groundtruth.csv
```

该文件包含 8 个 PHY header symbols 和 49 个 payload symbols。详细来源、字段定义与
交叉验证结果见同目录 `README.md`。

## 当前边界

```text
已经接通并实际验证：
IQ -> preamble -> frame locator -> frame sync -> 普通 FFT-bin

已实现但未接入默认入口：
payload codec、OS-LoRa/GLS、各类 baseline

当前不应默认宣称已验证：
payload bytes、dewhitening、最终 CRC 的端到端正确性
```
