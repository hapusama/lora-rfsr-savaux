# weak_decoder 当前主线

当前研究主线是 `os_lora/` 中的过采样、多 branch、非均匀采样与 GLS 解调。
活动代码按照下面三层组织：

```text
raw complex64 IQ
  -> synchronization/preamble_detector.py       前导码检测
  -> synchronization/frame_locator.py           sync word + SFD 帧定界
  -> synchronization/grlora_frame_sync.py       CFO/STO/SFO 与多 branch 同步
  -> decoding/header_first_demod.py              传统 FFT header/payload 参考链
  -> os_lora/                   OS-LoRa、GLS 与低复杂度实现
```

`chirp.py` 是同步、解调和 baseline 共用的信号工具。`decoding/payload_codec.py`
提供标准 LoRa PHY codec；`decoding/` 下的 path demod 模块作为论文 baseline 或
诊断对照保留，不属于 GLS 主线。

## Branch4 固定帧配置

固件当前参数集中记录在 `branch4_profile.py`：

```text
中心频率          487.7 MHz
SF                10
带宽              125 kHz
编码率            4/7
前导码            32 symbols
sync word         0x34（public network）
PHY header        explicit
PHY CRC           enabled
应用数据          20 bytes
PHY payload       33 bytes
FCnt              1（固定）
TX power          2 dBm
发射周期          3 s
采集采样率        500 ksample/s（OSR=4）
```

生成推荐 IQ 文件名：

```powershell
python -m weak_decoder.branch4_profile --condition high_snr --run 1
```

输出示例：

```text
high_snr/sf10_bw125_fs500_pre32_sw34_r001.bin
```

数据条件改由 `high_snr`、`low_snr`、`noise_only`、`interference` 子目录区分，
使用 `r001`、`r002` 区分重复采集。文件名只保留解码所需的 SF、带宽、采样率、
前导码长度和 sync word；其余条件见数据集 README，实际 UHD 参数以旁边的 `.bin.json` 为准。

## 从 bin 运行 FFT 前端

在 `weakPacket_decoding` 目录执行：

```powershell
python -m weak_decoder.run_iq_frontend `
  --input "USRP_collector\data\branch4_fixed\high_snr\sf10_bw125_fs500_pre32_sw34_r001.bin" `
  --max-packets 10
```

Branch4 参数已经作为默认值，不依赖从文件名猜 SF 或 preamble。默认同步结果写到：

```text
data/frontend/<输入文件名>_sync.csv
```

该 CSV 的 `grlora_fine_payload_start_sample`、CFO、STO、SFO 和 branch 同步字段，
就是后续传统 FFT demod 或 `os_lora` demod 的公共输入。

如果需要先验证传统 FFT header：

```powershell
python scripts/run_header_first_demod.py `
  --input "USRP_collector\data\<capture>.bin" `
  --sync-csv "data\frontend\<capture>_sync.csv" `
  --output "data\frontend\<capture>_symbols.csv" `
  --frames-output "data\frontend\<capture>_frames.csv" `
  --sf 10 --bw 125000 --samp-rate 500000
```

## 已退出活动代码的方向

早期 phase-guided、phase consistency、candidate pruning、symbol-phase two-stage、
codec/CRC beam search 已从活动源码中删除。它们不再参与同步、baseline 或 OS-LoRa/GLS。
研究结论和历史说明仍保留在 `doc/history/`、`notes/` 与 Git 历史中。
