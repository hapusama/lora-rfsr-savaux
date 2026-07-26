# 真实 USRP IQ 上的 Savaux / Current / Codec 对比记录

日期：2026-06-18

## 数据与评价口径

- IQ 数据目录：`gr-lora_sdr/data/USRP_IQ`
- 参数来自 `文件名描述.txt`：`samp-rate=500000`、`BW=125000`、`OS=4`、`sync-word=0x34`、`crc-mode=0`
- 文件名格式：`实验编号_走廊编号_位置编号_SF_TP_Preamble.bin`
- 真实 IQ 没有 byte/symbol ground truth，所以这里不算 SER/BER，只算 header-valid 包上的 CRC/PRR。

## 方法

- `traditional_fft`：单中心 FFT argmax。
- `current_selected`：当前 phase/threshold two-stage 选择器。
- `savaux_paper`：只实现论文 oversampled LoRa fixed-offset branch combining 的 baseline。
- `savaux_codec`：Savaux OSR soft evidence + LoRa codec/CRC beam；若 Savaux hard 已 CRC valid，则保留 Savaux hard。

## 主要输出

- 汇总表：`data/baseline_comparison/real_iq_crc_consolidated_summary.csv`
- 按 SF/TP 聚合：`data/baseline_comparison/real_iq_crc_consolidated_group_summary.csv`
- 真实 IQ 批处理 runner：`scripts/experiments/structured_paths/run_real_iq_batch_crc_probe.py`
- 单文件 probe：`scripts/experiments/structured_paths/run_real_iq_crc_probe.py`

## 聚合结果

| SF | TP | captures | header-valid | FFT | current | Savaux | Savaux+codec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 14 | 3 | 28 | 1.000 | 1.000 | 1.000 | 1.000 |
| 11 | 2 | 6 | 38 | 0.895 | 0.921 | 0.947 | 0.947 |
| 12 | 2 | 4 | 33 | 0.030 | 0.030 | 0.030 | 0.030 |
| 12 | 10 | 5 | 27 | 0.000 | 0.000 | 0.000 | 0.000 |
| 12 | 14 | 5 | 16 | 0.000 | 0.000 | 0.000 | 0.000 |

## 稳模式复核样本

`lab1_sf11_TP2/1_0_16_11_2_16.bin` 是目前最有区分度的真实 IQ 弱包文件。稳模式复核输出：

- 输出目录：`data/baseline_comparison/real_iq_crc_batch_sf11_weak_fullcheck`
- header-valid 包：4
- `traditional_fft`：0/4
- `multi_offset_argmax`：1/4
- `current_selected`：1/4
- `savaux_paper`：2/4
- `savaux_codec`：2/4

packet 级现象：

- packet 0：FFT/current 失败，Savaux paper 与 Savaux+codec 成功。
- packet 5：multi-offset/current/Savaux/Savaux+codec 成功，traditional FFT 失败。
- Savaux+codec 没超过 Savaux paper，是因为成功包里 Savaux hard 已经 CRC valid，codec 按保守策略直接保留。

## 仿真对比补充

AWGN 加噪仿真输出：`data/savaux_codec_policy_earliest_m12_m27`

mean-of-datasets 阈值：

- current selected：SER<=10% `-21.915 dB`，CRC>=80% `-20.424 dB`，CRC>=50% `-21.472 dB`
- Savaux paper：SER<=10% `-23.051 dB`，CRC>=80% `-21.443 dB`，CRC>=50% `-22.536 dB`
- Savaux+codec：SER<=10% `-23.467 dB`，CRC>=80% `-23.088 dB`，CRC>=50% `-24.206 dB`

相对增益：

- Savaux paper vs current：SER +1.135 dB，CRC80 +1.019 dB，CRC50 +1.064 dB
- Savaux+codec vs current：SER +1.552 dB，CRC80 +2.664 dB，CRC50 +2.734 dB
- Savaux+codec vs Savaux paper：SER +0.416 dB，CRC80 +1.645 dB，CRC50 +1.670 dB

## 结论

真实 IQ 目前多数样本处在两端：SF10/SF11 较强包全过，SF12 弱场景多数全不过。唯一稳定拉开差异的是 `1_0_16_11_2_16.bin`，其中 Savaux paper 的 OS branch 明确优于传统 FFT/current。Savaux+codec 在真实 IQ 上暂未额外超过 Savaux paper；它的优势主要在 AWGN 加噪仿真中体现为 CRC 阈值约 1.6 dB 继续下探。
