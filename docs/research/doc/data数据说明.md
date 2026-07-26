# weakPacket_decoding/data 数据说明

本文说明 `weakPacket_decoding/data/` 下当前数据的来源、文件命名、字段含义，以及它和 `noisy_iq/`、`gr-lora_sdr`、后续弱包解码原型之间的关系。

## 1. 数据来源

当前主要数据位于：

```text
weakPacket_decoding/data/noisy_iq/
```

这些数据由 `weakPacket_decoding/noisy_iq/` 包生成，常用入口是：

```powershell
python gr-lora_sdr/weakPacket_decoding/scripts/experiments/make_noisy_iq.py `
  -i gr-lora_sdr/data/USRP_IQ/0_0_0_10_14_8.bin `
  --samp-rate 500000 --bw 125000 --sync-word 0x34 `
  --noise-power-db 10 15 25 30 `
  --preamble-len 8 --overwrite
```

生成流程是：

1. 读取一个干净的 raw complex64 LoRa IQ 捕获。
2. 解析文件名或命令行参数得到 SF、带宽、采样率、前导码长度等参数。
3. 估计一个参考功率 `noise_reference_power`。
4. 按多个 `noise_power_db_relative` 档位加入复高斯白噪声。
5. 对每个 noisy IQ 文件调用当前 conda 环境 `gr-lora` 中已编译的 `gr-lora_sdr` 接收链，获得包检测、粗同步、header、payload、CRC、SNR 等测量结果。
6. 写出 `.bin`、逐文件 `.json`、整组 summary `.csv/.json`。

也就是说，`data/noisy_iq/` 不是手工标注数据集，而是“干净 IQ + 可复现 AWGN + gr-lora_sdr 测量 metadata”的实验产物。

## 2. 目录命名

每个子目录对应一个原始 IQ 捕获的 stem，例如：

```text
data/noisy_iq/0_0_0_10_14_8/
data/noisy_iq/0_0_0_10_14_16/
data/noisy_iq/0_0_0_10_14_32/
```

原始文件名前六段约定为：

```text
experiment_id_corridor_id_position_id_sf_tx_power_dbm_preamble_len.bin
```

以 `0_0_0_10_14_8.bin` 为例：

| 字段 | 值 | 含义 |
| --- | ---: | --- |
| `experiment_id` | 0 | 实验编号 |
| `corridor_id` | 0 | 走廊/场景编号 |
| `position_id` | 0 | 位置编号 |
| `sf` | 10 | LoRa spreading factor |
| `tx_power_dbm` | 14 | 发射功率，单位 dBm |
| `preamble_len` | 8 | 前导码 upchirp 数 |

`noisy_iq.capture.parse_capture_metadata()` 会从文件名中解析这些字段。命令行显式传入的参数优先级更高。

## 3. `.bin` 文件

文件名示例：

```text
0_0_0_10_14_8_noise_rel_25p0dB.bin
```

含义：

| 片段 | 含义 |
| --- | --- |
| `0_0_0_10_14_8` | 原始干净 IQ 文件 stem |
| `noise_rel_25p0dB` | 加入噪声功率相对参考功率为 `+25.0 dB` |
| `.bin` | raw `numpy.complex64` / GNU Radio `gr_complex` IQ 样本 |

注意：`noise_rel_25p0dB` 不是目标 SNR。它表示：

```text
added_noise_power = noise_reference_power * 10^(25.0 / 10)
```

复噪声按 I/Q 两路平均分配，所以 JSON 中还有：

```text
added_noise_sigma_per_iq_component = sqrt(added_noise_power / 2)
```

当前已有 `10/15/25/30 dB` 四个加噪档位。正值越大表示加入的噪声越强，实际 `gr-lora_sdr` 估计 SNR 会随之降低。

## 4. 单个噪声点 `.json`

文件名示例：

```text
0_0_0_10_14_8_noise_rel_25p0dB.json
```

这是同名 `.bin` 的完整 metadata。关键字段如下：

| 字段 | 含义 |
| --- | --- |
| `input_file` | 原始干净 IQ 文件 |
| `output_file` | 当前 noisy IQ `.bin` 文件 |
| `format` | 固定为 raw complex64 / gr_complex |
| `noise_power_db_relative` | 本次加入噪声相对参考功率的 dB 值 |
| `noise_reference_power` | 加噪参考功率，线性功率 |
| `noise_reference_power_db` | 加噪参考功率的 dB 表示 |
| `noise_reference_mode` | 参考功率模式，当前常见为 `total` |
| `added_noise_power` | 实际加入的复噪声总功率 |
| `seed` | 随机种子；默认各噪声档位复用同一套单位噪声再缩放 |
| `processed_samples` | 处理的 IQ 样本数 |
| `capture_metadata` | 从文件名和 CLI 解析出的捕获参数 |
| `groundtruth` | 从干净输入自动解码出的期望 payload 列表 |
| `args` | 生成与测量时使用的命令行参数 |
| `power_estimate` | 参考功率估计细节 |
| `grlora_snr_measurement` | 当前 noisy 文件经 gr-lora_sdr 接收链得到的测量结果 |

`groundtruth.expected_payload_hex` 是干净输入成功解码出的 payload，后续 noisy 点会用它判断正确包数、误包数和 BER。

## 5. `grlora_snr_measurement`

这是后续弱包解码原型最重要的部分，因为它提供了 `gr-lora_sdr` 的包检测和粗同步结果。主要字段：

| 字段 | 含义 |
| --- | --- |
| `detected_packets` | gr-lora_sdr 检测到的包数 |
| `decoded_payload_packets` | 成功给出 payload bytes 的包数 |
| `grlora_snr_db_summary` | gr-lora_sdr 内部 SNR 估计统计量 |
| `payload_check` | 与 groundtruth 比较后的正确包、误包、漏检、BER |
| `packet_measurements` | 每个检测包的详细 metadata |

`packet_measurements` 每一项对应一个检测到的包。常用字段：

| 字段 | 含义 |
| --- | --- |
| `frame_count` | gr-lora_sdr 帧计数 |
| `sf` / `bw` / `sample_rate` | 接收参数 |
| `samples_per_symbol` | 一个 LoRa 符号对应的输入 IQ 样本数 |
| `preamble_len` | 前导码长度 |
| `start_sample` / `end_sample` | gr-lora_sdr 发布的 preamble/sync/SFD 对齐范围 |
| `packet_start_sample` / `packet_end_sample` | 根据 header 和 LoRa airtime 估计的完整包范围 |
| `payload_symbols` | header 后 payload 部分的符号数 |
| `grlora_snr_db` | 当前包的 gr-lora_sdr SNR 估计 |
| `cfo` / `sto` / `sfo` | gr-lora_sdr 估计的 CFO、STO、SFO |
| `netid1` / `netid2` | sync word 相关检测结果 |
| `cr` / `pay_len` / `crc` / `ldro_mode` | PHY header 解出的参数 |
| `header_err` | header 校验错误标志，0 表示 header 可用 |
| `decoded_payload_hex` | gr-lora_sdr 输出的 payload |
| `crc_valid` | PHY CRC 是否通过 |

这些字段可以作为可复现实验缓存使用。新的 Python 弱包解码原型默认直接读取同名 `.bin`，现场复用 `examples/lora_file_RX.py` 里的 gr-lora_sdr 离线接收链生成 packet window；只有显式传入 `--metadata-json` 时，才会跳过检测并复用这里保存的 `packet_measurements`。

## 6. Summary 文件

每个目录下有：

```text
*_noise_sweep_summary.csv
*_noise_sweep_summary.json
```

它们是一组加噪实验的总表。一行对应 clean 输入或一个 noisy 档位。常看字段：

| 字段 | 含义 |
| --- | --- |
| `kind` | `clean` 或 `noisy` |
| `step_index` | sweep 序号 |
| `noise_power_db_relative` | 加噪档位 |
| `added_noise_power` | 实际加噪功率 |
| `detected_packets` | 检测包数 |
| `decoded_payload_packets` | 解出 payload 的包数 |
| `expected_packet_count` | groundtruth 包数 |
| `crc_valid_packets` | CRC 正确包数 |
| `correct_payload_packets` | payload 完全匹配包数 |
| `wrong_payload_packets` | 解出但 payload 错误的包数 |
| `missed_detection_packets` | 期望包数减检测包数 |
| `bit_error_count` / `ber` | 与期望 payload 比较后的误比特数和 BER |
| `grlora_snr_median` | gr-lora_sdr SNR 中位数 |

例如 `0_0_0_10_14_8` 这组数据中，clean 和 `10/15 dB` 档位还能全解，`25 dB` 开始漏检，`30 dB` 只剩少量检测且 CRC 失败。它们适合做“弱包算法是否能从低置信度候选中救回符号”的第一批样本。

## 7. Python 原型衍生文件

旧实验中已有一些：

```text
*_sequence_observations.json/csv
*_sequence_map.json
_smoke_*_obs.json/csv
_smoke_*_map.json
```

这些是早期序列推断原型输出，保留用于对照。新入口 `scripts/extract_sequence_observations.py` 以 noisy `.bin` 为主输入，会改为输出：

```text
*_phase_observations.json
*_phase_dp.json
*_phase_symbols.csv
```

含义：

| 文件 | 含义 |
| --- | --- |
| `*_phase_observations.json` | 从 packet window 中切出的逐符号 dechirp+FFT Top-K 复数候选 |
| `*_phase_dp.json` | 相位状态 DP 的路径结果、相位模型、每符号候选 branch metric 和 bit LLR |
| `*_phase_symbols.csv` | 一行一个符号，便于快速查看 hard decision 与 phase-DP 是否不同 |

这些衍生文件不改变原始 noisy IQ 数据。

## 8. 设计边界

- `data/noisy_iq/*.bin` 是原始实验样本，后续算法不应覆盖它们。
- 包检测、preamble 对齐、CFO/STO/SFO 粗估计、header/payload/CRC 基线测量都依赖 `gr-lora_sdr`。
- Python 弱包原型默认先复用 `lora_file_RX.py`/gr-lora_sdr 从 `.bin` 得到 packet window，再从这些窗口中抽取复数 FFT 候选，研究相位轨迹、branch metric、符号 LLR、编码约束。
- `noise_power_db_relative` 只描述“加了多少噪声”，不能直接当作 SNR。判断弱包强度应优先看 summary 中的 `grlora_snr_median`、`detected_packets`、`crc_valid_packets`、`ber`。
