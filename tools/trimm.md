整体思路是对的，但要把“包的位置”“包的身份”“理想 reference”三件事分开管理：

- 包的位置：只能由 cfile 的 IQ 检测结果决定。
- 包的身份：优先用 CRC 正确且与串口 reference 完全一致的解码结果；失败时才根据时序和前后锚点推断。
- 理想 reference：始终来自串口 ground truth 和现有 reference PHY，不由 `usrp_log.txt` 生成。

本轮只完成了设计和只读核对，没有修改文件或写代码。

## 当前数据的实际情况

我核对了现有文件：

- cfile：12,480,000,000 字节
- complex64 样点数：1,560,000,000
- 2 MS/s 下时长：780 秒
- `usrp_log.txt`：131 个 `rx msg`，131 个 CRC valid
- 131 帧都与 [packet_reference.txt](/D:/Desktop/proj/lora-rfsr-savaux/data/raw/packet_reference.txt) 中对应帧逐字节一致
- 日志里的 ID 顺序为：

```text
50, 0, 1, 2, ... 119, 0, 1, ... 9
```

开头孤立的 ID 50 很重要：实时解码支路不经过绿色录制开关，因此它可能是按下录制按钮之前的预检包，并不在 cfile 中。这进一步证明不能把“日志第 n 条”直接当成“cfile 第 n 个包”。

## 精确的裁剪和多相划分

现有 1 MS/s reference 的固定长度是：

```text
总长度             2,770,704 samples
前置零样点            10,000 samples = 10 ms
LoRa packet         2,760,704 samples
尾部零样点                    0
```

因此 2 MS/s 原始 IQ 的最终裁剪长度应为：

| 数据 | 采样率 | 总样点 | 包前 off-packet |
|---|---:|---:|---:|
| 原始固定裁剪 | 2 MS/s | 5,541,408 | 20,000 |
| 两个 OTA phase | 1 MS/s | 2,770,704 | 10,000 |
| 每个 OTA phase 的四个低速视图 | 250 kS/s | 692,676 | 2,500 |
| 理想 reference | 1 MS/s | 2,770,704 | 10,000 个零 |

假设检测出的前导码起点为 `S`：

```text
crop_2m = raw[S - 20_000 : S + 5_521_408]

ota_p0 = crop_2m[0::2]
ota_p1 = crop_2m[1::2]

x_pq = ota_p[q::4]
```

等价地：

```text
x_pq = crop_2m[p + 2*q :: 8]
```

其中：

```text
p = 0, 1
q = 0, 1, 2, 3
```

最终正好得到原始 2 MS/s 信号的八个 250 kS/s 抽取相位：

```text
phase = 0, 1, 2, 3, 4, 5, 6, 7
```

这套划分数学上完全成立，而且每个低速视图都保持：

```text
2,500 个真实 off-packet 噪声样点
+ 完整 packet
```

这里应理解为保持相同的物理时长 10 ms，而不是每个采样率都强行保留 10,000 个点。

为了与预训练一致，不加抗混叠滤波，直接做多相抽取。预训练本身也是从 1 MS/s reference 直接 `[::4]` 得到 250 kS/s。

## 推荐的处理流水线

建议写一个分阶段、可以断点续跑的脚本，例如：

```text
tools/build_rfsr_ota_dataset.py

catalog
detect
associate
trim
validate
all
```

### 1. catalog：构建串口真值目录

复用现有严格解析器：

[reference_phy.py](/D:/Desktop/proj/lora-rfsr-savaux/weak_decoder/rf_super_resolution/reference_phy.py)

从 `packet_reference.txt` 生成 `reference_catalog.csv`。内容包括：

```text
reference_id
payload_id
frame_hex
app_payload_hex
frame_bytes
SF/BW/CR/preamble/sync-word/CRC/LDRO
reference_iq_path
reference_iq_sha256
source_uart_path
source_uart_sha256
```

CSV 是方便程序使用的派生索引，不替代原始串口 txt。原始 txt 和哈希必须保留。

当前 UART 文件中 round 1 的 ID 0、1 与 round 0 完全相同，解析时应检查重复 ID 的 frame 是否一致，然后折叠成 120 个唯一 reference。

### 2. detect：从 IQ 获取包位置

直接复用现有离线检测器：

[detector.py](/D:/Desktop/proj/lora-rfsr-savaux/noisy_iq/detector.py)

它已经能够从 GNU Radio 的 `frame_sync`、header decoder 和 CRC block 获取：

```text
start_sample
end_sample
CRC valid
decoded frame
SNR
CFO
STO/SFO
header status
```

输出 `detections.csv`。

这一步必须直接读取 cfile，因此获得的 `start_sample` 与录制文件处于同一个坐标系，不受绿色开关之前的终端日志影响。

高 SNR 数据还应拿来校准 `frame_sync.start_sample` 是否存在固定偏差：用前导码相关在检测点附近精调，然后把偏差和残差写进检测结果。

### 3. associate：确定 reference_id

关联优先级建议如下：

1. `CRC valid + 33 字节 frame 与 catalog 完全相同`

   标为 `crc_exact`，最高可信度。

2. CRC valid，但 frame 不在 catalog

   不能因为 CRC 成功就强行接收，应标为异常并隔离。

3. CRC 失败，但存在前导码检测

   使用前后 CRC 锚点和周期模型推断 slot，再得到预期 ID。

4. 连前导码检测都丢失

   根据多个可靠锚点拟合发送周期，在预计位置附近重新进行局部前导码搜索和相关精调。

周期不能固定写死成恰好 12,000,000 个 2 MS/s 样点。应该用高可信检测点稳健拟合：

```text
start[n] ≈ offset + n × period_samples
```

同时必须支持“发射机重启/序列跳变分段”。当前日志从 50 跳到 0 就是实际例子，不能跨这个边界继续按 modulo 120 推理。

推断结果建议分级：

```text
crc_exact
neighbor_inferred_high
schedule_inferred_medium
ambiguous
rejected
```

只有前三级可以进入正式数据集；模糊项不能静默生成错误标签。

### 4. trim：写出两个 1 MS/s OTA 文件

建议每个物理 reception 生成两个文件：

```text
exp0_000157_rxg18_0_fulltrim.cfile
exp0_000157_rxg18_1_fulltrim.cfile
```

最后的 `0/1` 明确定义为 `adc_phase_2m`。

六位数字定义为物理接收事件编号，不等同于 reference ID。例如：

```text
capture_packet_index = 157
reference_id = 37
```

虽然当前刚好有 `157 % 120 = 37`，脚本也绝不能依赖文件名取模，而要读取 metadata 里的显式 `reference_id`。

### 5. validate：强制质量检查

每个输出至少检查：

- OTA 文件是 little-endian complex64。
- 每个 1 MS/s 文件恰好 2,770,704 点。
- 每个 250 kS/s 视图恰好 692,676 点。
- 两个 1 MS/s phase 重新交织后能逐点恢复原始 2 MS/s crop。
- packet 在 1 MS/s 文件的索引 10,000 附近开始。
- 前 10,000 个 OTA 点是真实噪声，不是脚本补零。
- 对应 reference 的前 10,000 点确实为零。
- reference frame 与关联的 decoded/expected frame 一致。
- 保存检测残差、前导码相关峰、CFO、SNR 和异常原因。

## 建议的数据目录

在你给出的结构上增加可审计的中间结果：

```text
data/reference_phy/rfsr_db/
├── ota/
│   ├── exp0_000037_rxg24_0_fulltrim.cfile
│   ├── exp0_000037_rxg24_1_fulltrim.cfile
│   ├── exp1_000157_rxg18_0_fulltrim.cfile
│   └── exp1_000157_rxg18_1_fulltrim.cfile
├── reference/
│   └── signalout_000037_fulltrim.cfile
├── metadata/
│   ├── exp0_000037_rxg24_0.json
│   ├── exp0_000037_rxg24_1.json
│   ├── exp1_000157_rxg18_0.json
│   └── exp1_000157_rxg18_1.json
├── manifests/
│   ├── reference_catalog.csv
│   ├── rx_events.csv
│   ├── detections.csv
│   ├── packets.csv
│   └── views.csv
└── diagnostics/
```

`reference/` 不需要重新生成理想波形。可以从现有 reference 文件创建硬链接，并验证哈希，避免重复占用约 2.66 GB 空间。

## Metadata 应记录什么

每个 1 MS/s OTA 文件一个 JSON，至少包含：

```text
schema/schema_version
physical_packet_uid
capture_packet_index
reference_id
expected_frame_hex

source_capture_path
source_capture_sha256
source_sample_rate_hz
center_frequency_hz
rx_gain_db
session/location/SNR condition

detected_preamble_start_2m
association_method
association_confidence
previous_anchor/next_anchor

trim_start_2m
trim_stop_2m
leading_off_packet_samples_2m
adc_phase_2m
output_samples_1m

available_lowrate_phases = [0,1,2,3]
combined_decimation_phases_2m
estimated_cfo_hz
timing_residual
complex_gain/phase estimate
QC result
split_group
```

两个 metadata 完全可以指向同一个：

```text
reference_id = 37
reference/signalout_000037_fulltrim.cfile
```

## 250 kS/s 文件是否落盘

我推荐只保存两个 1 MS/s OTA 文件，不把八个 250 kS/s 视图再次写盘。

原因是四个低速视图可以在 Dataset 中零成本得到：

```text
x = ota_1m[q::4]
```

`views.csv` 为每个 1 MS/s 文件枚举 `q=0..3` 即可。这样不会再额外产生一份约 5.8 GB 的重复数据。如果后续解调程序只能读独立 cfile，再增加可选的 `lowrate/` 缓存导出功能。

需要特别注意：这八个 phase 是同一个物理 packet 的八个相关视图，不是八个独立 packet。训练、验证和测试划分时必须全部使用同一个 `physical_packet_uid` 分组；更严格时还应按 `reference_id` 分组，防止同一理想标签同时出现在训练集和测试集。

我的推荐最终方案就是：

```text
UART reference = 唯一 payload ground truth
IQ detector     = 唯一样点位置 ground truth
CRC log         = 身份关联证据
2×1M 文件       = 磁盘上的标准 OTA 数据
8×250k phase    = Dataset 中的训练视图
1M ideal ref    = 监督标签
```

这版设计可以直接进入实现。唯一需要你确认的是：是否接受“只落盘两个 1 MS/s 文件、四个 250 kS/s phase 由 Dataset 动态生成”；我认为这是最合适的方式。

## 已实现的脚本与命名契约

实现入口：

```text
tools/build_rfsr_ota_dataset.py
```

脚本提供以下可断点执行的阶段：

```text
init-capture
catalog
detect
associate
trim
validate
all
```

所有持久化输出都会检查真实路径，输出根目录如果不在
`lora-rfsr-savaux/data/` 内会直接拒绝执行。

### 连续原始采集文件名

后续正式采集统一使用：

```text
rxcap_exp000_sess000_loclab1_condhighsnr_run000_
sf12_bw125000_fs2000000_pre16_sw12_cr48_crc1_
fc487700000_rxg20.cfile
```

实际文件名为单行，上面只是为了阅读换行。字段含义：

```text
exp       连续采集的唯一实验编号；每生成一个新 cfile 就递增
sess      实验 session
loc       位置短标签，只允许小写字母、数字和连字符
cond      highsnr/lowsnr/noiseonly/interference 等条件
run       同一配置的重复轮次
sf/bw/fs  LoRa SF、带宽 Hz、原始采样率 Hz
pre/sw    preamble 和 sync word
cr/crc    编码率和 PHY CRC
fc        USRP 中心频率 Hz
rxg       手动 RX gain，单位 dB
```

先生成规范名称和 sidecar：

```powershell
python tools\build_rfsr_ota_dataset.py init-capture `
  --experiment-id 1 `
  --session-id 0 `
  --location-id lab1 `
  --condition lowsnr `
  --run-id 0
```

默认会在 `data/raw/ota/` 下生成 `.cfile.json`，并打印应填入 GRC 的
cfile 路径。GRC 中的 `capture_experiment/session/location/condition/run`
应与 sidecar 一致。

### 逐包 fulltrim 文件名

每个物理 packet 落盘两个 1 MS/s phase：

```text
exp1_000157_rxg20_0_fulltrim.cfile
exp1_000157_rxg20_1_fulltrim.cfile
```

这里 `000157` 是该连续 cfile 内的物理包序号，最后的 `0/1` 是
`adc_phase_2m`。`reference_id` 不从文件名取模，而是显式保存在对应 JSON
和 `manifests/views.csv` 中。

### 当前高 SNR 文件与服务器入口

本轮 12.48 GB 原始 IQ 已按规范登记在项目内：

```text
data/raw/ota/
├── rxcap_exp000_sess000_loclab1_condhighsnr_run000_sf12_bw125000_fs2000000_pre16_sw12_cr48_crc1_fc487700000_rxg20.cfile
├── rxcap_exp000_sess000_loclab1_condhighsnr_run000_sf12_bw125000_fs2000000_pre16_sw12_cr48_crc1_fc487700000_rxg20.cfile.json
└── rxcap_exp000_sess000_loclab1_condhighsnr_run000_sf12_bw125000_fs2000000_pre16_sw12_cr48_crc1_fc487700000_rxg20.usrp.txt
```

采集机只负责需要 GNU Radio/OOT 模块的检测：

```powershell
python tools\build_rfsr_ota_dataset.py detect `
  --capture data\raw\ota\rxcap_exp000_sess000_loclab1_condhighsnr_run000_sf12_bw125000_fs2000000_pre16_sw12_cr48_crc1_fc487700000_rxg20.cfile
```

检测结果固定写到项目内的
`data/reference_phy/rfsr_db/manifests/captures/exp000_sess000_run000/detections.csv`。
复制整个 `lora-rfsr-savaux/` 到服务器后，使用不会导入 GNU Radio 的命令：

```bash
python tools/check_server_bundle.py --mode preprocess
python tools/build_rfsr_ota_dataset.py server \
  --capture data/raw/ota/rxcap_exp000_sess000_loclab1_condhighsnr_run000_sf12_bw125000_fs2000000_pre16_sw12_cr48_crc1_fc487700000_rxg20.cfile \
  --usrp-log data/raw/ota/rxcap_exp000_sess000_loclab1_condhighsnr_run000_sf12_bw125000_fs2000000_pre16_sw12_cr48_crc1_fc487700000_rxg20.usrp.txt
python tools/check_server_bundle.py --mode train
```

`server` 严格执行 `catalog -> associate -> trim -> validate`，缺少项目内
`detections.csv` 时直接失败，不会尝试寻找兄弟 `gr-lora_sdr/` 仓库。
