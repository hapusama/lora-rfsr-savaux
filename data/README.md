# 数据目录说明

`data/` 是本项目唯一的数据根目录。所有原始 IQ、理想 reference、微调数据、
清单和实验结果都应保存在这里，不要依赖项目旁边的其他仓库或绝对路径。

大体可以把它理解为三部分：

```text
data/
├── raw/                         # 原始实验输入，不可修改
├── reference_phy/               # 理想 PHY reference 与 RF-SR 数据库
└── processed/results/...        # 其他实验的中间结果和输出
```

## 一、原始采集数据：`raw/`

```text
raw/
├── packet_reference.txt         # STM32 串口输出的发送真值
└── ota/
    ├── <capture>.cfile           # USRP 连续 complex64 IQ
    ├── <capture>.cfile.json      # 该轮采集的参数 sidecar
    └── <capture>.usrp.txt        # GNU Radio 控制台解码与 CRC 日志
```

### `packet_reference.txt`

这是发送端 ground truth，包含每个 packet 的完整 33-byte `[TX Frame]`、payload
ID、PHY 参数和发包周期。理想 reference 必须从这里生成，不能只相信低 SNR
条件下可能出错的接收端日志。

### `raw/ota/*.cfile`

这是 USRP 输出的连续原始 IQ。目前正式 RF-SR 采集使用：

```text
数据类型       little-endian complex64（<c8）
采样率         2 Msample/s
文件头         无
```

原始 cfile 是不可替代的实验证据。采集完成后不要裁剪、归一化或覆盖它，所有
派生数据都应写到 `reference_phy/rfsr_db/`。

### `.cfile.json`

记录 experiment/session/location/condition/run，以及 SF、BW、采样率、前导码、
sync word、CR、CRC、中心频率和 RX gain。它与同名 cfile 必须一起保留。

### `.usrp.txt`

保存 GNU Radio 终端打印的 `rx msg` 和 CRC 结果。它用于辅助审计，不负责决定
理想 payload；最终真值仍来自 `packet_reference.txt`。

## 二、理想 reference 语料：`reference_phy/`

```text
reference_phy/
├── reference/
│   └── signalout_000037_fulltrim.cfile
├── metadata/
│   └── 000037.json
└── rfsr_db/
```

`reference/` 和 `metadata/` 由下面的脚本生成：

```bash
python tools/generate_reference_phy.py \
  --uart-log data/raw/packet_reference.txt \
  --output-root data/reference_phy
```

每个 reference 是一个 1 Msample/s 理想 LoRa 发射波形：

```text
10,000 个零前缀
+ preamble
+ sync word
+ SFD
+ explicit header
+ 完整 payload/CRC
```

它不包含实际信道、接收增益、CFO、AWGN 或人工多径，主要用于：

1. 合成预训练；
2. 建立 `reference_id → 完整帧 → 理想 IQ` 的 catalog；
3. 作为 OTA 微调的 1 Msample/s 标签。

## 三、OTA 微调数据库：`reference_phy/rfsr_db/`

这是 `tools/build_rfsr_ota_dataset.py` 的主要输出目录：

```text
rfsr_db/
├── ota/                         # 裁剪后的真实接收 IQ
├── reference/                   # 与 OTA 配对的理想标签
├── metadata/                    # 每个 OTA 文件的来源、身份与裁剪证据
└── manifests/
    ├── reference_catalog.csv    # UART/reference 总目录
    ├── views.csv                # 模型实际读取的训练视图表
    ├── validation.csv           # 每个 OTA 文件的校验结果
    ├── validation_summary.json  # 校验汇总
    └── captures/
        └── <capture_uid>/
            ├── capture.json
            ├── detections.csv
            ├── packets.csv
            ├── rx_events.csv
            └── trim_issues.csv
```

### `rfsr_db/ota/`

文件名示例：

```text
exp0_000157_rxg20_0_fulltrim.cfile
exp0_000157_rxg20_1_fulltrim.cfile
```

含义：

```text
exp0       第 0 号采集实验
000157     连续 capture 内的第 157 个物理包，不是 reference_id
rxg20      USRP 接收增益 20 dB
0 / 1      原始 2 MS/s IQ 的偶数/奇数 ADC 相位
fulltrim   真实包外 IQ + 完整 packet
```

每个 accepted 物理包会从 2 Msample/s 原始 IQ 拆出两个 1 Msample/s OTA
文件。每个文件再由训练 loader 动态产生四个 250 ksample/s 输入：

```python
x0 = ota[0::4]
x1 = ota[1::4]
x2 = ota[2::4]
x3 = ota[3::4]
```

所以一个物理包共有 `2 × 4 = 8` 个训练视图，但磁盘上只保存两个 1 Msample/s
OTA 文件。

### `rfsr_db/reference/`

这是微调数据库内部使用的标签目录。内容与
`data/reference_phy/reference/` 中的理想波形相同，通常通过硬链接共享磁盘
内容，并没有真正保存两份。

两个目录用途不同：

```text
reference_phy/reference/          理想 reference 原始语料库
reference_phy/rfsr_db/reference/  可独立搬运的微调数据库标签目录
```

不要通过 OTA 文件名猜测 reference。实际映射记录在同名 metadata 和
`manifests/views.csv` 中。

### `rfsr_db/metadata/`

每个 OTA cfile 都有一个同名 JSON，记录：

- 原始 capture 路径与哈希；
- packet 起始采样点；
- `reference_id` 和 reference 相对路径；
- CRC、关联方法和置信度；
- 2 Msample/s 裁剪范围；
- ADC phase 和可用的四个低采样率 phase；
- SNR、CFO、STO、SFO；
- 文件长度、功率和质量检查结果。

### `manifests/reference_catalog.csv`

保存：

```text
reference_id → 完整 33-byte frame → 理想 reference 路径
```

它由 UART ground truth 和理想 reference metadata 共同建立。

### `manifests/captures/<capture_uid>/detections.csv`

保存 GNU Radio 在连续 IQ 中检测到的：

- packet 起始采样点；
- 解码 frame；
- CRC；
- SNR、CFO、STO、SFO。

`detect` 是唯一需要 GNU Radio 的阶段。

### `manifests/captures/<capture_uid>/packets.csv`

把 `detections.csv` 中的检测位置关联到 `reference_id`，并给出：

```text
accepted    身份证据充分，可以进入 trim
ambiguous   证据不足，保留审计但不进入训练
rejected    明确冲突或重复，不进入训练
```

### `manifests/views.csv`

这是 OTA 微调 loader 真正读取的主表。每一行明确记录：

```text
OTA 文件
250 ksample/s 抽取相位
reference_id
1 Msample/s reference 文件
物理 packet split_group
```

模型不会通过文件名推测 reference，而是严格按照这张表加载：

```text
250 ksample/s 真实 OTA 输入 x
→ RF-SR 输出 1 Msample/s
→ 与 1 Msample/s 理想 reference y 计算损失
```

## 四、预训练与微调数据生成

生成预训练 reference：

```bash
python tools/generate_reference_phy.py \
  --uart-log data/raw/packet_reference.txt \
  --output-root data/reference_phy
```

采集机为 OTA cfile 生成检测结果：

```bash
python tools/build_rfsr_ota_dataset.py detect \
  --capture data/raw/ota/<capture>.cfile
```

服务器根据已有检测结果生成微调数据库：

```bash
python tools/build_rfsr_ota_dataset.py server \
  --capture data/raw/ota/<capture>.cfile \
  --usrp-log data/raw/ota/<capture>.usrp.txt
```

如果同一台采集机已经安装 GNU Radio，也可以一次运行：

```bash
python tools/build_rfsr_ota_dataset.py all \
  --capture data/raw/ota/<capture>.cfile \
  --usrp-log data/raw/ota/<capture>.usrp.txt
```

## 五、完整性、Git 与服务器上传

`.cfile` 等大 IQ 文件被 `.gitignore` 排除，所以只执行 `git clone` 不会得到
完整数据集。上传服务器时应复制或 `rsync` 整个 `lora-rfsr-savaux/` 目录。

上传前后可检查 SHA-256：

```bash
sha256sum data/raw/ota/<capture>.cfile
```

服务器预检：

```bash
python tools/check_server_bundle.py --mode preprocess
python tools/check_server_bundle.py --mode train
```

## 六、哪些内容不要删除

以下内容属于实验输入或训练数据库，应保留：

```text
data/raw/packet_reference.txt
data/raw/ota/
data/reference_phy/reference/
data/reference_phy/metadata/
data/reference_phy/rfsr_db/
```

`processed/`、`results/` 和顶层 `manifests/` 是其他实验的通用目录；确认没有
需要保留的中间结果后才能清理，它们不是本脚本当前 RF-SR OTA 主流程的核心输入。
