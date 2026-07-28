# USRP 实时检查与 IQ 采集

`usrp_iq_collector.grc` 是当前正式采集入口，流图同时提供：

```text
USRP Source
├── QT GUI 频谱与瀑布图
├── LoRa 同步、FFT、解码和 CRC 统计
└── 绿色 Recording 开关 -> File Sink
```

绿色按钮关闭时，USRP 和实时解码仍运行，但 selector 不把样点送入 File
Sink，因此不会增加 `.cfile` 内容。确认位置、频谱和 CRC 情况后再打开按钮。

当前 Branch4 固定帧参数：

```text
中心频率       487.7 MHz
USRP 采样率    2 Msample/s
LoRa 带宽      125 kHz
扩频因子       SF12
编码率         4/8
前导码         16 symbols
Sync word      0x12
PHY payload    33 bytes，CRC enabled
发包周期       6000 ms
USRP 增益      20 dB，关闭 AGC
天线端口       RX2
```

在采集机激活已安装 GNU Radio、UHD 和 `gnuradio.lora_sdr` 的环境后：

```powershell
uhd_find_devices
uhd_usrp_probe
.\acquisition\open_live_collector_grc.bat
```

从项目根目录启动时，默认输出目录是 `data/raw/ota/`。文件名由流图中的
experiment/session/location/condition/run 以及全部 PHY/射频参数生成。正式
采集前先运行 sidecar 初始化，并把打印出的文件名填入或核对 GRC：

```powershell
python tools\build_rfsr_ota_dataset.py init-capture `
  --experiment-id 1 `
  --session-id 0 `
  --location-id lab1 `
  --condition lowsnr `
  --run-id 0
```

采集结束后在采集机生成 `detections.csv`：

```powershell
python tools\build_rfsr_ota_dataset.py detect `
  --capture data\raw\ota\<规范文件名>.cfile
```

然后复制整个 `lora-rfsr-savaux/` 到服务器。服务器只运行 `server`、
`validate`、训练和实验入口，不需要 GNU Radio、UHD 或兄弟源码仓库。

`collect_usrp_iq.py` 是不带 LoRa 实时解码的轻量命令行备用入口，不用于当前
“现场先看 CRC、再决定是否保存”的正式流程。
