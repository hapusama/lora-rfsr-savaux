# USRP 离线 IQ 采集

推荐使用 `collect_usrp_iq.py`。它保存无文件头的 GNU Radio
`gr_complex`，即 `numpy.complex64` IQ，同时生成同名 `.bin.json` 元数据。
`usrp_iq_collector.grc` 用于 GNU Radio Companion 可视化调试。

当前固定帧实验默认参数：

```text
中心频率       487.7 MHz
采样率         500 ksample/s（RF-SR 配对实验另采/生成 250k 与 1M）
LoRa 带宽      125 kHz
编码率         4/7
前导码         32 symbols
Sync word      0x34
PHY payload    33 bytes，CRC enabled
USRP 增益      20 dB，关闭 AGC
天线端口       RX2
```

在安装了 GNU Radio UHD 的环境中：

```powershell
uhd_find_devices
uhd_usrp_probe
python .\acquisition\collect_usrp_iq.py `
  --output .\data\raw\branch4_fixed\high_snr\sf10_bw125_fs500_pre32_sw34_r001.bin `
  --duration 120 `
  --center-freq 487.7e6 `
  --samp-rate 500e3 `
  --lora-bandwidth 125e3 `
  --rf-bandwidth 500e3 `
  --gain 20 `
  --antenna RX2 `
  --device-args "serial=YOUR_USRP_SERIAL"
```

脚本默认拒绝覆盖已有文件；确实需要覆盖时才使用 `--overwrite`。采集后把
IQ、JSON 元数据和 SHA-256 同时登记到 `data/manifests/`。

500 ksample/s 的 complex64 约为 4 MB/s，即每分钟约 229 MiB。正式采集前先
用 10–20 秒短记录验证中心频率、增益、丢样和磁盘写速，避免产生无法使用的
大文件。
