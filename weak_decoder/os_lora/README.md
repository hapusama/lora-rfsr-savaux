# OS-LoRa / Savaux 模块边界

本目录同时保存当前联合解调所需的 Savaux 实现，以及此前 OS-LoRa 研究中仍有
复现价值的算法。两者必须分清：当前主链只执行

```text
RFSR 输出
  -> weak_decoder.synchronization.single_packet（在干净输出上检测与 FrameSync）
  -> 固定 FrameSync，在 RFSR 输出上加入不同强度 AWGN
  -> system.synchronized_savaux（带噪波形的 header-first Savaux）
  -> 条件 SER（只统计干净 FrameSync 成功包）
```

当前主链不调用 GLS、双峰重排、CRC 引导或任何 payload 真值引导方法。

## 目录职责

```text
os_lora/
├── system/              可复用算法；包含当前 Savaux 和保留的研究算法
├── experiment_support/  仍保留实验共用的离线基础设施
├── experiments/         可独立运行的历史评估与消融入口
├── tests/               算法、入口和目录依赖测试
├── doc/                 冻结实验结果与历史说明
└── __init__.py          兼容已有实验的公共导出
```

## 当前主链文件

- `system/synchronized_savaux.py`：接收已经通过 FrameSync 的整包 IQ；先用
  Savaux 解调 8 个 explicit-header symbols，解析 payload 长度、CR、CRC 和
  LDRO，再按同步器给出的 CFO/STO/SFO 状态推进 payload 游标。
- `../baselines/savaux_oversampled/paper_oversampled_demod.py`：实现 Savaux 论文中的
  polyphase branch 频谱、确定性相位对齐和相干合并。
- `../synchronization/single_packet.py`：复用已有 preamble detector、frame locator
  和 gr-lora FrameSync，在一条 trimmed IQ 中完成单包同步。
- `../../tools/evaluate_rfsr_savaux_ser.py`：选择 held-out 物理包，先运行 RFSR 和
  clean FrameSync，再在 RFSR 输出上添加配对 AWGN 并复用固定同步信息调用 Savaux，
  最后加载 reference 进行评分。

评测工具同时报告 `clean_synchronized_packets / packet_count` 和条件 SER。未通过
干净 FrameSync 的包会带 `exclusion_reason=clean_framesync_failed`，不会进入
SER 分子或分母。同步成功后，加噪 Savaux 发生的头部无效、数据截断或符号缺失
仍属于解调失败，会在该已同步包的 SER 中体现。

## 保留但不接入当前主链

- `system/oversampled_glrt.py`：Savaux branch GLS、低维噪声协方差和完整采样率
  双峰消融。文件保留供以后重新开展 GLS 对照，当前同步 Savaux 不导入它。
- `system/nonuniform_sampling.py`、`chirp_svd.py`、`litenap_savaux.py`：非均匀采样、
  ChirpSVD 和 LiteNap-Savaux 的历史研究实现。
- `experiments/`：仍被测试或冻结结果文档使用的可复现实验。已经无调用者、被新
  入口替代的一次性 sweep、标定和探针脚本已清理；新联合主链不得反向依赖这里。

顶层 `os_lora.__init__` 暂时保留部分 GLS 导出，以免破坏现有历史实验和测试；
这只是兼容接口，不表示 GLS 参与当前结果。

## 依赖约束

`system/` 不得导入 `experiments/` 或 `experiment_support/`。实验入口可以依赖
`system/` 和 `experiment_support/`，但不得互相导入。架构测试还会逐个导入
现存实验入口，防止清理后留下悬空引用。

```bash
python -m unittest \
  weak_decoder.os_lora.tests.test_architecture \
  weak_decoder.os_lora.tests.test_evaluate_rfsr_savaux_ser \
  weak_decoder.tests.test_single_packet_sync -v
```

历史 GLS、LiteNap 和非均匀采样的冻结命令与结论位于 `doc/`；它们只能作为
对应实验的记录，不应被描述为当前 RFSR + Savaux 主链结果。
