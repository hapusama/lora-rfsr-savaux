# XCopy 多副本同步

实现入口：

- `weak_decoder/synchronization/xcopy_sync.py`
- `scripts/run_xcopy_sync.py`

详细的论文对应关系、未公开参数和实验纪律见
`doc/xcopy_replication_and_real_payload.md`。方法依据为 XCopy（MobiCom 2023）：
<https://www4.comp.polyu.edu.hk/~csyqzheng/papers/XCopy-MobiCom23.pdf>。

## 两种模式

`paper` 是默认模式，不使用已知重传周期：

1. 对每个 M-chirp 长窗做一次相干 dechirp/长 FFT。
2. 用连续稳定峰游程独立产生 packet 候选。
3. 用峰频率给出粗定时，并保留 `+/-2 chirps` 搜索范围。
4. 对不同发射副本计算整包 `y_ref[n] * conj(y_copy[n + delay])`。
5. 用 Eq. (4) tone 的峰值/背景比完成分组，并估计相对 delay、CFO 和相位。
6. 相干合并后，用 preamble 和 SFD 选 Top-K 绝对边界；sync word 只给小额 bonus。
7. 把边界映射回每个未合并原始副本，导出真实低 SNR payload。

`periodic` 是 Branch4 的加速/对照模式。它使用实测
`1,500,365`-sample 发包周期先折叠前导，再执行 Eq. (4)；这不是论文原始 detector。

两种模式都只共轭不同时间的真实发射副本，不把 OSR=4 的四条 polyphase 分支当作副本。

## 运行

从 `gr-lora_sdr` 目录执行论文模式：

```powershell
python weakPacket_decoding/scripts/run_xcopy_sync.py `
  -i weakPacket_decoding/USRP_collector/data/branch4_fixed/low_snr/sf10_bw125_fs500_pre32_sw34_low4.bin `
  -o weakPacket_decoding/data/experiments/xcopy_paper_low4_20260724 `
  --detection-mode paper `
  --detection-chirps 4 `
  --max-copies 13 `
  --overwrite
```

Branch4 周期模式需要显式指定：

```powershell
python weakPacket_decoding/scripts/run_xcopy_sync.py `
  -i weakPacket_decoding/USRP_collector/data/branch4_fixed/low_snr/sf10_bw125_fs500_pre32_sw34_low4.bin `
  -o weakPacket_decoding/data/experiments/xcopy_periodic_low4 `
  --detection-mode periodic `
  --period-samples 1500365 `
  --max-copies 13
```

默认 PHY 参数为 SF10、BW125 kHz、Fs500 ksample/s、preamble 32、sync word `0x34`、
57 个 header+payload symbols。默认 paper detector 使用 4-chirp 长窗和 30% 首平台点；
这些数值均写入 `summary.json`。

## 输出

- `packet_detections.csv`：逐包长窗检测证据，周期模式为空。
- `copies.csv`：相对 delay、CFO、相位、Eq. (4) 分数及是否入选。
- `combined_iq.bin`：保留 OSR=4 的等权相干合并 IQ，只用于同步验证。
- `soft_frame_candidates.csv`：不以 sync word 硬拒绝的 Top-K 绝对边界。
- `aligned_raw_symbols.csv`：指回 collector 原始 IQ 的逐符号索引和估计状态。
- `aligned_raw_sync.csv`：兼容真实采集 GLS 评估器的逐副本同步记录。
- `combined_sync.csv`：合并帧诊断，兼容已有 header-first 解调器。
- `summary.json`：配置和各阶段状态。

`aligned_raw_symbols.csv` 才是 GLS/NUS 主实验输入。`combined_iq.bin` 含多副本能量增益，
不能用于宣称单包 GLS 的提升。

## low4 结果

2026-07-24 的论文模式结果：

| 指标 | 结果 |
| --- | ---: |
| 独立检测 packet | 13 |
| Eq. (4) 入选副本 | 12 |
| 导出原始 data rows | 684 |
| 实际 payload 评估 | 588 symbols |
| dechirp ground-truth SNR | -17.002 dB |
| ordinary FFT | 35/588 |
| Savaux | 1/588 |
| GLS cross-fit | 1/588 |
| GLS off-packet | 1/588 |

评估输出位于：

- `data/experiments/xcopy_paper_low4_20260724`
- `data/experiments/xcopy_paper_raw_low4_gls_20260724`

`low5` 在相同配置下独立检测为 0，失败发生在长前导 packet detection，早于 Eq. (4)、
SFD、sync word 和 header。DW=16 能产生 2 个候选；DW=20/24 会产生 Eq. (4) 噪声组，
但其 Top-1 只有 2 个稳定 preamble peaks，独立 ground truth 的 payload SER 接近 1。
因此这些结果保留为诊断但不再导出 raw payload。继续降低 sync-word 判据不会改变这个
边界。

## 判据边界

- Eq. (4) 多副本一致性是副本入组硬条件。
- 至少半数 preamble chirps 稳定且 SFD pair 一致，才接受绝对帧边界。
- sync word 是有界软分数，不硬拒绝。
- explicit-header checksum 和 ground truth 仅用于事后诊断。
- gr-lora_sdr CFO/STO/SFO 结果可以细化边界，但其 `valid` 标志不是 raw payload 导出的门。
- OSR=4 从检测、对齐到 raw symbol 导出始终保留。

论文没有公开检测阈值、滑窗步长、精搜调度和 SNR 权重公式。因此这里是算法级复现，不是
作者代码的 bit-exact 复刻；所有补充选择都在配置和文档中显式记录。
