# XCopy 同步复刻与真实低 SNR payload 数据集

## 目标

本工作的最终目标不是用 XCopy 合并帧代替 GLS，而是：

1. 用多个真实重传副本完成弱包检测、时间/CFO/相位同步。
2. 用合并帧确认帧边界是否可信。
3. 把该边界反映射到每个原始重传副本。
4. 导出未合并、未后加噪声、保留 OSR=4 的真实低 SNR payload symbol。
5. 在这些单副本 symbol 上比较普通 FFT、Savaux、GLS 和 NUS+GLS。

合并 payload 只能用于同步验证。若直接在合并 payload 上评价 GLS，约 10 dB 的多副本
增益会掩盖 GLS 自身差异，实验问题会再次被改写。

## 论文依据与可复现边界

依据为 XCopy，MobiCom 2023：
<https://www4.comp.polyu.edu.hk/~csyqzheng/papers/XCopy-MobiCom23.pdf>。

论文公开了算法结构和公式，但没有公开接收机代码，也没有给出以下实现参数：

- 检测功率阈值的数值和噪声估计方式。
- 滑动检测窗口的实际步长。
- Eq. (4) 精细时延搜索的分层策略和搜索步长。
- 多副本权重 `omega_i` 的具体计算公式。
- 送入标准 LoRa decoder 前的工程接口。

因此这里的“一比一复刻”定义为：处理步骤、使用的信号证据和参数可辨识关系与论文一致；
论文未公开的数值参数必须显式记录，不能声称与作者代码 bit-exact。

## XCopy 原文同步链

### 1. 单包长前导检测

论文默认使用 8-chirp preamble 和 4-chirp detection window。连续相同 upchirp 被 dechirp
后在相同频率产生峰值。本实现把 base downchirp 平铺到完整 detection window，并对
`M * RN` 个复数样点做一次长 FFT；这会保留 chirp 间相位并相干积累同一个 tone，而不是
把 M 个单 chirp FFT 的功率非相干相加。

当前 Branch4 数据有 32-chirp preamble。论文式模式仍默认使用 4-chirp detection window，
其余前导用于提高连续峰判定和粗定时稳定性。

与旧实现的差异：

- 论文先逐个检测重传包，再做 packet grouping。
- 旧 `scan_periodic_preamble` 先利用已知 3 秒周期折叠所有副本。
- 周期折叠可作为 Branch4 加速/补漏模式，但不能冒充论文原始 detector。

### 2. 粗帧定时

若 detection window 相对 chirp edge 偏移 `Delta n` 个样点，dechirp 后的频率满足：

```text
f = k * Delta_n / Fs
Delta_n = f * Fs / k
```

长 FFT 的长度为 `M * RN`。若长 FFT 的 signed bin 为 `b`，离散换算为
`Delta_n = b * OSR / M` 个 raw samples。CFO 与该频率耦合，因此这一步只提供论文所说的
粗估计，搜索区间仍保留 `+/-2 chirps`。

论文没有公开滑窗步长、检测阈值和“最高功率平台”选择参数。本实现明确采用：

- 每 `RN` 个 raw samples 移动一次 M-chirp 检测窗；
- 稳定游程至少为 `ceil(0.6 * (P - M + 1))`；
- 从达到游程最大峰值 30% 的第一个窗口开始做粗定时；
- Eq. (4) 先检查 `-2..+2 chirps` 的整数 chirp 假设及 `+/-8` raw samples 邻域，
  最终在原始采样率上重新计算整包共轭 FFT。

这些是论文未公开部分的可审计实现选择，不应声称与作者代码 bit-exact。

### 3. Eq. (4) 整包共轭精对齐

对两个不同重传副本：

```text
z_delta[n] = y_ref[n] * conj(y_copy[n + delta])
```

若两个副本属于同一 PHY 包且时间对齐，公共 preamble、sync、SFD、header 和 payload
波形被消除，整包能量集中成一个 tone。若错位或不是同一包，不同 chirp window 产生不同
tone，能量扩散。

每个 `delta` 的主要评分为：

```text
peak(delta) = max_k |FFT(z_delta)[k]|^2
```

峰值位置给出相对 CFO，峰值相角给出相对相位。精细搜索必须使用整包，不使用 sync word
或 header checksum 决定副本是否对齐。

### 4. STO heterogeneity

论文指出重传包的剩余 STO 差由接收机采样间隔约束。提高采样率可以缩小最大 STO 差；
当采样率高于 `3 * BW` 时，不同 symbol 的 STO 相位差通常保持在可建设性合并范围。

Branch4 的 `Fs=500 ksample/s`、`BW=125 kHz`，即 OSR=4，满足论文条件。整个 XCopy
和后续 GLS 数据导出必须保留 OSR=4，不能先降成 OSR=1。

### 5. CFO、相位补偿与加权合并

Eq. (4) 峰估计的是副本相对参考副本的 CFO 和相位。补偿后按论文 Eq. (3) 合并：

```text
Y[n] = sum_i omega_i * y_i_corrected[n]
```

论文只说明 `omega_i` 根据粗 SNR 调整，没有公开公式。当前实现采用等权相干平均；
尚未实现论文的 SNR 权重，因为猜测一个未公开公式会降低复现的可审计性。后续权重实验
应作为独立消融项，不应混入同步是否成功的主结论。

### 6. 标准 LoRa decoder

论文在合并后把信号送入标准 LoRa decoder，并没有把 sync word 精确匹配或 header checksum
写成 XCopy Eq. (4) 对齐的前置条件。因此本实现将证据分层：

```text
一级硬条件：多副本 Eq. (4) 对齐轨迹成立
二级主证据：长 preamble 稳定 + 两个 SFD downchirp 一致
三级软证据：已知 sync word 的相对 bin 距离
四级验证：explicit-header checksum / payload ground truth
```

sync word、header checksum 和 gr-lora_sdr `framesync_valid` 都不再否决已经成立的
XCopy 对齐；它们只参与候选排序和离线标签。

## Soft frame boundary

绝对 payload 起点不能只靠两个副本的 Eq. (4) 决定，因为 Eq. (4) 只给出副本间相对
时延。合并帧上保留 Top-K 边界候选：

```text
score =
    preamble stability
  + preamble peak concentration
  + SFD pair consistency
  + small sync-word bonus
```

不再使用旧 `frame_locator` 中与 sync word 距离成比例的无界负惩罚。候选即使
`sync1_distance` 或 `sync2_distance` 超过旧阈值，也可以进入 Top-K。最终选择必须记录
各分量，方便判断结果究竟由前导/SFD支持，还是仅由已知 sync word“拉”出来。

## 给 GLS/NUS 的导出

每个入选重传副本输出一组 symbol 元数据：

```text
copy_index
transmission_index
raw_frame_start_sample
raw_data_start_sample
frame_symbol_index
stage
raw_symbol_start_sample
relative_delay_samples
relative_cfo_hz
relative_phase_rad
xcopy_alignment_score
soft_boundary_score
```

`raw_symbol_start_sample` 指向原始 collector IQ，而不是 `combined_iq.bin`。实验程序以原始
IQ 和该 CSV 为输入，从高 SNR 固定帧 ground truth 注入 `gt_bin`，不得向 IQ 再加 AWGN。

实际输出为：

- `aligned_raw_symbols.csv`：所有通过 Eq. (4) 筛选的真实副本及原始 IQ 索引，用于主结果。
- `aligned_raw_sync.csv`：供现有真实采集评估器读取的逐副本同步状态。
- `soft_frame_candidates.csv`：Top-K 绝对边界及各评分分量。
- `packet_detections.csv`：论文模式下每个独立包的长窗检测证据。

header 或真值一致性只能另行构造 sanity-check 子集，不能作为方法主结果，否则会产生按
解码结果挑样本的 selection bias。

## 验收指标

### 同步层

- Eq. (4) 入选副本数量和峰值/背景比。
- 相对 delay/CFO 随 transmission index 的轨迹残差。
- Top-1 与 Top-2 soft boundary 分数间隔。
- 合并后 preamble、SFD、header 的诊断结果，但不作为前置硬门。

### 数据层

- 每个原始副本是否能完整提取 57 个 data symbols。
- 与高 SNR ground truth 比较时，允许每包一个统一整数 bin offset，再统计 residual SER。
- 报告全部 Eq. (4) 入选副本，不能只报告 header-valid 副本。

### GLS/NUS 层

- 所有方法使用完全相同的原始 symbol 窗口和外部 ground truth。
- 不后加 AWGN。
- 普通 FFT、Savaux、GLS、NUS+GLS 同时报告 SER 和 paired symbol wins/losses。
- 按 capture 留出训练噪声或交叉拟合，不能用测试 symbol 自身标签调协方差。

## 2026-07-24 实测

论文模式在 `low4` 上没有使用已知 `1,500,365`-sample 重传周期：

- 独立检测 13 个 packet，Eq. (4) 接受 12 个；
- 合并帧 Top-1 soft-boundary score 为 `328.529`；
- 导出 `12 * 57 = 684` 行原始符号，其中实际 payload 评估为 588 symbols；
- 评估器记录 `strict_sync_count=0`，即主结果没有按硬同步字/header-valid 筛包；
- ground-truth dechirp SNR 汇总为 `-17.002 dB`，单包范围 `-18.957..-15.204 dB`；
- ordinary FFT 为 `35/588` 错，Savaux 为 `1/588`，GLS cross-fit 和 off-packet
  也都是 `1/588`。

因此前端现在确实提供了未合并、未后加 AWGN 的真实低 SNR payload；但当前 GLS 在这组
数据上没有超过 Savaux，这也是后续优化 NUS+GLS 必须面对的基线，而不是同步成功后自动
出现的算法增益。

同一套论文模式在 `low5` 上独立 packet detection 为 `0`，流程尚未进入 Eq. (4)、SFD、
sync word 或 header。放松 sync word 对这个失败没有作用。若要推进到 `low5` 以下，需要
增加 detection window/preamble 的积累量或重新设计单包检测统计量，而不是继续放松帧后部
判据。

补充 detection-window sweep：

| DW | 独立候选 | Eq. (4) 入选 | 结论 |
| ---: | ---: | ---: | --- |
| 4 / 8 / 12 | 0 | 0 | detector 未启动 |
| 16 | 2 | 0 | 不足以分组 |
| 20 | 13 | 7 | 假 frame；GT payload SER 约 1 |
| 24 | 13 | 8 | 假 frame；GT payload SER 为 1 |

DW=20/24 的 Top-1 只有 2 个稳定 preamble peaks，虽然偶然得到相近的 SFD bins，但独立
ground truth 证明 payload 完全错位。当前实现因此保留这些 Top-K 诊断，却要求至少一半
preamble chirps 稳定且 SFD pair 一致，才允许标记 `ok_soft_boundary` 和导出 raw payload。
这条门不依赖 sync word 或 header。
