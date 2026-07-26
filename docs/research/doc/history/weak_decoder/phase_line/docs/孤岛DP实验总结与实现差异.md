# 孤岛DP实验总结与实现差异

记录时间：2026-06-29

## 一句话结论

你给的“高置信锚点 + 低置信孤岛 DP + 符号内两段相干重构”方向是对的，但当前实验说明：真正限制收益的不是 DP 连接方式，而是低置信符号的局部证据质量。新分支已经做到比原 v1 一阶相位 DP 更好一些，但还没有达到期望的 1-2 dB 横向增益。

当前最好结果来自：

```text
dual Stage-1 evidence
+ multi-origin 采样相位证据
+ weighted_unit 相位融合
+ packet gate
+ symbol gate
```

不是纯 island DP 本身。

## 当前最好实验结果

主测试集：

```text
dataset: 0_0_0_10_14_32
SNR    : -25, -26, -27, -28, -29 dB
seeds  : 42, 43, 44, 45, 46
packets: 每组最多 10
symbols: 6125
```

结果：

| 方法 | 错符号数 | 相对 v1 |
|---|---:|---:|
| v1 一阶相位 DP | 534 | baseline |
| dual evidence fusion | 515 | +19 |
| raw multi-origin | 515 | +19 |
| gated multi-origin + symbol gate | 497 | +37 |

更细的 rescue/break：

| 对比 | rescue | break | 净收益 |
|---|---:|---:|---:|
| gated vs v1 | 46 | 9 | +37 |
| gated vs dual | 21 | 3 | +18 |

换成 SER 看：

```text
v1     : 534 / 6125 = 8.72%
current: 497 / 6125 = 8.11%
```

这是一个小但真实的 FFT-bin-only 改善，约等于 6.9% 的相对错符号下降。它还不是 1-2 dB 级别的大跨越。

## Holdout 检查

symbol gate 的参数不是用 CRC、codec 或 GT 在运行时选的。GT 只用于离线扫阈值和评估。

当前 gate 在 train/test split 上的结果：

| seeds | dual 错误 | gated 错误 | rescue/break |
|---|---:|---:|---:|
| train 42-44 | 308 | 299 | 12 / 3 |
| test 45-46 | 207 | 198 | 9 / 0 |
| all 42-46 | 515 | 497 | 21 / 3 |

这个结果比之前 margin-only gate 更干净：同样能到 497 errors，但改动符号从 81 个降到 68 个，test split 上 break 为 0。

## 跨 dataset 检查

同一套 gate 分别跑 8/16/32 symbol fixture：

| dataset | v1 | dual | raw multi | gated |
|---|---:|---:|---:|---:|
| 0_0_0_10_14_8 | 0 | 0 | 0 | 0 |
| 0_0_0_10_14_16 | 0 | 0 | 10 | 0 |
| 0_0_0_10_14_32 | 534 | 515 | 515 | 497 |
| total | 534 | 515 | 525 | 497 |

所以收益主要集中在 32-symbol 弱包。8/16-symbol 在这个 SNR 切片下 v1 已经是 0 错，基本没有 rescue 空间，只能用来暴露 break 风险。

## 相对于原始孤岛算法的主要改动

### 1. 没有直接废掉 v1，而是把孤岛 DP 独立成实验分支

原设想是让 constrained Viterbi 成为二阶段主路径。实际实现时我把它放到：

```text
variants/island_dp_reconstruction/
```

原因是当时 v1 一阶相位 DP 已经是最强 baseline，直接替换风险太大。新分支可以独立跑评估，不污染主路径。

### 2. Anchor hard lock 保留，但更偏保守

原设想里高置信符号硬锁，低置信区间放开跑。实现里仍然保留这个核心：

```text
locked anchor 不允许被 island DP 修改
只有左右 anchor 都存在的低置信 interval 才跑 2D DP
```

这部分是对的，可以有效抑制 E3 类型错杀。

### 3. DP 状态扩成了 `(raw_bin, branch)`

原设想中的二维状态空间已经落地：

```text
state = (candidate_raw_bin, oversampling_branch)
```

branch 转移会参考锚点插值得到的局部 fractional-STO trend，并对不平滑跳变加惩罚。

### 4. 两段相干重构做了，但没有成为主收益来源

实现里 `compute_two_segment_score(...)` 使用 `dechirped_symbols` 时，可以按候选 bin 的 LoRa wrap point 拆成两段，再做候选专属补偿和复数拼接。

但是实验结果不理想：

```text
two-segment reconstruction probe:
GT median rank 约 52

independent per-branch phase DP:
SER 约 18%-23%
只修正了 3/101 个 fusion miss
```

也就是说，它目前还不能稳定把 GT 从 Top10/Top40 拉到 Top1。这里是当前最值得继续打的点。

### 5. 增加了 dual evidence fusion

为了补 Stage-1 局部证据，我加了 old center evidence 和 residual-STO-corrected evidence 的融合：

```text
old center phase
+ product_norm(old power, residual-STO-corrected power)
```

这是第一个稳定超过 v1 的来源：

```text
v1 errors     : 534
dual errors   : 515
net vs v1     : +19
```

### 6. 增加了 multi-origin 采样相位融合

你后面指出 frame_sync 原版 downsample 是一个中心采样点：

```text
m_os_factor / 2 + m_os_factor * ii - round(m_sto_frac * m_os_factor)
```

Python 这边做的是把这个口径推广成多个采样 origin 的 Stage-1 证据：

```text
origin = 0 .. os_factor-1
```

再融合每个 origin 的 per-bin power。这个不是完整替代 C++ frame_sync 的每 branch STO/SFO/CFO 估计，而是先在 Python selector 里验证“遍历采样相位是否能补证据”。

raw multi-origin 的特点很明显：

```text
rescue 很多，但 break 也很多
```

所以后面才加了 gate。

### 7. 增加了 weighted_unit 相位融合

只融合 power 会出现一个问题：DP 用的 complex phase 仍来自固定 origin，和 multi-origin power 不完全一致。

所以加了：

```text
--multi-phase-mode weighted_unit
```

做法是用各 origin 在目标 bin 上的归一化 power 给 unit complex phase 加权。它不是大杀器，但能把固定 origin 的 501 errors 推到 499，再配 symbol gate 到 497。

### 8. 增加 packet gate 和 symbol gate，而不是放开全包改写

原设想里说不要 `changed_symbols <= 2` 这种微观限制，这点我同意。但 raw multi-origin/island 直接放开后 break 太高，所以现在不是限制“最多改几个”，而是限制“哪些证据形态才允许接管”。

当前 symbol gate 条件：

```text
multi_trajectory_score - dual_trajectory_score >= 0.005618281741596176
old_power[multi_bin] / old_power_peak >= 0.7303704876428979
```

这两个量都来自运行时 FFT-bin evidence，不用 CRC、codec、payload template 或 GT。

## 为什么没有达到预期的 1-2 dB

目前残差错误的主要形态不是“路径连不上”，而是“局部候选证据不够会说话”。

关键诊断：

```text
product fusion 后，默认 locked anchor 里的剩余错误只有 21/515
更严格 anchor 可以把 locked error 压到 0

fusion-v1 miss 中：
Top40 union 覆盖 320/515
Top128 union 覆盖 412/515

但 GT 即使在候选集合里，简单 power / reconstruction score 也经常排不进 Top1
```

这说明继续调 DP 平滑惩罚、branch jump penalty、accept margin，收益会越来越小。真正要 1-2 dB，得让低置信 symbol 的 GT 局部分数更像 GT。

## 当前推荐复现实验命令

```powershell
python weak_decoder\phase_line\variants\island_dp_reconstruction\evaluate_multi_origin.py `
  --datasets 0_0_0_10_14_32 `
  --snrs -25 -26 -27 -28 -29 `
  --seeds 42 43 44 45 46 `
  --max-packets 10 `
  --multi-mode sum_norm `
  --multi-phase-mode weighted_unit `
  --enable-gate `
  --enable-symbol-gate `
  --symbol-min-trajectory-gain 0.005618281741596176 `
  --symbol-min-old-power-margin -999 `
  --symbol-min-old-multi-norm 0.7303704876428979 `
  --output-dir weak_decoder\phase_line\variants\island_dp_reconstruction\_eval\multi_sum_weighted_unit_symbolgate_oldnorm32_seed42_46_snr25_29
```

## 后续建议

下一步不要优先堆更多 DP 变体。更值得做的是：

```text
1. 把 frame_sync/downsample 的采样相位分支在 Python 里先验证彻底。
2. 对每个 branch/origin 的 STO/SFO/CFO 口径做一致性诊断。
3. 重新设计 candidate-local score，让 GT 在低置信 island 内更常进 Top1。
4. 保留 anchor hard lock 和 symbol gate，避免 raw rescue 把 break 一起放大。
```

我的判断是：孤岛 DP 的架构可以留，但真正能带来 1-2 dB 的不会是“更会走路的 DP”，而是“更准的每符号/每分支局部证据”。
