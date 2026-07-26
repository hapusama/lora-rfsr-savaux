# Phase-Line 滑动窗口方向停止报告

日期：2026-06-19

结论先说：

```text
当前 phase-line / sliding-window / delayed-decision 原型没有比 v3 更稳，也没有接近“比 v3 稳定高 1 dB”的目标。
在现有 raw FFT candidate phase 口径下，继续靠拟合全局 line、局部 line 或 clean/local oracle line 做 candidate rerank，意义不大。
本方向先停止继续调参和继续拟合。
```

## 1. 当前代码保存位置

核心代码：

```text
weak_decoder/phase_line/configs.py
weak_decoder/phase_line/trajectory.py
weak_decoder/phase_line/selector.py
weak_decoder/phase_line/__init__.py
```

实验脚本：

```text
scripts/experiments/phase_line/run_phase_line_threshold_sweep.py
scripts/experiments/phase_line/diagnose_phase_candidate_ranking.py
```

说明文档：

```text
weak_decoder/phase_line/README.md
weak_decoder/phase_line/PHASE_LINE_STOP_REPORT_2026-06-19.md
```

已生成诊断结果：

```text
data/phase_line_diagnostics/candidate_phase_ranking_0_0_0_10_14_16_m22_local.summary.json
data/phase_line_diagnostics/candidate_phase_ranking_0_0_0_10_14_16_m24_local.summary.json
```

## 2. 当前如何利用相位

当前实现是两阶段结构：

```text
Stage 1:
  multi-offset fused energy 生成每个 payload symbol 的 Top-L candidates。
  默认 Top-L = 24。
  这一步只负责高召回候选集，不作为最终论文主贡献。

Stage 2:
  尝试用 payload 内局部相位平滑性做候选路径选择。
```

具体尝试过两类第二阶段。

### 2.1 Causal phase beam

思路：

```text
从前往后走 payload symbols；
beam state 保存已选 candidate 的 unwrapped phase 历史；
用最近窗口预测当前 symbol phase；
低置信 symbol 给 phase continuity 更高权重；
高置信 symbol 提高 energy / coherence 权重。
```

问题：

```text
它会自我强化错误轨迹。
一旦前面选错，后面的预测相位会被错误 candidate 带偏。
```

所以该路径已不作为默认 selector。

### 2.2 Delayed local-anchor selector

思路：

```text
先识别高可靠 Top-1 symbols 作为 payload-local anchors；
对目标 symbol 附近 anchors 拟合局部 phase reference；
低置信 symbol 只有当非 Top-1 candidate 的 phase 分数明显优于 Top-1，
且能量跌落受限时，才允许替换。
```

这里 phase 的作用是：

```text
不是直接替代 FFT evidence；
而是在 Top-L candidate 内作为 guarded override 的依据。
```

但实际效果仍不理想，见下一节。

## 3. 当前效果

主要 probe 数据集：

```text
dataset = 0_0_0_10_14_16
SNR = -22, -24 dB
reference power 来自历史 low_snr_gt_bin metadata:
signal_reference_power = 7.297427983845209e-05
seed = 20260531
```

### 3.1 Threshold sweep 对比

输出目录：

```text
data/phase_line_selector_probe_local_anchor/
data/phase_line_selector_probe_local_clean_oracle/
```

代表性结果：

```text
SNR -22 dB:
  v3 coherence-selected SER      = 0.060
  phase_line_selected SER        = 0.270

SNR -24 dB:
  v3 coherence-selected SER      = 0.296
  phase_line_selected SER        = 0.644
```

结论：

```text
当前 phase-line selector 明显弱于 v3。
它不是差 1 dB，而是当前候选判据本身不成立。
```

### 3.2 Clean/local phase oracle 诊断

为了确认是不是“无 GT 的 phase line 估计不准”，做了 clean/local oracle：

```text
使用 clean header-first CSV 中的 payload peak_phase 作为 GT phase reference；
再在 noisy Top-L candidates 里按 phase residual / phase+energy 排名。
```

结果仍然不好。

候选级诊断脚本：

```text
scripts/experiments/phase_line/diagnose_phase_candidate_ranking.py
```

关键统计：

```text
SNR -22 dB:
  GT recall@24 = 0.969
  v3 error rows = 11
  v3 错误处 GT local phase rank=1 rate = 0.182
  v3 错误处 GT local phase rank<=8 rate = 0.364

SNR -24 dB:
  GT recall@24 = 0.855
  v3 error rows = 58
  v3 错误处 GT local phase rank=1 rate = 0.103
  v3 错误处 GT local phase rank<=8 rate = 0.448
```

解释：

```text
Top-L 里确实经常包含 GT；
但 GT 在 raw phase residual 排名里通常不靠前；
错误 bin 经常在 modulo phase 上比 GT 更贴 clean/local phase line。
```

这说明问题不是“line 拟合得不够好”，而是：

```text
raw FFT candidate phase 到 phase line 的单点 residual，不具备足够候选判别力。
```

## 4. 为什么现在停下来

当前用户目标是：

```text
如果滑动窗口不行就停下来；
不要继续做没有意义的拟合。
```

根据上面的结果，继续沿以下方向调参意义不大：

```text
global phase line fitting
local sliding-window phase line fitting
clean/local oracle line residual ranking
phase residual 直接作为 candidate 主排序
```

这些方法都绕不开同一个事实：

```text
错误 candidate 的 raw FFT phase 也能贴上线。
```

所以本分支先冻结为实验原型和负结果记录。

## 5. 如果以后还要利用相位，不能再怎么做

不建议继续：

```text
candidate_score = mostly phase_residual_to_line
```

也不建议继续投入：

```text
更多 line 拟合技巧
更复杂滑动窗口拟合
只靠 phase residual 的 beam / Viterbi
```

## 6. 以后可能有意义的相位方向

如果未来重新打开 phase 方向，需要先改变相位特征本身，而不是继续拟合：

```text
1. bin-dependent deterministic phase correction
   raw FFT phase 可能包含 symbol/bin/reference 的确定性相位项。
   先把这些项建模或消掉，再谈 phase-line residual。

2. pairwise phase increment
   不看 phi_k 是否贴 line，而看候选路径的相邻相位增量是否稳定。

3. second-order phase smoothness
   看 phi_k - 2 phi_{k-1} + phi_{k-2}，减少绝对相位 offset 的影响。

4. phase-consistent subset gate
   phase 只负责缩小候选集合，不直接选最终 bin；
   最终仍由 energy / coherence 在 subset 内选。

5. candidate phase reliability model
   先判断当前 symbol 的 phase 是否可信；
   phase 不可信时不要让它参与判决。
```

但这些都是新问题，不应继续包装成“再拟合一个更好的 phase line”。

## 7. 当前代码状态建议

保留：

```text
weak_decoder/phase_line/
scripts/experiments/phase_line/
data/phase_line_diagnostics/
```

用途：

```text
作为 phase-line/sliding-window 方向的实验原型、负结果和后续复盘依据。
```

不要把当前 `phase_line_selected` 写进论文主结果。

当前可以在论文或内部报告中表述为：

```text
We investigated a phase-smooth candidate path selector using multi-offset FFT
as candidate generation.  However, raw FFT-bin phase residuals were not
discriminative enough inside the Top-L candidate set: wrong bins often matched
the local phase trajectory better than the ground-truth bin.  Therefore this
branch was not used as the final demodulator.
```
