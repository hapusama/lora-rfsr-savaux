# Phase Viterbi / DP 实现说明

## 为什么图里写 DP，其实就是 Viterbi

这里的 `first-order phase DP` 和 `second-order phase DP` 都是 Viterbi-style dynamic programming。

它们不是逐点贪心，也不是拟合一条 phase line，而是在整段 payload 的 Top-L candidate graph 上找一条总分最高的 raw FFT bin path：

```text
每个 payload symbol t:
  candidates[t] = Top-L raw FFT bins from multi-offset evidence

目标:
  找 x_0, x_1, ..., x_{N-1}
  其中 x_t 属于 candidates[t]
  使整条 path 的 local evidence 高，同时相位变化不要乱跳
```

代码入口：

```text
weak_decoder/phase_line/selector.py
  select_phase_viterbi_path(...)
```

配置入口：

```text
weak_decoder/phase_line/configs.py
  PhasePathSelectorConfig
```

绘图脚本：

```text
scripts/experiments/phase_line/plot_phase_path_examples.py
```

## Stage 1：候选集生成

第一阶段仍然使用 multi-offset FFT evidence，只做 proposal generator。

对每个 payload symbol 取 Top-L：

```text
c_{t,l} = {
  raw_bin,
  phase = angle(center_fft[raw_bin]),
  energy_score,
  coherence_score,
  rank_score
}
```

局部分数：

```text
local_score =
  energy_weight    * energy_score
  + coherence_weight * coherence_score
  + rank_weight      * rank_score
  + optional_top1_soft_bonus
```

当前最好配置里：

```text
energy_weight = 0.25
coherence_weight = 0.40
rank_weight = 0.05
```

注意：这说明当前好结果里 offset coherence 仍然很强，phase smoothness 不是唯一主证据。

## First-order phase Viterbi

一阶版本惩罚相邻两个候选点之间的相位跳变：

```text
r1 = wrap(phi_t - phi_{t-1})
```

更完整地写，令第 `t` 个 payload symbol 的第 `i` 个候选为：

```text
c_{t,i}
```

它的 raw FFT bin、相位和局部分数分别是：

```text
b_{t,i}
phi_{t,i}
q_{t,i}
```

其中：

```text
q_{t,i}
= w_E E_{t,i}
+ w_C C_{t,i}
+ w_R R_{t,i}
+ B_{t,i}
```

`E` 是 multi-offset energy score，`C` 是 offset coherence score，`R` 是候选 rank score，`B` 是可选的 high-confidence Top-1 soft bonus。

一阶相位残差：

```text
r1(t, j, i)
= wrap(phi_{t,i} - phi_{t-1,j})
```

一阶转移代价：

```text
P1(t, j, i)
= lambda1 * Huber(r1(t, j, i) / sigma1)
```

递推公式：

```text
D_0(i) = q_{0,i}

D_t(i)
= q_{t,i}
+ max_j [
    D_{t-1}(j) - P1(t, j, i)
  ]
```

回溯指针：

```text
prev_t(i)
= argmax_j [
    D_{t-1}(j) - P1(t, j, i)
  ]
```

最后：

```text
i*_{N-1} = argmax_i D_{N-1}(i)
```

然后沿 `prev_t(i)` 从后往前回溯，得到整条 raw FFT bin path。

转移分数：

```text
score(t, curr) =
  max_prev [
    score(t-1, prev)
    + local_score(curr)
    - lambda1 * Huber(r1 / scale1)
  ]
```

直观含义：

```text
相位允许变化，但不希望每个 symbol 之间突然乱跳。
```

在当前代码里，为了和二阶实现共用一个框架，状态仍然写成 `(prev_candidate, curr_candidate)`，但当 `phase_order=1` 时，转移只用 `prev -> curr` 的 `r1`。

已测结果里，first-order Viterbi 反而是最稳的 phase path 版本。

## Second-order phase Viterbi

二阶版本惩罚连续三个候选点的相位二阶差分：

```text
r2 = wrap(phi_t - 2 * phi_{t-1} + phi_{t-2})
```

二阶版本的状态必须记住前两个候选，所以状态写成：

```text
D_t(j, i)
```

含义是：

```text
处理到第 t 个 payload symbol，
第 t-1 个 symbol 选择候选 j，
第 t 个 symbol 选择候选 i，
此时的最优累计得分。
```

一阶残差：

```text
r1(t, j, i)
= wrap(phi_{t,i} - phi_{t-1,j})
```

二阶残差：

```text
r2(t, k, j, i)
= wrap(phi_{t,i} - 2 phi_{t-1,j} + phi_{t-2,k})
```

转移惩罚：

```text
P(t, k, j, i)
= lambda1 * Huber(r1(t, j, i) / sigma1)
+ lambda2 * Huber(r2(t, k, j, i) / sigma2)
```

初始化：

```text
D_1(j, i)
= q_{0,j}
+ q_{1,i}
- lambda1 * Huber(
    wrap(phi_{1,i} - phi_{0,j}) / sigma1
  )
```

如果 `lambda1 = 0`，初始化就是：

```text
D_1(j, i) = q_{0,j} + q_{1,i}
```

二阶递推公式：

```text
D_t(j, i)
= q_{t,i}
+ max_k [
    D_{t-1}(k, j)
    - P(t, k, j, i)
  ]
```

回溯指针：

```text
prev_t(j, i)
= argmax_k [
    D_{t-1}(k, j)
    - P(t, k, j, i)
  ]
```

终止：

```text
(j*, i*) = argmax_{j,i} D_{N-1}(j, i)
```

然后用 `prev_t(j, i)` 从 `(j*, i*)` 往前回溯，得到完整路径：

```text
x_0, x_1, ..., x_{N-1}
```

最终输出 raw FFT bin：

```text
b_{0,x_0}, b_{1,x_1}, ..., b_{N-1,x_{N-1}}
```

状态：

```text
DP[t][prev_candidate][curr_candidate]
```

转移：

```text
score(t, prev, curr) =
  max_prevprev [
    score(t-1, prevprev, prev)
    + local_score(curr)
    - lambda1 * Huber(r1 / scale1)
    - lambda2 * Huber(r2 / scale2)
  ]
```

直观含义：

```text
相位可以一直下降，也可以有曲率；
但“相位变化速度”不要突然跳。
```

复杂度大约：

```text
O(N * L^3)
```

在当前实验中，`N≈35`、`L=24`，完全跑得动。

## 为什么 second-order 目前没有赢

诊断图里可以看到一个关键现象：

```text
GT path 的二阶差分确实比 random Top-L path 更小；
但是 second-order Viterbi 会选出比 GT 还过度平滑的 path。
```

这说明二阶 smoothness 有信息，但不能作为强主导项。

如果 `lambda2` 太大，算法会偏向一种“看起来很平滑，但 bin 是错的”的轨迹。

所以当前结果是：

```text
first-order Viterbi + coherence-local evidence
  比 second-order Viterbi 更稳
```

## Header slope 版本

`phase_dp_first_header` 不是把 header phase 作为全局 line anchor。

它只在 payload 开头弱惩罚 slope mismatch：

```text
r_header =
  wrap((phi_1 - phi_0) - header_slope)
```

并且只作用前几个 payload symbol。

当前 `-22 dB` 和 `-24 dB` 的测试里它没有改善，所以图里和 first-order payload-only 基本重合。

## 当前实现状态

目前已经实现并画图的 Viterbi 版本：

| 图中名称 | 实际算法 |
|---|---|
| `first-order phase Viterbi (DP)` | payload-only 一阶相位 Viterbi |
| `first-order Viterbi + header slope` | 一阶 Viterbi + payload 开头 header slope 弱约束 |
| `second-order phase Viterbi (DP)` | payload-only 二阶相位 Viterbi |
| `second-order Viterbi + soft anchor` | 二阶 Viterbi + 高置信 Top-1 soft bonus |

最重要的判断：

```text
VTB/DP 是有的，只是之前图标题写得太像普通 DP。
当前最佳表现来自 first-order Viterbi + 较强 coherence evidence；
second-order Viterbi 目前没有稳定优势。
```
