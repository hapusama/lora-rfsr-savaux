# Phase-Smooth Path Decoding 消融报告（2026-06-19）

## 结论

这轮把附件里建议的方案按物理层 raw FFT bin 级别逐项试了一遍。最有价值的结果是：

- `payload-only first-order phase DP + local coherence` 在 `0_0_0_10_14_16` 上能小幅接近或超过 v3 的 symbol SER。
- `second-order phase DP` 没有成为主增益来源；默认二阶权重大时会选择“过度平滑但错误”的 path。
- `second-order DP + high-confidence soft anchor` 当前没有带来额外收益。
- `header weak slope constraint` 在已测的 `-22 dB`、`-24 dB` 上没有变化。
- 当前还不能说“比 v3 稳定高 1 dB”。更准确的说法是：phase-smooth DP 有局部机会，但要接近 v3 仍需要较强的 offset coherence 局部证据。

因此，这条路线可以继续作为“phase-smooth sequence-level selector”保留，但不能直接作为论文主结果。若要投稿，必须避免把较高 coherence 权重带来的收益误称为 phase-line 主收益。

## 已实现代码

- `configs.py`
  - 新增 `PhasePathSelectorConfig`
  - 支持 first-order / second-order phase smoothness
  - 支持 high-confidence soft anchor
  - 支持 header 开头弱 slope 约束

- `selector.py`
  - 新增 `select_phase_viterbi_path(...)`
  - 使用 payload-only Viterbi，不拟合 packet-local phase line
  - 状态为 `DP[t][prev_candidate][curr_candidate]`
  - CRC 不参与搜索

- `scripts/experiments/phase_line/run_phase_path_ablation.py`
  - 同时比较：
    - `center`
    - `multi`
    - `v3`
    - 当前 `phase_line`
    - `phase_dp_first`
    - `phase_dp_first_header`
    - `phase_dp_second`
    - `phase_dp_second_anchor`
  - 输出 per-packet 指标、SNR 曲线、GT / random / selected path 的二阶平滑度。

## 核心配置

当前表现最好的配置不是 phase-only，而是：

```text
top_l = 24
energy_weight = 0.25
coherence_weight = 0.40
rank_weight = 0.05
first_order_weight = 0.18
second_order_weight = 0.05
top1_soft_bonus = 0.05
```

这说明需要很强的局部 offset coherence 才能追上 v3。phase smoothness 主要在相邻候选接近时做 sequence-level tie-break。

## 主要结果

数据集：`0_0_0_10_14_16`

输出目录：

```text
data/phase_path_ablation_best_coh040_m22_m26/
```

| SNR(dB) | Top-L recall | multi SER | v3 SER | old phase-line SER | first-order DP SER | second-order DP SER | second-order + anchor SER |
|---:|---:|---:|---:|---:|---:|---:|---:|
| -22 | 0.969 | 0.179 | 0.060 | 0.270 | 0.078 | 0.075 | 0.075 |
| -23 | 0.925 | 0.319 | 0.166 | 0.431 | 0.148 | 0.158 | 0.158 |
| -24 | 0.855 | 0.509 | 0.296 | 0.644 | 0.283 | 0.306 | 0.306 |
| -25 | 0.730 | 0.644 | 0.436 | 0.686 | 0.410 | 0.462 | 0.462 |
| -26 | 0.610 | 0.779 | 0.587 | 0.784 | 0.574 | 0.616 | 0.616 |

SER 观察：

- `first-order DP` 在 `-23` 到 `-26 dB` 小幅优于 v3。
- `-22 dB` 时 v3 更好。
- `second-order DP` 通常不如 first-order DP。
- 旧 `phase_line` rerank 明显不适合继续作为主线。

CRC 观察：

| SNR(dB) | v3 CRC | first-order DP CRC | second-order DP CRC |
|---:|---:|---:|---:|
| -22 | 0.364 | 0.091 | 0.091 |
| -23 | 0.000 | 0.091 | 0.091 |
| -24 | 0.000 | 0.000 | 0.000 |
| -25 | 0.000 | 0.000 | 0.000 |
| -26 | 0.000 | 0.000 | 0.000 |

CRC 没有形成稳定收益。即使 SER 小幅下降，完整 payload 仍容易有残余错误。

## Phase smoothness 诊断

| SNR(dB) | GT second abs/pi | random Top-L p10 | v3 second abs/pi | first-order DP second abs/pi | second-order DP second abs/pi |
|---:|---:|---:|---:|---:|---:|
| -22 | 0.264 | 0.438 | 0.293 | 0.247 | 0.238 |
| -23 | 0.288 | 0.441 | 0.353 | 0.268 | 0.239 |
| -24 | 0.310 | 0.439 | 0.418 | 0.310 | 0.237 |
| -25 | 0.336 | 0.432 | 0.469 | 0.328 | 0.247 |
| -26 | 0.366 | 0.435 | 0.476 | 0.339 | 0.237 |

这个诊断说明：

- GT path 的二阶相位差分确实明显小于 random Top-L path。
- `first-order DP` 选出的 path 在二阶平滑度上接近 GT。
- `second-order DP` 经常比 GT 还平滑，说明它会偏向“虚假的超平滑错误路径”。

## Header 约束结果

测试目录：

```text
data/phase_path_ablation_header_coh040_m22/
data/phase_path_ablation_header_coh040_m24/
```

结果：

| SNR(dB) | payload-only first-order DP SER | header weak slope DP SER |
|---:|---:|---:|
| -22 | 0.078 | 0.078 |
| -24 | 0.283 | 0.283 |

header slope 只影响 payload 开头，不惩罚绝对相位跳变；目前没有观察到收益。

## 判断

这批方法都试过后，最合理的技术判断是：

1. Top-L proposal generator 是正确的，`-22 dB` 仍有 `0.969` recall，但到 `-26 dB` 只剩 `0.610`，后端 path selector 无法弥补候选缺失。
2. phase smoothness 有真实信号，但它不是足够强的单独主证据。
3. 二阶 smoothness 不能直接加大权重，否则会奖励错误 path。
4. 目前最稳的是 first-order phase DP + coherence-local evidence，而不是 second-order DP。
5. 若继续推进，应把研究问题改成：

```text
phase-smooth sequence-level correction can slightly improve coherence-based weak-bin selection,
but the dominant evidence is still multi-offset coherence / local FFT reliability.
```

这不是你最初想要的“phase line 主导弱包解调”，但可以作为一个更诚实的物理层 ablation 结果。
