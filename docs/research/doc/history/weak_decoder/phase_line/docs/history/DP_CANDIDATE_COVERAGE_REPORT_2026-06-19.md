# Phase-DP 候选覆盖诊断报告（2026-06-19）

## 结论

当前 phase-DP 的失败可以分成两类：

1. **第一阶段候选缺失**：正确 raw FFT bin 没有进入 Top-L proposal set。  
   这种错误第二阶段 Viterbi / DP 无法修复。

2. **第二阶段选择失败**：正确 bin 已经在 Top-L 里，但 DP 没有选中。  
   这才是 phase smoothness / coherence / local score 需要继续优化的部分。

当前结果说明：低 SNR 下第一阶段已经是明显瓶颈。尤其到 `-25 dB`、`-26 dB`，很多正确 bin 根本不在 Top-24 里。

## 全 SNR 候选覆盖情况

数据集：

```text
0_0_0_10_14_16
```

配置：

```text
Top-L = 24
energy_weight = 0.25
coherence_weight = 0.40
rank_weight = 0.05
first_order_weight = 0.18
second_order_weight = 0.05
```

结果来自：

```text
data/phase_path_ablation_best_coh040_m22_m26/snr_curve_summary.csv
```

| SNR(dB) | GT in Top-24 | 候选缺失导致的 SER 下界 | v3 SER | first-order DP SER | second-order DP SER | first-order 超出下界 | second-order 超出下界 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| -22 | 0.969 | 0.031 | 0.060 | 0.078 | 0.075 | 0.047 | 0.044 |
| -23 | 0.925 | 0.075 | 0.166 | 0.148 | 0.158 | 0.073 | 0.083 |
| -24 | 0.855 | 0.145 | 0.296 | 0.283 | 0.306 | 0.138 | 0.161 |
| -25 | 0.730 | 0.270 | 0.436 | 0.410 | 0.462 | 0.140 | 0.192 |
| -26 | 0.610 | 0.390 | 0.587 | 0.574 | 0.616 | 0.184 | 0.226 |

解释：

```text
候选缺失导致的 SER 下界 = 1 - GT in Top-24
```

也就是说，如果 GT 没进候选集，DP 不可能选对。这个下界代表当前第一阶段 proposal generator 的理论限制。

观察：

- `-22 dB` 时 Top-24 覆盖率很高，DP 的错误主要来自第二阶段选择。
- `-24 dB` 后第一阶段候选缺失明显增加。
- `-26 dB` 时 Top-24 只覆盖 61.0%，单靠第二阶段不可能把 SER 压到 39% 以下。

## 代表例：-25 dB packet 10

图对应目录：

```text
data/phase_path_figures/0_0_0_10_14_16_snr_m25p0_packet_10/
```

候选诊断输出：

```text
data/phase_dp_candidate_coverage/
  0_0_0_10_14_16_snr_m25_to_m25_packet_10_summary.csv
  0_0_0_10_14_16_snr_m25_to_m25_packet_10_per_symbol.csv
```

这个包一共有 35 个 payload symbols。

第一阶段覆盖情况：

```text
GT in Top-24 = 25 / 35
GT missing   = 10 / 35
```

因此这个包无论第二阶段怎么做，理论最低 raw-bin SER 都是：

```text
10 / 35 = 0.286
```

### first-order phase Viterbi

```text
hit = 21 / 35
SER = 14 / 35 = 0.400
```

错误拆分：

```text
GT missing from Top-24: 10
GT present but DP chose wrong: 4
```

可救但选错的 symbol：

| payload idx | GT rank in Top-24 | GT bin | selected bin |
|---:|---:|---:|---:|
| 0 | 24 | 803 | 819 |
| 4 | 2 | 107 | 512 |
| 6 | 13 | 89 | 941 |
| 15 | 16 | 267 | 378 |

### second-order phase Viterbi

```text
hit = 18 / 35
SER = 17 / 35 = 0.486
```

错误拆分：

```text
GT missing from Top-24: 10
GT present but DP chose wrong: 7
```

可救但选错的 symbol：

| payload idx | GT rank in Top-24 | GT bin | selected bin |
|---:|---:|---:|---:|
| 0 | 24 | 803 | 819 |
| 4 | 2 | 107 | 512 |
| 6 | 13 | 89 | 941 |
| 15 | 16 | 267 | 378 |
| 30 | 16 | 826 | 258 |
| 31 | 17 | 904 | 903 |
| 32 | 5 | 760 | 856 |

这说明 second-order DP 在这个包上比 first-order 多犯了 3 个“本来可救”的错误。

## 判断

目前 DP 的问题不是单一原因：

```text
第一阶段 Top-L 候选集召回不足
+ 第二阶段在少数可救位置上仍会选错
```

其中：

- `-25 dB packet 10`：10/14 个 first-order DP 错误来自候选缺失。
- 这说明 first-order DP 在“GT 已经进候选集”的位置上其实选得还可以：25 个可选中选对了 21 个。
- second-order DP 更容易为了平滑 path 牺牲局部正确 bin，所以可救位置错得更多。

## 下一步建议

如果继续往这条线推进，优先级应该是：

1. **提高第一阶段 Top-L recall**  
   例如扩大 Top-L、加入 coherence-top candidates、降低 energy-drop filtering，或者设计更好的 offset proposal。

2. **单独研究 selectable miss**  
   只看 GT 已经在 Top-L 的 symbol，分析 DP 为什么没有选中。这里才是 phase smoothness 权重、coherence 权重和 transition penalty 该调的地方。

3. **不要只看最终 SER**  
   每次实验都应该同时报告：

```text
GT in Top-L recall
candidate-missing lower-bound SER
selectable miss rate
final selected SER
```

这样才能判断问题到底在第一阶段 proposal，还是第二阶段 path selection。
