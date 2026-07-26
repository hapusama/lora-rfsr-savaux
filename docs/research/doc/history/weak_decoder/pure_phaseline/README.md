# Pure Phase-Line 消融实验

这个目录用于验证一个简单问题：

如果不使用当前系统里的两阶段解码，也就是不先用 Savaux / Top-K evidence
生成较可靠候选，而是直接靠 phase-line 从 payload FFT 频谱里选 bin，效果会怎样？

实验结论很明确：**pure phase-line 表现很差，接近甚至差于 Argmax**。这说明当前
Phase-Line 的收益并不是来自“相位线单独万能”，而是来自：

1. 第一阶段先把候选空间压干净；
2. 第二阶段再用相位连续性在较可靠候选里做路径选择。

也就是说，两阶段解码是必要的。

## 实现内容

核心代码：

```text
weak_decoder/pure_phaseline/pure_phase_line.py
```

实验入口：

```text
weak_decoder/pure_phaseline/run_pure_phase_line_experiment.py
```

当前实现提供两种 pure 模式：

1. `--selection-mode line`

   直接按 phase-line 的绝对相位残差逐 symbol 选 bin。

2. `--selection-mode path`

   不走 Savaux stage-1，也不走当前 `phase_line/selector.py` 的候选生成和 Viterbi
   逻辑；只在每个 symbol 的宽候选上，用相邻 symbol 的相位连续性做一个纯路径 DP。

这两个模式都不使用 CRC、payload byte search、包结构模板或跨包先验来选择结果。
CRC 只作为输出诊断。

## 复现实验命令

主实验使用 `path + Top-32`，数据集和之前 baseline 对比一致：

```powershell
python "gr-lora_sdr\weakPacket_decoding copy\weak_decoder\pure_phaseline\run_pure_phase_line_experiment.py" `
  --dataset 0_0_0_10_14_16 `
  --snrs -22 -23 -24 -25 -26 `
  --seeds 42 43 44 45 46 `
  --selection-mode path `
  --candidate-mode top_l `
  --top-l 32 `
  --output-dir "gr-lora_sdr\weakPacket_decoding copy\data\baseline_comparison\pure_phaseline_0_0_0_10_14_16_m22_m26_after_intercept_fix"
```

输出文件：

```text
data/baseline_comparison/pure_phaseline_0_0_0_10_14_16_m22_m26_after_intercept_fix/
  per_packet_metrics.csv
  snr_summary.csv
  summary.json
  pure_vs_baselines_ser.csv
  pure_vs_baselines_ser.png
  pure_vs_baselines_ser.pdf
```

## SER 结果

数据集：`0_0_0_10_14_16`

每个 SNR 使用 5 个 noise seed：`42, 43, 44, 45, 46`

每个 SNR 共 55 个有效 packet。

```text
SNR(dB)   Best Phase-Line   Savaux      Argmax      Pure Phase-Line
-22       0.0062            0.0109      0.6644      0.6519
-23       0.0275            0.0390      0.7678      0.8099
-24       0.0992            0.1231      0.8535      0.8836
-25       0.2151            0.2473      0.9096      0.9309
-26       0.3860            0.4078      0.9418      0.9616
```

平均 SER：

```text
Best Phase-Line   0.1468
Savaux            0.1656
Argmax            0.8274
Pure Phase-Line   0.8476
```

## 结果解释

Pure phase-line 的主要问题是候选空间太大。低 SNR 下，很多噪声 bin 的相位会偶然和
phase-line 对齐；如果没有第一阶段提供较可靠的候选集合，phase-line 很容易选出一条
“看起来很平滑”的错误路径。

当前最好结果 `Best Phase-Line` 使用的是：

```text
Savaux stage-1 evidence -> Phase-Line path selection
```

它比 pure phase-line 好很多，说明 Savaux / Top-K evidence stage 的作用不是可有可无的
预处理，而是 phase-line 能有效工作的前提之一。
