# Handoff: Phase-Assisted Weak LoRa Demodulation

Date: 2026-06-19

## 当前目标

当前目标是做低 SNR LoRa payload FFT bin 选择：利用正确 payload bin 在包内呈现稳定相位轨迹这一现象，作为 Top-L candidate rerank 的辅助约束。

核心问题可以抽象成：

```text
每个 payload symbol 有 L 个候选 FFT bin。
每个候选有 energy、multi-offset coherence、complex FFT phase、bin index。
正确候选序列的 unwrapped phase 通常服从 packet-local smooth trajectory。
实际解码时 GT trajectory 不知道，需要边选候选边估 trajectory。
CRC 只做最终验证，不参与搜索。
```

## 当前实验链路

当前低 SNR selector 实验不是每个 noisy SNR 都重新跑完整 frame_sync，而是 header-first / post-sync 实验：

- 已知 packet timing、CFO、header、payload_len 等参数。
- 在 noisy IQ 上提取 payload symbol 的 FFT evidence。
- 每个 payload symbol 做 multi-offset FFT evidence：
  - 多个 sample offset 下 downsample
  - dechirp
  - FFT
  - 多 offset 能量归一化融合
- 高置信 symbol 锁 Top-1。
- 低置信 symbol 保留 Top-L，当前默认 Top-24。
- rerank 当前主要依赖：
  - candidate energy
  - multi-offset coherence
  - packet-local phase-line residual

CRC 只做最终验证，不参与候选搜索。

## 关键实验发现

### Payload GT phase line 很稳

在 dataset `0_0_0_10_14_16`、packet 10、payload len 33、SNR -22 到 -27 dB 下，payload GT raw FFT bin 的 complex phase 仍然形成稳定轨迹。

典型结果：

```text
payload GT phase-line R2 ≈ 0.95 ~ 0.96
```

这说明正确 bin 的复数相位在极低 SNR 下没有完全被噪声打散，仍然携带很强的 packet-local structure。

### Argmax / selector 选错时相位明显偏离

argmax 或当前 selector 选错时，对应候选的 phase 往往不贴 GT phase trajectory。

因此 phase line 可以作为弱包 rerank 的辅助证据，尤其是在候选能量接近、offset coherence 也不明显时。

### Offset coherence 仍是主增益

当前观察里，offset coherence 是最主要的增益来源。

phase line 更适合作为 Top-L candidate rerank 的辅助项，而不是替代 energy/coherence。

## Preamble / Sync / SFD 与 Payload 相位关系

已经做过同网格图：用 `fine_payload_start` 反推全包网格：

```text
preamble_start = fine_payload_start - (preamble_len + 4.25) * chirp_samples
```

对 packet 10：

```text
fine_payload_start = 16021471
fine_payload_backtrack preamble_start = 15938527
synced_preamble_start = 15938627
delta = -100 samples = -25 chips
```

相关脚本：

```text
gr-lora_sdr/weakPacket_decoding copy/scripts/experiments/plot_gt_phase_sweep.py
```

新增/当前支持模式：

```text
--front-grid fine_payload_backtrack
--front-grid frame_sync_raw
```

含义：

- `fine_payload_backtrack`：preamble/sync/SFD/header/payload 都放在同一个 payload fine grid 上。
- `frame_sync_raw`：只画 frame_sync 原始 preamble/sync/SFD。

生成图和 CSV 在：

```text
gr-lora_sdr/weakPacket_decoding copy/data/phase_line/gt_preamble_header_payload_sweep/
```

重要输出：

```text
0_0_0_10_14_16_packet_010_fine_payload_backtrack_gt_phase_snr_m22_m27.png
0_0_0_10_14_16_packet_010_frame_sync_raw_gt_phase_snr_m22_m27.png
0_0_0_10_14_16_packet_010_fine_payload_backtrack_gt_phase_points.csv
0_0_0_10_14_16_packet_010_fine_payload_backtrack_gt_phase_summary.csv
```

运行命令：

```powershell
D:\mysoft2\miniconda3\envs\gr-lora\python.exe "gr-lora_sdr/weakPacket_decoding copy/scripts/experiments/plot_gt_phase_sweep.py" --front-grid fine_payload_backtrack

D:\mysoft2\miniconda3\envs\gr-lora\python.exe "gr-lora_sdr/weakPacket_decoding copy/scripts/experiments/plot_gt_phase_sweep.py" --front-grid frame_sync_raw
```

### 重要结论

即使使用同一个 fine payload 反推网格，并使用同款 CFO common-phase correction，preamble/sync/SFD 到 header/payload 仍存在明显相位断层。

clean IQ 中也存在这个断层，所以它不是低 SNR 噪声，也不是 selector 选错。

当前判断：

```text
raw FFT bin complex phase 不是纯 carrier/channel phase。
它包含 field/reference/symbol/bin 的确定性相位项。
```

所以不要强行要求 preamble/sync/SFD 与 payload 落在同一条绝对 phase line 上。

更可靠的分工：

```text
preamble/sync/SFD: 用于 timing / CFO / SFO / 同步质量估计
header/payload: 用于建立 data-section packet-local phase trajectory
payload Top-L candidates: 用 phase residual + offset coherence + energy 做 rerank
```

如果后续要利用 preamble/sync/SFD 的相位，需要引入 field-to-data phase offset，而不是直接用一条全包绝对相位线。

## 推荐算法方向

这个问题不应只检索 `LoRa phase decoding`，更应检索“带轨迹先验的离散候选序列选择”。

推荐关键词：

```text
track-before-detect
multiple hypothesis tracking
Viterbi with smoothness prior
dynamic programming path through candidate points
joint data association and trajectory estimation
robust curve fitting with latent inlier assignment
RANSAC curve fitting multiple candidates
EM for hidden trajectory estimation
factor graph candidate sequence selection
HMM with continuous latent phase state
Kalman filter with discrete measurement association
particle filtering data association
blind phase estimation
decision-directed phase tracking
ridge tracking in time-frequency maps
phase unwrapping candidate selection dynamic programming
circular regression with outliers
```

## 给 AI / 文献检索的总 Prompt

```text
I am working on low-SNR LoRa/CSS demodulation. For each payload symbol, I have Top-L FFT-bin candidates. Each candidate has energy, multi-offset coherence, complex FFT phase, and bin index.

Empirically, the correct payload FFT-bin sequence has a strong packet-local phase trajectory: the unwrapped complex phase of the correct bin across symbols is smooth and often approximated by a line or low-order curve. During actual decoding, the ground-truth curve is unknown.

I need algorithms that jointly select one candidate per symbol and estimate the hidden smooth phase trajectory. CRC is only final validation, not used during search.

Please search related work beyond LoRa: communications, signal processing, tracking, graphical models, dynamic programming, robust regression, factor graphs, Viterbi, Kalman/particle filtering, multiple hypothesis tracking, track-before-detect, RANSAC/EM curve fitting, ridge tracking.

For each relevant method, explain:
1. What problem it solves.
2. Why it matches this Top-L candidate selection problem.
3. Objective/probabilistic model.
4. Handling of outliers, phase wrapping, missing anchors, unknown curve parameters.
5. How to adapt it to phase-assisted LoRa weak-packet demodulation.
```

## 可能的实现路线

### 1. Beam Search / Viterbi Baseline

每个 symbol 保留 Top-L candidate，路径代价包括：

```text
score = energy_score
      + offset_coherence_score
      - phase_residual_penalty
      - smoothness_penalty
```

相位部分可先用一阶差分或二阶差分约束：

```text
first-order:  unwrap(phi_t - phi_{t-1}) should be stable
second-order: phi_t - 2 phi_{t-1} + phi_{t-2} should be small
```

适合作为第一个工程 baseline。

### 2. RANSAC / EM Robust Curve Fit

思路：

1. 从高置信 anchors 初始化 phase line。
2. 对低置信 symbol，在 Top-L 中选 residual 最小的 candidate。
3. 用当前选中的 candidate 重新拟合 line / polynomial。
4. 迭代，允许 outlier 权重降低。

优点：

- 简单。
- 容易利用已有 high-confidence locks。
- 可处理少量错误候选。

风险：

- 初始化不好时可能收敛到错误轨迹。
- phase wrapping 要小心处理。

### 3. HMM / Factor Graph / MAP

建模：

```text
latent state: phase trajectory parameters or local phase/slope
observation: Top-L candidate complex phase/energy/coherence
transition: phase/slope slowly varying
emission: candidate residual + local evidence likelihood
```

用 max-product / Viterbi / particle filtering / Kalman mixture 做 MAP path。

优点：

- 形式最完整。
- 可自然加入 energy、coherence、phase residual。
- 可处理 missing/low-confidence symbols。

缺点：

- 实现复杂度高。
- 参数调节更多。

## 当前工作区注意事项

根目录当前存在这些状态，不要误动：

```text
D  _gen_run.py
D  test_creation.py
?? lora_low_snr_phy_flowchart.png
```

`gr-lora_sdr/weakPacket_decoding copy/` 似乎不在当前 git tracking 里，`git diff` 不会显示那里的改动。

## 一句话结论

目前最值得继续推进的是：

```text
不要再要求 preamble 到 payload 是一条绝对相位线；
真正可利用的是 payload data-section 内正确 bin 的 packet-local smooth phase trajectory。

下一步应把问题抽象为：
Top-L candidate sequence selection under an unknown smooth circular phase trajectory prior.
```
