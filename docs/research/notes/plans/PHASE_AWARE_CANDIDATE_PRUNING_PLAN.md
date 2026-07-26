# Phase-aware candidate pruning plan

本文档整理 `weakPacket_decoding copy` 当前目录中的脚本、实验流程、数据输出和下一阶段研究目标。当前阶段只关注第一阶段：

```text
每个 payload symbol: 2^SF 个 FFT bins -> Top-L 候选 bins
目标: GT-bin Recall@L 显著高于传统 argmax / energy Top-L baseline
约束: 不使用 payload template、计数器、跨包 joint prior；不做昂贵的全路径相位搜索
```

## 1. 当前目录结构和关键脚本

```text
weakPacket_decoding copy/
├── README.md                         主流程说明：弱检测 -> framesync -> header-first -> payload FFT peak
├── notes/plans/TWO_STAGE_WEAK_DECODER.md          当前 two-stage PHY-only 解码器说明和已验证结果
├── HANDOFF.md                         历史实验长记录
├── notes/plans/PHASE_AWARE_CANDIDATE_PRUNING_PLAN.md  当前第一阶段候选筛选入口
├── weak_decoder/                      核心 Python 模块
├── scripts/                           主链入口和 session/phase 实验脚本
├── scripts/experiments/               非主链科研实验、评估和诊断脚本
├── data/                              中间结果、低 SNR 数据、CSV/JSON/PNG 输出
├── noisy_iq/                          早期 noisy-IQ 工具包
└── doc/                               设计笔记和周报
```

关键模块：

- `weak_decoder/chirp.py`: gr-lora_sdr 兼容 chirp、dechirp FFT、bin 到 symbol 映射。
- `weak_decoder/preamble_detector.py`: 弱前导码检测，高召回候选生成器。
- `weak_decoder/frame_locator.py`: sync word + SFD 粗帧定界。
- `weak_decoder/grlora_frame_sync.py`: CFO/STO/SFO 估计和 framesync 验证。
- `weak_decoder/header_first_demod.py`: explicit header hard decode，并导出 header/payload symbol 级 FFT peak。
- `weak_decoder/phase_guided_demod.py`: header/preamble phase line、phase-guided scoring、历史 Phase-MAP/phase rescue 逻辑。
- `weak_decoder/payload_codec.py`: LoRa PHY 编解码、whitening、CRC、Hamming/interleaver 对齐。
- `weak_decoder/two_stage_weak_decoder.py`: 当前更可靠的 single-packet PHY-only two-stage 解码器。
- `weak_decoder/blind_payload_search.py` 和 `blind_payload_decoder.py`: 较早的 blind payload beam search 原型，文档较多但不应作为当前第一阶段主线。

关键脚本：

- `scripts/run_weak_sync_chain.py`: raw IQ 到 sync_chain CSV、STFT、framesync peak/spectrum。
- `scripts/run_header_first_demod.py`: 从 framesync-valid 候选导出 header/payload symbol CSV，是 GT-bin 的主要 clean 来源。
- `scripts/plot_payload_peak_trends.py`: 每包 payload selected peak 的相位/幅度趋势图。
- `scripts/experiments/run_low_snr_gt_bin_experiment.py`: 对 clean IQ 加 AWGN，强行读取 clean GT bin 的低 SNR 复数 FFT 值。
- `scripts/experiments/run_low_snr_wrong_bin_experiment.py`: GT bin、argmax、wrong_peak、offset/fixed wrong bins 的相位/幅度对照。
- `scripts/experiments/evaluate_phase_bin_metric.py`: 已有 phase-aware Top-L bin recall 评估脚本。
- `scripts/experiments/evaluate_codec_bin_metric.py`: codec-consistent block candidate Top-L recall 评估。
- `scripts/experiments/run_two_stage_weak_decoder.py`: two-stage decoder runner，支持 `center` 与 `multi-offset` FFT evidence。

## 2. 已有实验流程和输入输出

主链流程：

```text
raw complex64 IQ
  -> scripts/run_weak_sync_chain.py
  -> data/weak_sync_chain/sync_chain/*.csv
  -> scripts/run_header_first_demod.py
  -> data/weak_sync_chain/header_first/*_header_first_symbols.csv
  -> 低 SNR / candidate recall / two-stage decoder experiments
```

主输入：

- 原始 IQ: `../data/USRP_IQ/0_0_0_10_14_16.bin`、`0_0_0_10_14_8.bin`、`0_0_0_10_14_32.bin`。
- Clean GT symbol CSV:
  - `data/weak_sync_chain/header_first/0_0_0_10_14_16_header_first_symbols.csv`
  - `data/weak_sync_chain/header_first/0_0_0_10_14_8_header_first_symbols.csv`
  - `data/weak_sync_chain/header_first/0_0_0_10_14_32_header_first_symbols.csv`
- 低 SNR IQ:
  - `data/low_snr_gt_bin/<stem>/<stem>_snr_mXXdB.bin`
  - `data/low_snr_gt_bin/<stem>_extreme_snr/<stem>_snr_mXXdB.bin`

主输出：

- `data/weak_sync_chain/sync_chain/*.csv`: detection/frame/sync 主表。
- `data/weak_sync_chain/header_first/*_symbols.csv`: header/payload symbol 级 FFT peak、timing、CFO/SFO、payload_len/CR/CRC/LDRO。
- `data/low_snr_gt_bin/*/*_gt_bin_features.csv`: 低 SNR 下 GT bin 的复数值、rank、phase、energy。
- `data/low_snr_gt_bin/*/*_low_snr_gt_bin_summary.csv`: packet/SNR 粒度统计，含 `gt_top8/16/32_recall`。
- `data/low_snr_gt_bin/*/wrong_bin_control/*_summary.csv`: GT 与 wrong-bin 对照统计。
- `data/two_stage_weak_decoder/*_summary.json`: two-stage 解码和 Top-K recall 结果。

已有重要结果：

- Center FFT energy baseline 在 `0_0_0_10_14_16`、`-20 dB` 上：
  - Top-1 recall: `0.642`
  - Top-4 recall: `0.777`
  - Top-16 recall: `0.888`
  - Top-32 recall: `0.927`
- 直接的 header phase line / preamble profile 加权在已有 grid 中没有超过 energy Top-L；相位权重稍大时 recall 下降。
- `multi-offset` 非相干 FFT evidence 是当前已验证的强 baseline；在 `-20 dB` 上把 GT-bin Top-32 recall 推到 `1.000`，并让 two-stage decoder 在 11 个包上达到 `two_stage_symbol_ser=0`。
- 在 `len8/len32 -23 dB` 上，multi-offset 仍有改进但没有完全解决：Top-K recall 分别约 `0.909` 和 `0.865`，two-stage 仍大量 fallback。

## 3. 第一阶段候选筛选指标设计

### 3.1 设计原则

现有实验证明：简单 `energy + phase_weight * cos(phase residual)` 容易伤害 recall。原因很可能是 header-to-payload phase intercept/slope 偏差、低幅度 bin 相位噪声、以及随机噪声 bin 的相位偶然匹配。因此第一阶段应采用保守的 phase-aware pruning：

```text
energy/multi-offset evidence 负责主排序
phase structure 只负责在中等能量候选中做轻量 rescue/bonus
不让 phase-only 随机匹配的低能量 bin 抢占 Top-L
```

### 3.2 推荐指标：Phase-Gated Multi-Offset Evidence

对第 `k` 个 payload symbol、bin `b`：

1. 计算基础能量证据：

```text
E[k,b] = sum_offset |FFT_offset[k,b]|^2 / max_b |FFT_offset[k,b]|^2
```

如果不启用 multi-offset，则退化为 center FFT power。

2. 用 header 和 early high-confidence payload symbols 拟合 packet-local phase trend：

```text
phi_pred[k] = slope * abs_symbol_index[k] + intercept + packet_residual_offset
```

其中 `packet_residual_offset` 建议用 early high-margin payload argmax 的 phase residual 中位数估计，避免绝对 header line 偏置。

3. 只对能量预筛后的 bins 计算轻量 phase bonus：

```text
M = max(4L, 32) 或 energy rank <= M
r[k,b] = wrap(angle(Z_center[k,b]) - phi_pred[k])
phase_bonus[k,b] = max(0, cos(r[k,b]) - cos(tau)) / (1 - cos(tau))
```

`tau` 可从 early anchor residual std 自适应，默认可先取 `pi/3` 或 `pi/2`。这是单符号 O(M) 向量运算，不做跨符号路径搜索。

4. 最终分数：

```text
energy_score = log1p(E[k,b] / noise_floor[k])
q_line       = clamp01(function(line_r2, residual_std, anchor_count))
amp_gate     = sigmoid((rel_db[k,b] - energy_floor_db) / temp_db)

S[k,b] = energy_score
       + lambda * q_line * amp_gate * phase_bonus[k,b]
```

保守设置：

- `lambda` 从 `0.02, 0.05, 0.10, 0.20` sweep。
- `phase_bonus` 只加分，不做负惩罚。
- phase bonus 只作用于 energy top-M；其他 bins 保持低分，避免随机 phase noise 进入 Top-L。
- 如果 `q_line` 低于阈值，自动退化为 energy/multi-offset baseline。

### 3.3 需要对比的候选集

必须同时输出这些 baseline 和新指标：

- `center_energy_topL`: center FFT power Top-L。
- `multi_offset_energy_topL`: 当前最强已知 baseline。
- `phase_line_direct_topL`: 现有 direct phase metric，用来证明新指标不是重复旧失败路径。
- `phase_gated_multioffset_topL`: 新指标。
- 可选 `phase_rescue_union_topL`: energy top `L-r` + phase-gated top `r` 的候选集合，用于看 rescue lane 是否更稳。

## 4. 需要新增或修改的脚本清单

优先新增，少改旧逻辑：

1. 新增 `weak_decoder/candidate_pruning.py`
   - 放纯函数：multi-offset evidence、phase trend anchor fit、phase-gated score、rank/Top-L。
   - 不接 payload codec，不做 beam search。

2. 新增 `scripts/experiments/evaluate_candidate_pruning_metric.py`
   - 输入：低 SNR IQ + clean header-first symbol CSV。
   - 输出：per-symbol CSV + summary JSON。
   - 对同一批 symbol 同时评估 center energy、multi-offset energy、新 phase-gated metric。

3. 小改 `scripts/experiments/evaluate_phase_bin_metric.py`
   - 可选：复用新模块，保留旧参数兼容。
   - 或保持不动，把它作为旧 direct phase baseline。

4. 小改 `scripts/experiments/run_two_stage_weak_decoder.py`
   - 暂不接入新 metric；第一阶段验证通过后，再增加 `--fft-evidence-mode phase-gated`。
   - 当前只需复用其中 `load_packets()`、`extract_multi_offset_fft_evidence()` 的逻辑。

5. 新增 `scripts/experiments/plot_candidate_pruning_recall.py`
   - 读取 summary JSON/CSV，画 Recall@L vs L、Recall gain vs baseline、per-packet heatmap。

6. 文档更新
   - 本文档作为当前阶段入口。
   - 若后续实现第一阶段脚本，可在本文档追加最小复现实验命令。

## 5. 后续评估指标

第一阶段只评估 candidate pruning，不评估 payload 解码成功率。核心指标：

```text
Recall@L = mean( gt_raw_fft_bin in Top-L candidates )
```

建议 L：

```text
L = 1, 2, 4, 8, 16, 32, 64
```

必须报告：

- `argmax_accuracy`: 等价于 center energy Recall@1。
- `center_energy_recall@L`
- `multi_offset_energy_recall@L`
- `phase_gated_recall@L`
- `gain_vs_center@L`
- `gain_vs_multi_offset@L`
- `mean_gt_rank / median_gt_rank / p95_gt_rank`
- `per_packet_recall@L`
- `line_quality`: anchor_count、line_r2、residual_std、q_line。
- `phase_rescue_count`: GT 不在 energy Top-L 但进入 phase-gated Top-L 的次数。
- `phase_damage_count`: GT 在 energy Top-L 但被 phase-gated 挤出 Top-L 的次数。
- `net_rescue = phase_rescue_count - phase_damage_count`

成功标准建议：

- 对 center FFT energy Top-L：`phase_gated_recall@L` 在 `L=8/16/32` 有稳定正增益。
- 对 multi-offset energy Top-L：至少在困难集（如 `len8/len32 -23 dB` 或更低 SNR）有正增益，或在不降低 recall 的情况下显著降低候选 L。
- `damage_count` 必须很低；如果 phase 不能稳定增益，应自动退化为 energy baseline。

推荐第一轮实验矩阵：

```text
datasets:
  0_0_0_10_14_16 at -20 dB
  0_0_0_10_14_8  at -23 dB
  0_0_0_10_14_32 at -23 dB

lambda:
  0.02, 0.05, 0.10, 0.20

energy_preselect_M:
  2L, 4L, 8L, 64

phase source:
  header
  header + early high-confidence payload residual offset
```

若第一阶段 Recall@L 不超过 `multi_offset` baseline，不应继续接 two-stage decoder；应先诊断 phase residual offset、header/payload slope mismatch 和 low-amplitude phase noise。

## 6. 2026-06-15 实现状态

本轮已完成第一阶段候选筛选指标的最小可运行实现，仍限定在 `weakPacket_decoding copy/` 内，未改动主解码链路。

新增文件：

- `weak_decoder/candidate_pruning.py`
  - `score_energy()`: robust energy score。
  - `fit_early_payload_phase_trend()`: 从每包前若干个 payload symbol 的 energy-selected bin 拟合 packet-local payload phase line；不使用 GT。
  - `phase_gated_scores()`: energy score + one-sided phase bonus。
  - `phase_rescue_scores()`: 可选 exact Top-L rescue 集合，形式为 energy top `L-r` + phase-gated top `r`。
- `scripts/experiments/evaluate_candidate_pruning_metric.py`
  - 输入低 SNR IQ 和 clean header-first symbol CSV。
  - 输出 per-symbol CSV 与 summary JSON。
  - 同时报告 center energy、multi-offset energy、phase-gated metric、rescue/damage 计数。

当前推荐指标：

```text
trend = early-payload phase line, fallback = header/header-offset
base evidence = multi-offset energy
score = log1p(energy / noise_floor)
      + lambda * q_line * amp_gate * one_sided_phase_bonus
```

相位项只做轻量 per-bin 残差计算，不做跨 symbol 路径搜索。默认使用 energy preselect；若需要验证全-bin 轻量扫描，可设置：

```text
--energy-preselect-count 0 --energy-preselect-factor 0
```

最小复现实验命令：

```powershell
python scripts/experiments/evaluate_candidate_pruning_metric.py `
  -i data/low_snr_gt_bin/0_0_0_10_14_16_extreme_snr/0_0_0_10_14_16_snr_m23dB.bin `
  -s data/weak_sync_chain/header_first/0_0_0_10_14_16_header_first_symbols.csv `
  -o data/candidate_pruning/0_0_0_10_14_16_snr_m23dB_all_early.csv `
  --summary-json data/candidate_pruning/0_0_0_10_14_16_snr_m23dB_all_early_summary.json `
  --phase-bonus-weights 0.05,0.1,0.2,0.5 `
  --phase-gate-width-pi 0.25,0.333333,0.5
```

已生成输出：

```text
data/candidate_pruning/0_0_0_10_14_16_snr_m20dB_all_early.csv
data/candidate_pruning/0_0_0_10_14_16_snr_m20dB_all_early_summary.json
data/candidate_pruning/0_0_0_10_14_16_snr_m23dB_all_early.csv
data/candidate_pruning/0_0_0_10_14_16_snr_m23dB_all_early_summary.json
data/candidate_pruning/0_0_0_10_14_16_snr_m23dB_all_early_allbins_summary.json
data/candidate_pruning/0_0_0_10_14_16_snr_m23dB_all_headeroffset_summary.json
```

初步结果（`0_0_0_10_14_16`，全 11 包）：

| Dataset | L | Best phase config | Center energy | Multi-offset energy | Phase-gated | Gain vs multi |
|---|---:|---|---:|---:|---:|---:|
| -20 dB | 1 | `w=0.1, gate=0.25pi` | 0.6416 | 0.9844 | 0.9818 | -0.0026 |
| -20 dB | 8 | `w=0.1, gate=0.5pi` | 0.8390 | 1.0000 | 1.0000 | 0.0000 |
| -20 dB | 16 | `w=0.2, gate=0.25pi` | 0.8883 | 1.0000 | 1.0000 | 0.0000 |
| -20 dB | 32 | `w=0.2, gate=0.25pi` | 0.9273 | 1.0000 | 1.0000 | 0.0000 |
| -23 dB | 1 | `w=0.2, gate=0.333333pi` | 0.2675 | 0.6805 | 0.6961 | +0.0156 |
| -23 dB | 8 | `w=0.2, gate=0.5pi` | 0.5273 | 0.8571 | 0.8727 | +0.0156 |
| -23 dB | 16 | `w=0.2, gate=0.25pi` | 0.5922 | 0.9143 | 0.9117 | -0.0026 |
| -23 dB | 32 | `w=0.2, gate=0.5pi` | 0.6831 | 0.9429 | 0.9455 | +0.0026 |

结论：

- 相比传统 center FFT energy Top-L，`multi-offset + phase-gated` 在困难 SNR 下提升很大。
- 相比当前最强的 `multi-offset energy` baseline，phase-aware 指标在 -23 dB 的 Top-1/Top-8/Top-32 有小幅正增益，但 Top-16 仍可能轻微 damage。
- `early-payload` trend 明显优于 `header-offset` trend；header-only slope/intercept 仍不足以稳定预测 payload。
- 暂不建议直接接入 two-stage decoder 主链。下一步应优先做 per-packet quality gate 和自适应 lambda/gate，确保 phase 只在高质量趋势时介入，否则退化为 multi-offset energy。
