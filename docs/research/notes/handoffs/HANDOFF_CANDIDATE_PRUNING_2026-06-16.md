# Handoff - Phase-aware candidate pruning recall validation

Date: 2026-06-16

Working directory:

```text
d:\Desktop\proj\gr-lora_sdr\weakPacket_decoding copy
```

## 1. 当前目标

当前任务是 LoRa 弱包解码增强的第一阶段：不要直接用 payload FFT argmax 选唯一 bin，而是在每个 payload symbol 上从 `2^SF` 个 FFT bins 中筛出 Top-L 候选，使 clean GT bin 的 `Recall@L` 高于传统 center FFT argmax / energy Top-L baseline。

核心研究假设：

```text
同一个 packet 内，payload GT-bin 的复数 FFT 相位在去除 offsets 后，随 symbol index 呈明显连续/线性结构。
```

当前实现目标不是完整 payload 解码，不接 dewhitening/CRC，不接 two-stage decoder 主链；只评估 candidate pruning 的 GT-bin Recall@L。

下一步目标：

```text
验证当前 Recall@L 设计是否具有普遍性。
用更多 USRP IQ 数据、多 preamble length、多 SNR、多 packet 组合跑实验，判断 phase-aware bonus 是否稳定优于 argmax/top-L energy baseline。
```

## 2. 已经做过的关键修改

### 文档整理

新增/更新：

- `notes/plans/PHASE_AWARE_CANDIDATE_PRUNING_PLAN.md`
  - 整理当前目录结构、关键脚本、已有实验流程、输入输出、第一阶段设计方案、评估指标。
  - 追加 2026-06-15 实现状态和初步 Recall@L 结果。
- `README.md`
  - 顶部加入当前研究入口：phase-aware candidate pruning。

### 新增候选筛选模块

新增：

```text
weak_decoder/candidate_pruning.py
```

关键函数：

- `score_energy(power, config)`
  - robust energy score:
    ```text
    energy_score[b] = log1p(power[b] / percentile_floor)
    ```
- `fit_early_payload_phase_trend(...)`
  - 从当前低 SNR packet 的前若干个 payload symbol 中，选 energy argmax bin 作为 early anchors。
  - 读取这些 anchor bin 的 center FFT phase。
  - 拟合 packet-local payload phase line:
    ```text
    phi_pred(k) = slope * abs_symbol_index + intercept
    ```
  - 不使用 GT bin。
- `phase_gated_scores(...)`
  - 计算最终候选分数：
    ```text
    S[b] = energy_score[b]
         + lambda * q_line * amp_gate[b] * phase_bonus[b]
    ```
  - 其中：
    ```text
    residual[b] = wrap(angle(center_fft[b]) - phi_pred(k))
    phase_bonus[b] = max(0, cos(residual[b]) - cos(gate)) / (1 - cos(gate))
    ```
  - phase bonus 只加分，不惩罚。
  - 默认只对 energy preselect bins 计算 phase bonus；也支持全-bin 轻量扫描：
    ```text
    --energy-preselect-count 0 --energy-preselect-factor 0
    ```
- `phase_rescue_scores(...)`
  - 可选 exact Top-L rescue 集合：
    ```text
    energy top (L-r) + phase-gated top r
    ```

### 新增评估脚本

新增：

```text
scripts/experiments/evaluate_candidate_pruning_metric.py
```

功能：

- 输入低 SNR IQ 和 clean header-first symbol CSV。
- 对每个 payload symbol 同时计算：
  - `center_rank`: center FFT energy rank，也就是 argmax / Top-L baseline。
  - `multi_offset_rank`: multi-offset normalized peak-consensus rank。
  - `phase_gated_rank`: 当前 phase-aware score rank。
- 输出 per-symbol CSV 和 summary JSON。
- summary JSON 报告：
  - `center_recall@L`
  - `multi_offset_recall@L`
  - `phase_gated_recall@L`
  - `gain_vs_center@L`
  - `gain_vs_multi_offset@L`
  - `phase_rescue_count@L`
  - `phase_damage_count@L`

### 已生成实验输出

主要输出目录：

```text
data/candidate_pruning/
```

关键结果文件：

```text
0_0_0_10_14_16_snr_m20dB_all_early.csv
0_0_0_10_14_16_snr_m20dB_all_early_summary.json
0_0_0_10_14_16_snr_m23dB_all_early.csv
0_0_0_10_14_16_snr_m23dB_all_early_summary.json
0_0_0_10_14_16_snr_m23dB_all_early_allbins_summary.json
0_0_0_10_14_16_snr_m23dB_all_headeroffset_summary.json
```

## 3. 涉及文件

本轮核心新增/修改：

```text
weakPacket_decoding copy/README.md
weakPacket_decoding copy/notes/plans/PHASE_AWARE_CANDIDATE_PRUNING_PLAN.md
weakPacket_decoding copy/notes/handoffs/HANDOFF_CANDIDATE_PRUNING_2026-06-16.md
weakPacket_decoding copy/weak_decoder/candidate_pruning.py
weakPacket_decoding copy/scripts/experiments/evaluate_candidate_pruning_metric.py
weakPacket_decoding copy/data/candidate_pruning/
```

当前评估脚本复用：

```text
scripts/experiments/run_two_stage_weak_decoder.py
```

其中复用：

- `load_packets()`
- `extract_fft()`
- `extract_multi_offset_fft_evidence()`

注意：`weak_decoder/phase_guided_demod.py` 当前也有 git diff，但内容只是 dataclass mutable default 修正：

```text
PhaseSegments:
  preamble_line = field(default_factory=PhaseLine)
  header_line = field(default_factory=PhaseLine)
  payload_own_line = field(default_factory=PhaseLine)
```

这个改动不是 candidate pruning 设计核心。

## 4. 重要设计决策和被否掉的方案

### 设计决策 1：第一阶段只做 Recall@L，不做完整解码

理由：

- 先验证候选筛选是否真的把 GT bin 拉进 Top-L。
- 不让 payload codec、CRC、beam search 等因素混淆指标。

### 设计决策 2：以 energy 为主，phase 只做保守 bonus

最终分数：

```text
S[b] = energy_score[b]
     + lambda * q_line * amp_gate[b] * phase_bonus[b]
```

原因：

- 直接 phase-only 容易被低能噪声 bin 的随机相位骗到。
- 低 SNR 时必须防止低能 bin 靠偶然相位匹配冲进 Top-L。

### 设计决策 3：phase bonus 只加分，不扣分

被 phase 线预测不匹配的 bin 不受惩罚，只是不给额外分。

原因：

- 当前 phase trend 仍有 slope/intercept 偏差风险。
- 扣分容易把原本 energy Top-L 内的 GT bin 挤出去，造成 `phase_damage`。

### 设计决策 4：优先使用 early-payload phase trend

当前默认：

```text
--phase-trend-source early-payload
```

原因：

- 已观察到 header/preamble 到 payload 存在 slope mismatch。
- `header-offset` 只能修截距，不能稳定修 slope。
- early payload argmax anchors 虽然不完美，但对 packet-local payload phase 结构更贴近。

### 设计决策 5：multi-offset 当前不是严格 energy 模型

当前 `extract_multi_offset_fft_evidence()` 中的融合：

```text
F[b] = sum_offset power_offset[b] / max_b(power_offset[b])
```

它更准确应叫：

```text
normalized multi-offset peak-consensus score
```

不是严格的接收能量估计。它本质上给每个 decimation offset 一票，观察某个 bin 是否在多个 offset 下都像峰。

需要注意：

- 好 offset 和坏 offset 权重一样。
- 完全噪声 offset 的最大峰也会被归一化成 1。
- 后续应做 weighted offset / best offset / quality-gated offset 对比。

### 被否掉或暂缓的方案

- 直接 `energy + large phase_weight * cos(residual)`
  - phase 权重大时 recall 会下降，damage 增加。
- phase-only bin selection
  - 太容易被随机相位噪声欺骗。
- full-rate oversampled dechirp + FFT
  - LoRa dechirp 前若不按 chip-rate 采样，symbol 内 segment/wrap 会导致频率不集中。
  - 当前 multi-offset 不是 full-rate FFT，而是对每个 decimation phase 先降到 chip-rate 再 FFT。
- 立刻接入 two-stage decoder 主链
  - 当前 phase 对 multi-offset baseline 只小幅增益，还需验证普遍性和稳定性。

## 5. 当前 git diff / git status 重点

在 `d:\Desktop\proj\gr-lora_sdr` 下：

```text
 M "weakPacket_decoding copy/README.md"
 M "weakPacket_decoding copy/weak_decoder/phase_guided_demod.py"
 D "weakPacket_decoding/doc/弱包解码方案粗设计.md"
?? "weakPacket_decoding copy/notes/plans/PHASE_AWARE_CANDIDATE_PRUNING_PLAN.md"
?? "weakPacket_decoding copy/notes/plans/TWO_STAGE_WEAK_DECODER.md"
?? "weakPacket_decoding copy/data/candidate_pruning/"
?? "weakPacket_decoding copy/data/two_stage_weak_decoder/"
?? "weakPacket_decoding copy/scripts/experiments/evaluate_candidate_pruning_metric.py"
?? "weakPacket_decoding copy/weak_decoder/candidate_pruning.py"
...
```

重点说明：

- 本轮核心是 `candidate_pruning.py`、`evaluate_candidate_pruning_metric.py`、`PHASE_AWARE_CANDIDATE_PRUNING_PLAN.md`、`README.md`、`data/candidate_pruning/`。
- 工作区里还有很多之前已经存在的未跟踪文件，例如 two-stage/blind decoder 相关脚本和数据；不要误认为全部是本轮新改。
- `weakPacket_decoding/doc/弱包解码方案粗设计.md` 删除发生在非 copy 目录，需谨慎确认是否是用户/历史操作，不要随意恢复或提交。

## 6. 已运行的测试命令和结果

### 语法检查

```powershell
cd "d:\Desktop\proj\gr-lora_sdr\weakPacket_decoding copy"
python -m py_compile weak_decoder/candidate_pruning.py scripts/experiments/evaluate_candidate_pruning_metric.py
```

结果：通过，无输出。

### -20 dB 全包评估

```powershell
python scripts/experiments/evaluate_candidate_pruning_metric.py `
  -i data/low_snr_gt_bin/0_0_0_10_14_16/0_0_0_10_14_16_snr_m20dB.bin `
  -s data/weak_sync_chain/header_first/0_0_0_10_14_16_header_first_symbols.csv `
  -o data/candidate_pruning/0_0_0_10_14_16_snr_m20dB_all_early.csv `
  --summary-json data/candidate_pruning/0_0_0_10_14_16_snr_m20dB_all_early_summary.json `
  --quiet `
  --phase-bonus-weights 0.05,0.1,0.2 `
  --phase-gate-width-pi 0.25,0.333333,0.5
```

关键结果：

| L | Center energy | Multi-offset | Phase-gated | Gain vs center | Gain vs multi |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.6416 | 0.9844 | 0.9818 | +0.3403 | -0.0026 |
| 8 | 0.8390 | 1.0000 | 1.0000 | +0.1610 | 0.0000 |
| 16 | 0.8883 | 1.0000 | 1.0000 | +0.1117 | 0.0000 |
| 32 | 0.9273 | 1.0000 | 1.0000 | +0.0727 | 0.0000 |

解释：-20 dB 下 multi-offset 已基本饱和，phase-gated 主要持平。

### -23 dB 全包评估

```powershell
python scripts/experiments/evaluate_candidate_pruning_metric.py `
  -i data/low_snr_gt_bin/0_0_0_10_14_16_extreme_snr/0_0_0_10_14_16_snr_m23dB.bin `
  -s data/weak_sync_chain/header_first/0_0_0_10_14_16_header_first_symbols.csv `
  -o data/candidate_pruning/0_0_0_10_14_16_snr_m23dB_all_early.csv `
  --summary-json data/candidate_pruning/0_0_0_10_14_16_snr_m23dB_all_early_summary.json `
  --quiet `
  --phase-bonus-weights 0.05,0.1,0.2,0.5 `
  --phase-gate-width-pi 0.25,0.333333,0.5
```

关键结果：

| L | Center energy | Multi-offset | Phase-gated | Gain vs center | Gain vs multi | Best config |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.2675 | 0.6805 | 0.6961 | +0.4286 | +0.0156 | `w=0.2, gate=0.333333pi` |
| 8 | 0.5273 | 0.8571 | 0.8727 | +0.3455 | +0.0156 | `w=0.2, gate=0.5pi` |
| 16 | 0.5922 | 0.9143 | 0.9117 | +0.3195 | -0.0026 | `w=0.2, gate=0.25pi` |
| 32 | 0.6831 | 0.9429 | 0.9455 | +0.2623 | +0.0026 | `w=0.2, gate=0.5pi` |

解释：相比 center argmax/Top-L 提升很大；相比当前更强的 multi-offset baseline，phase 只有小幅正增益，且 Top-16 有轻微 damage。

### 全-bin phase bonus 对照

```powershell
python scripts/experiments/evaluate_candidate_pruning_metric.py `
  -i data/low_snr_gt_bin/0_0_0_10_14_16_extreme_snr/0_0_0_10_14_16_snr_m23dB.bin `
  -s data/weak_sync_chain/header_first/0_0_0_10_14_16_header_first_symbols.csv `
  -o data/candidate_pruning/0_0_0_10_14_16_snr_m23dB_all_early_allbins.csv `
  --summary-json data/candidate_pruning/0_0_0_10_14_16_snr_m23dB_all_early_allbins_summary.json `
  --quiet `
  --energy-preselect-count 0 `
  --energy-preselect-factor 0 `
  --phase-bonus-weights 0.02,0.05,0.1,0.2,0.5 `
  --phase-gate-width-pi 0.166667,0.25,0.333333,0.5
```

结果：与 energy preselect 模式接近，没有明显额外收益。说明当前限制主要不是预筛漏掉 GT，而是 phase trend 质量和 phase/energy 权重 tradeoff。

### header-offset 对照

```powershell
python scripts/experiments/evaluate_candidate_pruning_metric.py `
  -i data/low_snr_gt_bin/0_0_0_10_14_16_extreme_snr/0_0_0_10_14_16_snr_m23dB.bin `
  -s data/weak_sync_chain/header_first/0_0_0_10_14_16_header_first_symbols.csv `
  -o data/candidate_pruning/0_0_0_10_14_16_snr_m23dB_all_headeroffset.csv `
  --summary-json data/candidate_pruning/0_0_0_10_14_16_snr_m23dB_all_headeroffset_summary.json `
  --quiet `
  --phase-trend-source header-offset `
  --phase-bonus-weights 0.05,0.1,0.2,0.5 `
  --phase-gate-width-pi 0.25,0.333333,0.5
```

结果：`early-payload` trend 明显优于 `header-offset` trend。

## 7. 还没解决的问题

1. 普遍性未知
   - 当前主要只在 `0_0_0_10_14_16` 的 -20/-23 dB 上跑过。
   - 需要用更多 USRP IQ 数据、多 preamble length、多 SNR 验证。

2. multi-offset 融合公式不够严谨
   - 当前是 normalized peak-consensus：
     ```text
     F[b] = sum_offset power_offset[b] / max_b(power_offset[b])
     ```
   - 它实验有效，但不是严格 energy fusion。
   - 坏 offset 也会贡献一个归一化噪声峰。

3. phase 对 multi-offset 的额外增益还小
   - 相比 center argmax 提升大。
   - 相比 multi-offset baseline，Top-1/Top-8 有小幅正增益，但仍有 damage。

4. phase trend 质量门控还粗糙
   - 当前 `q_line` 只是 anchor_count、R2、residual std 的保守组合。
   - 还没有 per-packet 自适应判断：什么时候开启 phase，什么时候退化回 pure multi-offset。

5. early-payload anchors 有自举风险
   - early anchors 来自 energy argmax，不用 GT，但低 SNR 下可能选错。
   - 需要分析 early anchor 错误率和 phase line 质量的关系。

6. 尚未接入实际 payload 解码链
   - 目前只验证 candidate pruning recall。
   - 即使 Recall@L 提升，也还要验证是否能转化为 two-stage/beam/codec 解码收益。

## 8. 下一步建议

### 8.1 验证 Recall 设计是否普遍

优先跑更多已有 USRP IQ 数据：

```text
0_0_0_10_14_8
0_0_0_10_14_16
0_0_0_10_14_32
```

建议 SNR：

```text
-20 dB
-23 dB
-25 dB
-27 dB
```

优先矩阵：

```text
phase-trend-source:
  early-payload
  header-offset

phase-bonus-weights:
  0.02, 0.05, 0.1, 0.2, 0.5

phase-gate-width-pi:
  0.166667, 0.25, 0.333333, 0.5

energy-preselect:
  default
  all-bin phase bonus: --energy-preselect-count 0 --energy-preselect-factor 0
```

需要报告：

```text
center_recall@L
multi_offset_recall@L
phase_gated_recall@L
gain_vs_center@L
gain_vs_multi_offset@L
phase_rescue_count@L
phase_damage_count@L
per_packet recall
trend_source / trend_r2 / residual_std / early_anchor_count
```

### 8.2 做一个批量 runner

建议新增：

```text
scripts/experiments/run_candidate_pruning_sweep.py
```

输入配置：

```text
dataset stem
SNR IQ path
symbol CSV path
preamble length
phase param grid
```

输出：

```text
data/candidate_pruning/sweeps/<timestamp>/*_summary.json
data/candidate_pruning/sweeps/<timestamp>/merged_summary.csv
```

这样下一轮不需要手动敲多条命令。

### 8.3 修正或对照 multi-offset baseline

建议新增三个 baseline：

1. `center`
   - 当前 center FFT argmax。
2. `best-offset`
   - 每个 offset 单独 FFT，按 peak margin / entropy 选一个最佳 offset。
3. `weighted-offset`
   - 用 offset 质量加权：
     ```text
     F[b] = sum_offset w_offset * normalized_power_offset[b]
     ```

可选 offset quality：

```text
peak_margin_db = 10log10(top1/top2)
entropy_score = 1 - spectral_entropy
peak_to_median = top1 / median_power
```

### 8.4 增加 phase quality gate

建议规则：

```text
if trend_quality < threshold:
    use pure multi-offset energy
else:
    use phase-gated score
```

或者 per-packet 自适应选择 `lambda/gate`：

```text
low residual_std + high R2 -> stronger phase bonus
high residual_std / low anchor_count -> lambda = 0
```

### 8.5 再考虑接入 two-stage decoder

只有当多数据集上 `phase_gated_recall@L` 稳定超过 `multi_offset_recall@L`，并且 `phase_damage_count` 可控，再接：

```text
run_two_stage_weak_decoder.py --fft-evidence-mode phase-gated
```

否则先不要把它放进主解码链，避免把不稳定 phase bonus 带入后续复杂搜索。
