# Handoff: Two-Stage Retreat to Physical-Layer OS Enhancement

日期: 2026-06-18

范围: `D:/Desktop/proj/gr-lora_sdr/weakPacket_decoding copy`

## 1. 当前目标

本窗口的目标从“继续榨干两阶段弱包解码架构”转为“把论文主线收束到物理层 oversampling 能量增强”。

当前最重要的结论:

```text
两阶段解码不建议继续作为论文主线。
它可以保留为工程诊断、消融实验、可选后处理或历史分支，但不要让它承担核心创新叙事。
```

原因非常关键，后续窗口请不要绕开这一点:

- 用户明确判断“两阶段解码无用”，至少对当前论文主线而言，它不再是值得继续主攻的方向。
- 当前课题是学术论文，需要围绕某一个清晰物理层点展开。用户现在真正想做的是“物理层能量增强”，特别是利用 OS factor 的副本能量、对齐和叠加。
- 两阶段架构的学术叙事不干净。它依赖 CRC、beam、候选搜索、payload/code constraint 和列表筛选，容易被审稿人看成 CRC-assisted 后处理技巧，而不是一个干净的物理层信号增强贡献。
- 两阶段架构在 AWGN 仿真中有 CRC/PRR 增益，但在真实 IQ 上不能稳定压过 Savaux paper baseline。把它放在主线会拖慢更有价值的方向。
- 后续论文主线建议围绕:
  1. 弱包对齐
  2. OS 副本叠加/融合
  3. 质量感知的 OS 副本选择、加权或可靠性建模

一句话交接:

```text
Savaux 论文 baseline 已经正面撞上本项目最有价值的 OS 增强方向。
下一阶段应该在物理层 OS replica fusion 上超过 Savaux，而不是继续把 CRC/beam/two-stage 搜索包装成主贡献。
```

## 2. 已经做过的关键修改

### 2.1 Savaux oversampled paper baseline

已经实现并整理为独立 baseline 文件夹:

```text
weak_decoder/baselines/savaux_oversampled/
scripts/experiments/baselines/savaux_oversampled/
```

实现边界记录在:

```text
notes/baselines/PAPER_OVERSAMPLED_DEMOD_BASELINE_2026-06-17.md
```

重要边界:

```text
只实现论文 A Low-Complexity Demodulation for Oversampled LoRa Signal 中 Eq. (34)-(37) 的 OSR branch DFT + phase-compensated sum + argmax。
没有使用本项目已有的 offset coherence、phase line、Top-L lock、CRC candidate search、payload prior 或其他本地技巧。
```

这保证 Savaux baseline 可以作为干净的论文对照组。

### 2.2 Savaux+codec / two-stage engineering branch

在 `weak_decoder/two_stage_weak_decoder.py` 上增加了大量 default-off 诊断开关和候选搜索路径，目标是测试 “Savaux OSR evidence + LoRa PHY codec/CRC constraint” 是否还能稳定提升。

新增或强化的能力包括:

- block symbol seed search
- global rank-diverse search
- global rank-cost search
- CRC candidate high-rank evidence gate
- CRC-state search backend
- 更详细的候选来源、rank、evidence margin、fix/break 统计

当前保守默认:

```text
crc_candidate_max_beam_rank = 64
crc_candidate_high_rank_max_beam_rank = 256
crc_candidate_high_rank_min_evidence_margin = 4.0
block_symbol_seed_top_m = 0
block_symbol_seed_quota = 0
global_rank_cost_max = 0.0
global_rank_cost_state_limit = 0
crc_state_search_block_top_r = 0
```

结论:

```text
这个分支有工程价值和诊断价值，但不应作为论文主线。
```

### 2.3 Structured / adaptive / timing path experiments

新增了几个物理启发但结果不够稳定的 OS path 模块:

```text
weak_decoder/structured_path_demod.py
weak_decoder/adaptive_path_demod.py
weak_decoder/timing_path_demod.py
```

对应脚本在:

```text
scripts/experiments/structured_paths/
```

尝试过的方向包括:

- fixed OS branch paths
- modular linear path `q[p] = a*p + b mod R`
- period-2 paths
- piecewise constant paths
- smooth adaptive paths
- fractional timing paths
- packet-shared timing diagnostics
- structured/adaptive/timing oracle diagnostics

结论:

```text
这些不是完全没信息，但目前没有形成稳定压过 Savaux 的主性能引擎。
它们适合写成消融、负结果或 future work，不建议继续在当前主线里加复杂度。
```

### 2.4 Real-IQ comparison runners

新增真实 IQ 对比脚本:

```text
scripts/experiments/structured_paths/run_real_iq_crc_probe.py
scripts/experiments/structured_paths/run_real_iq_batch_crc_probe.py
```

真实 IQ 数据源:

```text
D:/Desktop/proj/gr-lora_sdr/data/USRP_IQ
```

参数来自该目录下说明:

```text
samp-rate = 500000
BW = 125000
OS = 4
center-freq = 487.7e6
sync-word = 0x34
crc-mode = 0
```

真实 IQ 当前没有 byte-level 或 symbol-level GT，因此只做 header-valid packet 上的 CRC/PRR 对比，不做 SER/BER。

## 3. 涉及文件

### 3.1 主要代码文件

```text
weak_decoder/two_stage_weak_decoder.py
weak_decoder/structured_path_demod.py
weak_decoder/adaptive_path_demod.py
weak_decoder/timing_path_demod.py
weak_decoder/baselines/savaux_oversampled/paper_oversampled_demod.py
```

### 3.2 Baseline runner

```text
scripts/experiments/baselines/savaux_oversampled/run_paper_oversampled_baseline.py
scripts/experiments/baselines/savaux_oversampled/run_savaux_current_threshold_sweep.py
```

### 3.3 Structured/two-stage diagnostic scripts

```text
scripts/experiments/structured_paths/run_savaux_codec_sweep.py
scripts/experiments/structured_paths/run_structured_path_sweep.py
scripts/experiments/structured_paths/run_adaptive_path_sweep.py
scripts/experiments/structured_paths/run_real_iq_crc_probe.py
scripts/experiments/structured_paths/run_real_iq_batch_crc_probe.py
scripts/experiments/structured_paths/diagnose_codec_gt_path.py
scripts/experiments/structured_paths/diagnose_structured_ensemble_rank.py
scripts/experiments/structured_paths/diagnose_adaptive_path_oracle.py
scripts/experiments/structured_paths/diagnose_timing_path_oracle.py
```

### 3.4 关键 notes

```text
notes/baselines/PAPER_OVERSAMPLED_DEMOD_BASELINE_2026-06-17.md
notes/baselines/SAVAUX_CODEC_PRUNING_DIAGNOSTICS_2026-06-18.md
notes/baselines/REAL_IQ_LARGE_SCALE_COMPARISON_2026-06-18.md
notes/baselines/REAL_IQ_SAVAUX_CODEC_COMPARISON_2026-06-18.md
notes/handoffs/HANDOFF_TWO_STAGE_RETREAT_TO_PHY_OS_ENHANCEMENT_2026-06-18.md
```

### 3.5 关键输出目录

大量实验输出在 `data/` 下，尤其是:

```text
data/paper_oversampled_baseline/
data/savaux_codec_default_rank64_all_m22_m26/
data/savaux_codec_highrank_gate_all_m22_m26/
data/baseline_comparison/real_iq_crc_batch_sf11_all_fastscreen_sample20m_hop2_probe3/
data/baseline_comparison/real_iq_crc_batch_sf11_weak_fullcheck/
data/baseline_comparison/real_iq_crc_consolidated_summary.csv
data/baseline_comparison/real_iq_crc_consolidated_group_summary.csv
```

注意:

```text
data/ 下有大量中间实验目录，不建议全部提交。
后续提交时建议只保留 summary CSV、关键 notes 和可复现实验脚本。
```

## 4. 重要设计决策和被否掉的方案

### 4.1 Savaux baseline 必须保持干净

决策:

```text
Savaux paper baseline 只实现论文公式，不混入本项目已有设计。
```

原因:

```text
后续论文必须能公平地说: 我们的方法相对 Savaux 这个 OSR baseline 又多做了什么。
```

### 4.2 两阶段解码从主线降级

决策:

```text
two-stage weak decoder 保留为 diagnostic/ablation/optional post-processing，不作为论文主贡献。
```

这是本 handoff 最重要的学术决策。

原因:

- 它依赖 CRC、beam、candidate search、LoRa codec constraints，叙事像后处理或 list decoding。
- 它在 AWGN 上有 CRC 增益，但真实 IQ 上没有稳定超过 Savaux paper baseline。
- 它会把“OS 物理层能量增强”的主线搅浑。
- 用户明确认为当前两阶段解码对论文主线无用，应记录为路线撤退。

保留它的理由:

- 可用于诊断 Savaux evidence 的错误形态。
- 可作为 ablation: “如果引入 CRC/codec 后处理会怎样”。
- 可作为工程备选，不删代码。

### 4.3 High-rank evidence gate 是当前 two-stage 最保守可留设置

决策:

```text
rank <= 64 的 CRC candidate 正常接受。
rank 65..256 只有 evidence margin >= 4.0 才接受。
```

原因:

- 可恢复少数真实 high-rank rescue。
- 可拒绝已观察到的 rank-193 false positive。
- 三数据集 AWGN 中没有观察到 symbol break。

但它仍然是 CRC/beam 后处理技巧，因此不能解决学术叙事问题。

### 4.4 被否掉或降级的方案

以下方案已经试过，不建议继续作为主线追:

```text
phase-line 作为主 decoder
fixed structured OS paths
modular linear OS paths
piecewise constant OS paths
periodic OS paths
smooth adaptive phase-consistency paths
fractional timing path
packet-shared timing path
block symbol seed search
global rank-cost search
CRC-state search
two-stage CRC/beam candidate search as paper mainline
```

统一结论:

```text
这些方向不是完全没有信号，但目前没有稳定、干净、可讲清楚的增益。
它们可以做负结果和消融，不要继续压在论文主线。
```

## 5. 当前 git diff / git status 重点

当前 `git diff --stat` 重点:

```text
weak_decoder/two_stage_weak_decoder.py | 873 ++++++++++++++++++++-
1 file changed, 834 insertions(+), 39 deletions(-)
```

当前 `git status --short` 重点:

```text
 M weak_decoder/two_stage_weak_decoder.py
?? weak_decoder/structured_path_demod.py
?? weak_decoder/adaptive_path_demod.py
?? weak_decoder/timing_path_demod.py
?? scripts/experiments/structured_paths/
?? notes/baselines/REAL_IQ_LARGE_SCALE_COMPARISON_2026-06-18.md
?? notes/baselines/REAL_IQ_SAVAUX_CODEC_COMPARISON_2026-06-18.md
?? notes/baselines/SAVAUX_CODEC_PRUNING_DIAGNOSTICS_2026-06-18.md
?? notes/handoffs/HANDOFF_TWO_STAGE_RETREAT_TO_PHY_OS_ENHANCEMENT_2026-06-18.md
?? data/... many experimental output directories
```

注意:

- Savaux baseline 文件夹当前没有出现在 status 重点里，说明它已经在本窗口之前或前一阶段落盘并处于 clean/已跟踪状态。
- `data/` 下未跟踪目录非常多，包含大量中间实验输出。不要盲目全提交。
- PowerShell 中看到过 `LF will be replaced by CRLF` 提示，属于 Windows 换行提示。

## 6. 已运行的测试命令和结果

### 6.1 Savaux paper baseline sanity

编译检查:

```powershell
python -m py_compile `
  "weak_decoder\baselines\savaux_oversampled\paper_oversampled_demod.py" `
  "scripts\experiments\baselines\savaux_oversampled\run_paper_oversampled_baseline.py"
```

结果:

```text
passed
```

synthetic chirp check:

```text
SF=7, OSR=4, symbol ids 0/1/2/17/63/126/127 all return expected raw FFT bin.
```

clean packet check:

```text
packets=3
center_symbol_ser=0.000
paper_symbol_ser=0.000
paper_crc_valid_rate=1.000
```

低 SNR noisy IQ baseline:

```text
data/paper_oversampled_baseline/0_0_0_10_14_16_m22_m27_summary.csv
```

结果:

| SNR | packets | center SER | paper OSR SER | paper CRC |
| ---: | ---: | ---: | ---: | ---: |
| -22 | 11 | 0.603 | 0.005 | 0.818 |
| -23 | 11 | 0.732 | 0.049 | 0.273 |
| -24 | 11 | 0.834 | 0.104 | 0.000 |
| -25 | 11 | 0.912 | 0.229 | 0.000 |
| -26 | 11 | 0.951 | 0.371 | 0.000 |
| -27 | 11 | 0.969 | 0.558 | 0.000 |

解释:

```text
Savaux paper OSR baseline 已经非常强，说明 OS 副本能量确实是本项目最有价值的方向。
```

### 6.2 Savaux+codec AWGN sweep

核心输出:

```text
data/savaux_codec_default_rank64_all_m22_m26/
data/savaux_codec_highrank_gate_all_m22_m26/
```

rank64 default 三数据集 AWGN 平均 CRC:

| SNR | Savaux paper | codec rank <= 64 |
| ---: | ---: | ---: |
| -22 | 0.650 | 0.952 |
| -23 | 0.370 | 0.792 |
| -24 | 0.256 | 0.539 |
| -25 | 0.033 | 0.350 |
| -26 | 0.000 | 0.097 |

high-rank gate 三数据集 AWGN 平均 CRC:

| SNR | Savaux paper | rank64 codec | high-rank gate codec |
| ---: | ---: | ---: | ---: |
| -22 | 0.650 | 0.952 | 0.952 |
| -23 | 0.370 | 0.792 | 0.825 |
| -24 | 0.256 | 0.539 | 0.539 |
| -25 | 0.033 | 0.350 | 0.350 |
| -26 | 0.000 | 0.097 | 0.097 |

结论:

```text
AWGN 上 codec/CRC beam 能明显改善 CRC50，但 SER threshold 对 Savaux 的平均增益只有约 0.4 dB。
这不足以支撑“稳定 2-3 dB 压过 Savaux”的目标，也不能解决学术叙事不干净的问题。
```

### 6.3 Real IQ large-scale comparison

记录文件:

```text
notes/baselines/REAL_IQ_LARGE_SCALE_COMPARISON_2026-06-18.md
```

真实 IQ 跨 SF/TP fast screen:

| SF | TP | captures | header-valid packets | FFT | current | Savaux paper | Savaux+codec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 14 | 3 | 8 | 1.000 | 1.000 | 1.000 | 1.000 |
| 11 | 2 | 4 | 12 | 0.917 | 0.917 | 1.000 | 1.000 |
| 12 | 2 | 4 | 11 | 0.091 | 0.091 | 0.091 | 0.091 |
| 12 | 6 | 4 | 12 | 0.000 | 0.000 | 0.000 | 0.000 |
| 12 | 10 | 4 | 12 | 0.000 | 0.000 | 0.000 | 0.000 |
| 12 | 14 | 4 | 12 | 0.000 | 0.000 | 0.000 | 0.000 |

SF11/TP2 full fast set:

| SF | TP | captures | header-valid packets | FFT | current | Savaux paper | Savaux+codec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 2 | 17 | 45 | 0.911 | 0.911 | 0.956 | 0.956 |

weak SF11 deep check:

```text
lab1_sf11_TP2/1_0_16_11_2_16.bin
```

| method | CRC-valid packets |
| --- | ---: |
| traditional FFT | 0/4 |
| multi-offset argmax | 1/4 |
| current selected | 1/4 |
| Savaux paper | 2/4 |
| Savaux+codec | 2/4 |

结论:

```text
真实 IQ 上，Savaux paper 在 SF11/TP2 transition region 比 current selected 更强。
Savaux+codec 在真实 IQ 中基本匹配 Savaux paper，没有稳定额外收益。
SF12 组多数 payload CRC 全失败，当前更像 sync/window/channel 问题，不适合直接比较 payload demodulator。
```

### 6.4 Python compile

已运行并通过:

```powershell
python -m py_compile `
  weak_decoder\two_stage_weak_decoder.py `
  scripts\experiments\structured_paths\run_savaux_codec_sweep.py `
  scripts\experiments\structured_paths\run_real_iq_crc_probe.py `
  scripts\experiments\structured_paths\run_real_iq_batch_crc_probe.py `
  scripts\experiments\structured_paths\diagnose_codec_gt_path.py
```

结果:

```text
passed
```

还有一个 full all-capture real-IQ job 被 1 小时 timeout 停止，但已经写出了有用的 SF12/TP10 完整组结果:

```text
SF12/TP10: 17 captures, 51 header-valid packets, all methods CRC = 0.000
```

该 timeout 不代表脚本编译或运行逻辑失败，只是大规模真实 IQ 扫描太慢。

## 7. 还没解决的问题

### 7.1 真实 IQ 没有 byte-level / symbol-level GT

当前真实 IQ 只能用 CRC/PRR 评估。

未解决:

```text
没有真实 IQ 的 byte GT，因此 BER 不科学。
没有真实 IQ 的 symbol GT，因此 SER 也不能直接算。
```

这也是之前讨论过的问题:

```text
在没搞到 byte-level GT 前，真实 IQ 上做 BER 计算不太科学。
```

### 7.2 SF12 真实 IQ 失败原因未定位

SF12/TP2、TP6、TP10、TP14 很多 header-valid packet payload CRC 全失败。

可能原因:

- sync search 仍然不稳
- frame window 不准
- CFO/STO/channel drift 更严重
- packet selection 选到了不适合 payload 比较的窗口

未解决:

```text
这部分还不能说明某个 payload demodulator 更差，只能说明当前真实 IQ pipeline 没把 SF12 变成有效对比区间。
```

### 7.3 第三个论文创新点还没定型

当前比较稳的前两点:

```text
1. 对齐
2. OS 副本叠加/融合
```

第三点建议从下面挑一个继续实证:

```text
quality-aware OS replica weighting
bad-branch rejection
preamble/SFD consistency based branch reliability
alignment-aware coherent/noncoherent hybrid fusion
per-symbol reliability map for OS branches
```

不要把第三点定成 CRC/beam/two-stage search。

### 7.4 two-stage 文件变大，需要后续清理

`weak_decoder/two_stage_weak_decoder.py` 已经膨胀很多。

如果后续保留它，建议拆分:

```text
two_stage_weak_decoder.py
codec_candidate_search.py
crc_state_search.py
diagnostic_metrics.py
```

但如果它只作为诊断分支，短期可以不重构，避免继续投入。

## 8. 下一步建议

### 8.1 论文主线建议

建议新主线写成:

```text
Weak LoRa packet decoding through alignment-aware oversampled replica enhancement.
```

核心叙事:

1. 弱包失败的关键不是单个 symbol FFT，而是 OS=4 下存在多个带不同相位/时延/质量的观测副本。
2. Savaux 已经证明低复杂度 OSR branch combining 很强，是必须超越的 baseline。
3. 本项目的贡献应该是: 在弱包场景中先做更稳的 packet/symbol alignment，再对 OS replicas 做质量感知融合，而不是只做固定公式叠加。

### 8.2 立即要做的工程动作

建议下一窗口优先做:

- 冻结 two-stage 主线，不删除代码，只在 notes 和脚本命名上标成 diagnostic/ablation。
- 建一个新 note，例如:

```text
notes/design/PHY_OS_ENHANCEMENT_MAINLINE_2026-06-18.md
```

内容专门写:

```text
main claim
baseline list
metrics
ablation plan
why not two-stage
```

- 新实验不要再默认跑 `savaux_codec`，而是至少比较:

```text
traditional_fft
Savaux paper baseline
proposed physical OS enhancement
```

`current_selected` 可以作为历史工程方法保留。

### 8.3 下一步可尝试的具体算法方向

建议从最干净的物理层方向开始:

```text
对每个 OS branch 估计 reliability:
  preamble consistency
  sync/SFD consistency
  peak sharpness
  branch energy stability
  phase coherence across known chirps

然后在 payload demod 中:
  对可靠 branch 加权
  对明显坏 branch 降权或剔除
  保持不使用 CRC 决策 symbol bin
```

这比 two-stage 更适合论文:

```text
它仍然发生在 demod metric 层面，是物理层增强。
不会被审稿人轻易归类为 CRC 后处理。
也更容易和 Savaux 的固定 OSR combining 形成清晰差异。
```

### 8.4 评估建议

建议固定三类评估:

1. AWGN synthetic with symbol GT:
   - 指标: SER、CRC/PRR
   - 用来画主曲线和阈值增益
2. Real IQ transition region:
   - 优先 `SF11/TP2`
   - 指标: header-valid packet CRC/PRR
3. Real IQ hard region:
   - `SF12` 暂时作为 sync/window 诊断，不作为 payload demod 主结论

必须始终和 Savaux paper baseline 对齐比较。

## 最后提醒

这次窗口最有价值的不是 two-stage 又多加了多少搜索，而是确认了路线边界:

```text
两阶段能救一些包，但它不像论文主贡献。
OS factor 的物理层能量增强才是主线。
Savaux 是必须面对的强 baseline。
下一阶段要做的是在不依赖 CRC/beam/list search 的前提下，把 OS replica fusion 做得比 Savaux 更适合弱包。
```

