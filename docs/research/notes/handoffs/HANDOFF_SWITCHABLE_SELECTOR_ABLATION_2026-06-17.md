# Handoff: Switchable Selector Ablation

日期：2026-06-17

## 1. 当前目标

当前目标不是重写或简化主解码链，而是在现有 symbol-level selector / threshold sweep 里加入可切换的模块开关，用同一套评估流程回答一个问题：

```text
不做 high-confidence / low-confidence 两阶段区分，
不使用 packet-local phase line，
直接用 multi-offset energy + offset coherence 对每个 payload symbol 独立选 argmax，
效果是否比当前默认 A5 更好？
```

要求：

- 保留当前默认 A5 行为。
- 不删除现有 two-stage selector。
- 不引入 payload byte template、counter prior、cross-packet joint prior 或 CRC-guided search。
- 只新增可选 selector mode / config flag，跑消融看结果。

## 2. 已经做过的关键修改

本窗口已完成旧 session-prior 线清理和文档整理：

- 删除 learn-session、payload template、joint residual、旧 Phase-MAP 相关脚本。
- 删除 `data/phase_guided/` 旧中间结果目录。
- 清理旧脚本残留 `.pyc` 缓存。
- 从 `weak_decoder/phase_guided_demod.py` 移除：
  - `PayloadPrior*`
  - expected payload/header symbol API
  - payload template apply
  - prior candidate selection
  - session header prior path
- 保留 `phase_guided_demod.py` 本体，因为当前实验仍复用其中的 `PhaseLine`、phase fit、header anchor、candidate scoring 等基础工具。
- 将 two-stage 中旧的 `_score_payload_symbol_prior_candidate` 改名为 `score_projected_payload_symbols`，语义改成“对当前 packet decoder beam 生成的 projected symbols 做 PHY evidence 评分”。
- 更新 `README.md` 和 `data/README.md`，明确当前主线是单包 PHY evidence，不使用 payload byte prior、template、counter、cross-packet joint prior。
- 将实验输出字段 `uses_payload_template=False` 改成 `uses_packet_structure_prior=False`。

## 3. 涉及文件

已改动的主要文件：

```text
README.md
data/README.md
weak_decoder/phase_guided_demod.py
weak_decoder/two_stage_weak_decoder.py
weak_decoder/payload_codec.py
scripts/experiments/run_symbol_phase_two_stage.py
scripts/experiments/run_two_stage_weak_decoder.py
```

当前新增 handoff：

```text
notes/handoffs/HANDOFF_SWITCHABLE_SELECTOR_ABLATION_2026-06-17.md
```

确认已不存在的旧入口：

```text
scripts/evaluate_session_priors.py
scripts/joint_decode_residual_candidates.py
scripts/learn_session_byte_priors_from_symbols.py
scripts/learn_session_dynamic_byte_models.py
scripts/learn_session_payload_byte_template.py
scripts/learn_session_payload_template.py
scripts/make_generalization_validation_table.py
scripts/make_joint_threshold_table.py
scripts/make_phase_ablation_table.py
scripts/reconstruct_session_payloads.py
scripts/reproduce_phase_map_paper_artifacts.py
scripts/run_phase_guided_demod.py
scripts/sweep_joint_residual_session.py
scripts/sweep_map_kappa_from_candidates.py
scripts/sweep_phase_guided_session.py
data/phase_guided/
```

## 4. 重要设计决策和被否掉的方案

### 保留当前默认 A5

当前默认 A5 仍然是主基线：

```text
multi-offset energy
Top-24 candidate set
high-confidence lock
low-confidence rerank
score = 0.50 * energy + 0.90 * offset_coherence + 0.05 * packet_phase
CRC only for final validation
```

不要删除这条路径。

### 下一步只做开关消融

下一步不是把代码大改成 direct coherence，也不是把 two-stage 删除，而是在现有 selector / sweep 上加可选模式：

```text
current-default
direct-energy-coherence
direct-coherence-only
direct-energy-only
```

默认仍应保持 `current-default`，保证已有实验可复现。

### 被否掉或降级的方向

- payload byte template / session prior / counter prior / cross-packet joint decoding：已清理，不再作为主线。
- packet-line phase only：消融表现差，不作为主力。
- smooth trajectory beam：quick probe 表现不划算，不替代默认。
- 旧 override gating 公式：可作为历史消融，但不是当前默认 A5。

## 5. 当前 git diff / git status 重点

顶层 git repo 是：

```text
D:/Desktop/proj
```

当前 `git status --short` 只显示两个无关删除：

```text
D  _gen_run.py
D  test_creation.py
```

这两个状态与本任务无关，不要顺手处理。

重点：`gr-lora_sdr/weakPacket_decoding copy` 目录当前没有被顶层 git 跟踪。

验证命令：

```powershell
git ls-files -- "gr-lora_sdr/weakPacket_decoding copy/*"
```

输出为空。

因此：

- 本窗口改动存在于本地文件系统。
- `git diff --stat` 对这些文件不会显示内容。
- 接手者不要误以为修改不存在，只是这个 copy 目录不在当前 git tracking 内。

## 6. 已运行的测试命令和结果

已运行：

```powershell
python -m py_compile weak_decoder\phase_guided_demod.py weak_decoder\two_stage_weak_decoder.py weak_decoder\payload_codec.py scripts\experiments\run_symbol_phase_two_stage.py scripts\experiments\run_two_stage_weak_decoder.py scripts\experiments\analyze_phase_opportunity_space.py scripts\experiments\make_offset_coherence_ablation_table.py scripts\experiments\run_symbol_phase_threshold_sweep.py
```

结果：通过。

已运行：

```powershell
python -m py_compile scripts\detect_weak_preamble.py scripts\plot_payload_peak_trends.py scripts\run_blind_payload_search.py scripts\run_header_first_demod.py scripts\run_weak_sync_chain.py scripts\verify_payload_codec_alignment.py
```

结果：通过。

已运行：

```powershell
python -m compileall -q scripts weak_decoder
```

结果：

```text
compileall ok
```

已运行旧 API 扫描：

```powershell
rg -n "PayloadPrior|payload_prior|payload_template|pre_template|expected_payload_symbol_candidates|expected_payload_candidate_prior_scores|expected_payload_symbols|expected_header_symbols|candidate_search|_score_payload_symbol_prior_candidate|session_header_prior|apply_payload_symbol_template|select_payload_symbol_prior_candidate" weak_decoder scripts -g "*.py"
```

结果：无匹配。

## 7. 还没解决的问题

还没有实现 switchable selector modes。

还没有跑 direct coherence 相关消融。

所以目前还不知道：

```text
direct energy + offset coherence argmax 是否真的比当前 A5 更好。
```

这正是下一步要通过开关消融回答的问题。

## 8. 下一步建议

### 8.1 最小代码改动原则

只做小范围开关式改动：

- 不删除 `select_symbol_bins_two_stage`。
- 不改变默认参数结果。
- 新增 selector mode 或 config flags。
- 复用现有 `E_k`、`H_k`、Top-L、hard decode / CRC 评估。
- CRC 仍只做最终验证，不参与搜索。

### 8.2 建议新增参数

建议在 `scripts/experiments/run_symbol_phase_two_stage.py` 和 `scripts/experiments/run_symbol_phase_threshold_sweep.py` 暴露：

```text
--selector-mode
  current-default
  direct-energy-coherence
  direct-coherence-only
  direct-energy-only

--selector-top-l
--selector-energy-weight
--selector-coherence-weight
--selector-phase-weight
```

也可以内部映射到 `SymbolPhaseConfig`，保持旧参数兼容。

### 8.3 推荐模式定义

```text
current-default:
  lock = on
  phase_line = on
  candidate_scope = low_confidence_only
  score = 0.50 * A + 0.90 * H + 0.05 * q
  top_l = 24

direct-energy-coherence:
  lock = off
  phase_line = off
  candidate_scope = all_symbols
  score = 0.50 * A + 0.90 * H
  top_l = 24

direct-coherence-only:
  lock = off
  phase_line = off
  candidate_scope = all_symbols
  score = H
  top_l = 24

direct-energy-only:
  lock = off
  phase_line = off
  candidate_scope = all_symbols
  score = A
  top_l = 24
```

### 8.4 数学形式

Multi-offset fused energy：

```math
E_k[b]=\sum_r \frac{|Z_{k,r}[b]|^2}{\max_{b'}|Z_{k,r}[b']|^2+\epsilon}
```

Energy score：

```math
A_k(b)=\frac{E_k[b]}{\max_{b'}E_k[b']+\epsilon}
```

Offset coherence：

```math
H_k(b)=
\frac{|\sum_r Z_{k,r}[b]|}
{\sum_r |Z_{k,r}[b]|+\epsilon}
```

Direct mode：

```math
C_k=\operatorname{TopL}(E_k)
```

```math
S_k(b)=w_A A_k(b)+w_H H_k(b)
```

```math
\hat b_k=\arg\max_{b\in C_k}S_k(b)
```

### 8.5 推荐消融矩阵

先跑 quick probe：

```text
D1: direct energy + coherence, Top-24, 0.50/0.90
D2: direct energy + coherence, Top-16, 0.50/0.90
D3: direct energy + coherence, Top-32, 0.50/0.90
D4: direct coherence only, Top-24
D5: direct energy only, Top-24
D6: direct energy + coherence, Top-24, 0.75/0.75
D7: direct energy + coherence, Top-24, 1.00/0.50
D8: direct energy + coherence, Top-24, 0.35/1.00
```

建议 quick probe 范围：

```powershell
python scripts\experiments\run_symbol_phase_threshold_sweep.py `
  --snr-start -20 --snr-stop -23 --snr-step -1 `
  --selector-mode direct-energy-coherence `
  --selector-top-l 24 `
  --selector-energy-weight 0.50 `
  --selector-coherence-weight 0.90 `
  --output-dir data\probe_direct_energy_coherence_top24
```

如果 quick probe 接近或超过 A5，再跑完整 sweep：

```powershell
python scripts\experiments\run_symbol_phase_threshold_sweep.py `
  --snr-start -12 --snr-stop -26 --snr-step -1 `
  --selector-mode direct-energy-coherence `
  --selector-top-l 24 `
  --selector-energy-weight 0.50 `
  --selector-coherence-weight 0.90 `
  --output-dir data\ablation_direct_energy_coherence_top24
```

### 8.6 重点指标

```text
SER10 threshold
CRC90 threshold
SER@-20/-21/-22/-23
repair rate
damage rate
selected SER on already-correct symbols
selected SER on repairable symbols
```

### 8.7 判断口径

```text
如果 direct-energy-coherence >= A5:
  说明当前收益主要来自 offset coherence，lock / phase-line 对正式指标不是必要条件。
  但仍不要直接删 A5，先保留为对照。

如果 direct-energy-coherence 略低于 A5:
  说明 offset coherence 是主力，但 high-confidence lock 仍有保护价值。

如果 direct-coherence-only 差、direct-energy-coherence 好:
  说明 coherence 必须被 energy gate / energy score 约束。

如果 direct-energy-only 等于 multi-offset:
  说明 direct 框架本身没有额外收益，收益确实来自 offset coherence。
```

## 9. 一句话交接

请在现有 symbol-level selector / threshold sweep 里加入开关式 selector modes，保留当前 A5 默认不变，新增 direct energy + offset coherence argmax 等消融模式，跑同一套 SNR sweep，判断不使用 two-stage lock / phase-line 时 direct coherence rerank 是否能超过当前默认。
