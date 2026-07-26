# Handoff: From Phase-Line Retreat to Oversampling Enhancement

日期：2026-06-17

范围：`D:/Desktop/proj/gr-lora_sdr/weakPacket_decoding copy`

## 1. 当前目标

本窗口的中心任务已经发生转移。

原来的路线是：在 symbol-level weak decoder 里继续榨取 packet-local phase line，希望用 payload 相位轨迹强化低 SNR 解码。

本窗口的实验结论是：**phase line 不是完全没信息，但不适合作为主证据继续押注**。这条路线目前要“耻辱离开”，但不是白忙：它留下了清楚的失败边界、诊断脚本和一批数值证据。

新的中心目标：

```text
从 phase-line 强化解码，转向 oversampling 下的弱信号叠加增强。
目标是 follow 并超越论文 A Low-Complexity Demodulation for Oversampled LoRa Signal。
当前项目已经与该方向有重叠：multi-offset energy、offset coherence、oversampling phase consistency。
下一阶段应把问题改写成 oversampled LoRa signal 的低复杂度增强/融合问题，而不是继续把 phase line 当主 decoder。
```

注意：本窗口没有重新阅读论文正文、公式和实验设置。下一窗口应先正式读论文，复刻其 baseline，再定义本项目要超越的指标。

## 2. 已经做过的关键修改

### 2.1 中文化 offset-coherence ablation 报告

已将：

```text
data/ablation_offset_coherence_summary/ablation_report.md
```

从英文报告改成中文说明。

主要内容不改数值，只翻译：

- 标题和 scope
- full sweep / quick probe 说明
- 表头
- interpretation

核心结论保持不变：

```text
A5 current default 是当前最佳默认配置。
offset coherence 是主要增益来源。
packet-line phase 只是小辅助项。
high-confidence lock 有保护作用。
Top-24 是当前低复杂度折中。
```

### 2.2 新增 phase line vs GT payload 对比诊断脚本

新增：

```text
scripts/experiments/compare_phase_line_to_gt_payload.py
```

用途：

```text
在低 SNR AWGN 条件下，
复用当前 A5 selector 和 header phase line，
同时强行读取 GT raw FFT bin 的 payload 相位，
把以下几条 line 放到同一张逐包表里比较：

1. selector fitted phase line
2. header phase line
3. noisy GT payload-native phase line
4. clean GT payload-native phase line
```

该脚本只做诊断，不参与解码主链，不使用 CRC 搜索，不训练模型。

输出：

```text
data/phase_line_gt_compare/m22_m27_all_datasets/phase_line_gt_compare_packets.csv
data/phase_line_gt_compare/m22_m27_all_datasets/phase_line_gt_compare_summary.csv
```

脚本已中文化：

- 顶部 docstring
- 函数说明
- CLI help
- 关键错误提示

保留英文变量名和 CSV 字段名，避免破坏后续汇总脚本。

### 2.3 给 STO phase-jump 脚本增加 no-plots 开关

修改：

```text
scripts/experiments/run_low_snr_sto_phase_jump_experiment.py
```

新增：

```text
--no-plots
```

原因：

默认 Python 环境没有 `matplotlib`，第一次运行该脚本时 CSV 已经写完，但最后画 PNG 阶段失败：

```text
ModuleNotFoundError: No module named 'matplotlib'
```

现在可以用：

```powershell
python scripts\experiments\run_low_snr_sto_phase_jump_experiment.py ... --no-plots
```

完成 CSV 实验。

备注：用户确认 `gr-lora` conda 环境有 matplotlib。本窗口已验证：

```powershell
D:\mysoft2\miniconda3\envs\gr-lora\python.exe -c "import matplotlib; print(matplotlib.__version__)"
```

结果：

```text
3.8.4
```

后续需要出图时应优先使用：

```text
D:\mysoft2\miniconda3\envs\gr-lora\python.exe
```

### 2.4 运行低 SNR GT-bin phase probe

运行范围：

```text
dataset: 0_0_0_10_14_16
SNR: -22 -23 -24 -25 -26 -27 dB
cfo_correction_mode: continuous
```

输出：

```text
data/low_snr_gt_bin/0_0_0_10_14_16_m22_m27_phase_probe/
```

该目录只保存 CSV，不保存 noisy IQ，不画图。

重点结果：

```text
SNR   argmax_correct  phase_R2  rmse_pi  residual_std_pi
-22   0.397           0.904     0.336    0.273
-23   0.268           0.862     0.360    0.279
-24   0.166           0.851     0.382    0.294
-25   0.088           0.822     0.418    0.309
-26   0.049           0.763     0.474    0.335
-27   0.031           0.692     0.499    0.357
```

解释：

```text
即使强行读取 GT bin，payload unwrap phase 在低 SNR 下仍有一定线性结构，
但 SNR 越低越散。GT bin 自己的相位轨迹也不是稳定到足以独立当 decoder 主证据。
```

### 2.5 运行 selector/header line vs GT payload line 对比

运行范围：

```text
datasets:
  0_0_0_10_14_8
  0_0_0_10_14_16
  0_0_0_10_14_32

SNR:
  -22 -23 -24 -25 -26 -27 dB
```

输出：

```text
data/phase_line_gt_compare/m22_m27_all_datasets/
```

关键汇总：

```text
SNR  packets  selected_SER  lock_ratio  selector_R2  gt_noisy_R2  selector_vs_GT_abs_pi
-22  28       0.101         0.420       0.853        0.722        0.309
-23  28       0.186         0.267       0.869        0.661        0.311
-24  28       0.307         0.165       0.775        0.671        0.357
-25  28       0.452         0.082       0.751        0.644        0.403
-26  28       0.584         0.046       0.651        0.596        0.446
-27  28       0.710         0.022       0.581        0.578        0.481
```

补充统计：

```text
corr(selector_vs_gt_noisy_resid_mean_abs_pi, selected_SER) ~= +0.554
corr(selector_line_rmse_pi, selected_SER) ~= +0.522
corr(locked_ratio, selected_SER) ~= -0.769
corr(selector_line_r2, selected_SER) ~= -0.355
```

解释：

```text
phase line 离 GT payload line 越远，SER 越高；
但 line R2 本身不够可靠，因为错误候选也能拟合出自洽的相位线。
```

### 2.6 运行 STO phase-jump 补偿对照

为 `run_low_snr_sto_phase_jump_experiment.py` 生成了 noisy IQ 输入：

```text
data/low_snr_gt_bin/0_0_0_10_14_16_m22_m27_sto_input/
```

该目录包含 noisy `.bin`，总大小约：

```text
1.36 GB
```

然后运行：

```text
tau_source=sfo_cum_before
phase_sign=plus
cfo_correction_mode=continuous
SNR=-22..-27 dB
```

输出：

```text
data/low_snr_gt_bin/0_0_0_10_14_16_m22_m27_sto_input/sto_phase_jump_corrected/
```

对照结果：

```text
SNR  base_R2  sto_R2  base_RMSE  sto_RMSE  base_argmax  sto_argmax
-22  0.904    0.918   0.336      0.324     0.397        0.392
-23  0.862    0.846   0.360      0.370     0.268        0.286
-24  0.851    0.834   0.382      0.391     0.166        0.174
-25  0.822    0.807   0.418      0.423     0.088        0.101
-26  0.763    0.759   0.474      0.417     0.049        0.052
-27  0.692    0.719   0.499      0.478     0.031        0.036
```

解释：

```text
STO phase-jump 补偿不是银弹。
它在 -22/-27 dB 有改善，中间 SNR 点略差或接近。
```

## 3. 涉及文件

本窗口新增：

```text
scripts/experiments/compare_phase_line_to_gt_payload.py
notes/handoffs/HANDOFF_PHASELINE_RETREAT_TO_OVERSAMPLING_ENHANCEMENT_2026-06-17.md
```

本窗口修改：

```text
data/ablation_offset_coherence_summary/ablation_report.md
scripts/experiments/run_low_snr_sto_phase_jump_experiment.py
```

本窗口生成数据：

```text
data/phase_line_gt_compare/m22_m27_all_datasets/
data/low_snr_gt_bin/0_0_0_10_14_16_m22_m27_phase_probe/
data/low_snr_gt_bin/0_0_0_10_14_16_m22_m27_sto_input/
data/low_snr_gt_bin/0_0_0_10_14_16_m22_m27_sto_input/sto_phase_jump_corrected/
```

注意大文件：

```text
data/low_snr_gt_bin/0_0_0_10_14_16_m22_m27_sto_input/
```

该目录保存 noisy IQ bin，总大小约 `1.36 GB`。如果后续只保留 CSV，可考虑清理 noisy bin，但不要顺手删，先确认是否还要画图或复现实验。

## 4. 重要设计决策和被否掉的方案

### 4.1 Phase line 不能再作为主证据

当前判断：

```text
phase line 不是完全没法用，但只能作为弱辅助特征；
不能继续作为主 decoder 证据押注。
```

理由：

- clean payload GT phase 有线性骨架，平均 R2 约 `0.872`。
- noisy GT payload phase 在 `-22..-27 dB` 明显变散。
- selector fitted line 有时 R2 很高，但可能拟合到错误候选路径上。
- header line 到 payload GT line 的 circular residual 很大，平均接近 `0.46π`。
- A6 packet-line phase only 消融表现很差。

### 4.2 不建议马上做 LSTM / BiLSTM

用户判断：

```text
深度神经网络结构上天生利好 decoder，只要有数据可以直接训练；
但 LSTM 作为弱信号 decoder 本身不一定适配，工程上也麻烦。
LSTM/BiLSTM 更适合做并发/序列相似性特征，但当前不应马上构建。
```

本窗口建议：

```text
先做特征工程，不训练模型。
如果相位动态特征在简单 ranker/MLP/GBDT 上有增益，再考虑序列模型。
```

候选特征方向：

```text
per candidate:
  energy_norm
  energy_rank
  energy_drop_db_vs_top1
  offset_coherence
  center_phase_sin/cos
  phase_residual_to_header_line_sin/cos
  phase_residual_to_selector_line_sin/cos
  peak_margin_db
  locked_mask
  symbol_index_normalized

local dynamics:
  delta_phase_sin/cos
  delta2_phase_sin/cos
  local_linear_residual
  local_slope_pi_per_symbol
  local_curvature
  window_phase_rmse
  window_phase_resultant_length

oversampling-specific:
  offset_phase_spread
  offset_phase_resultant_length
  pairwise_offset_phase_diff
  offset_energy_variance
  coherent_sum_amp / incoherent_sum_amp
```

### 4.3 新主线：oversampling 下弱信号叠加增强

应把当前成果重新表述为：

```text
multi-offset energy 是 oversampling 下的非相干增强底座。
offset coherence 是 oversampling offset 间复数一致性的可靠证据。
phase line 是 packet-level 全局相位结构，但在低 SNR 下不够稳定。
```

下一阶段应转向：

```text
如何在 oversampled LoRa signal 上进行更强的低复杂度叠加增强，
比 A Low-Complexity Demodulation for Oversampled LoRa Signal 更强。
```

### 4.4 继续避免的方向

仍然不要回到旧先验路线：

```text
payload byte template
counter prior
session prior
cross-packet joint prior
CRC-guided search
```

这些在前序窗口已经清理，不应作为主线复活。

## 5. 当前 git diff / git status 重点

这里有两个 git 视角，必须分清：

```text
D:/Desktop/proj             顶层 git repo
D:/Desktop/proj/gr-lora_sdr  gr-lora_sdr 子仓库 git repo
```

### 5.1 顶层 `proj` 仓库

顶层 repo：

```text
D:/Desktop/proj
```

当前 `git status --short`：

```text
D  _gen_run.py
D  test_creation.py
```

这两个删除与本任务无关，不要顺手处理。

从顶层 `proj` 视角看：

```powershell
git ls-files -- "gr-lora_sdr/weakPacket_decoding copy/*"
```

输出为空。

因此：

- 顶层 `proj` 仓库不跟踪 `gr-lora_sdr/weakPacket_decoding copy` 的内容。
- 在顶层 `proj` 运行 `git diff` 不会显示这个子仓库内部的改动。
- 不要用顶层 `proj` 的 git 状态判断 `gr-lora_sdr` 内部是否有改动。

### 5.2 `gr-lora_sdr` 子仓库

真正应该查看本任务改动的仓库是：

```powershell
cd D:\Desktop\proj\gr-lora_sdr
```

该仓库中：

```powershell
git ls-files -- "weakPacket_decoding copy/*"
```

会输出大量文件，说明 `weakPacket_decoding copy` 在 `gr-lora_sdr` 子仓库里是被跟踪的。

当前 `gr-lora_sdr` 子仓库的重点状态是：

```text
?? "weakPacket_decoding copy/notes/handoffs/HANDOFF_PHASELINE_RETREAT_TO_OVERSAMPLING_ENHANCEMENT_2026-06-17.md"
```

也就是说，本窗口新建的 handoff 文件还未被 add。其他本窗口修改/新增文件是否已跟踪，应在 `D:/Desktop/proj/gr-lora_sdr` 下用 `git status --short` 和 `git diff` 查看。

## 6. 已运行的测试命令和结果

### 6.1 py_compile

默认 Python：

```powershell
python -m py_compile scripts\experiments\compare_phase_line_to_gt_payload.py
python -m py_compile scripts\experiments\run_low_snr_sto_phase_jump_experiment.py
```

结果：通过。

`gr-lora` conda Python：

```powershell
& 'D:\mysoft2\miniconda3\envs\gr-lora\python.exe' -m py_compile `
  scripts\experiments\compare_phase_line_to_gt_payload.py `
  scripts\experiments\run_low_snr_sto_phase_jump_experiment.py
```

结果：通过。

### 6.2 matplotlib 验证

```powershell
& 'D:\mysoft2\miniconda3\envs\gr-lora\python.exe' -c "import matplotlib; print(matplotlib.__version__)"
```

结果：

```text
3.8.4
```

### 6.3 低 SNR GT-bin phase probe

```powershell
python scripts\experiments\run_low_snr_gt_bin_experiment.py `
  -i ..\data\USRP_IQ\0_0_0_10_14_16.bin `
  -g data\weak_sync_chain\header_first\0_0_0_10_14_16_header_first_symbols.csv `
  -o data\low_snr_gt_bin\0_0_0_10_14_16_m22_m27_phase_probe `
  --target-snr-db -22 -23 -24 -25 -26 -27 `
  --cfo-correction-mode continuous `
  --no-write-noisy-bin `
  --no-plots `
  --overwrite
```

结果：通过，输出 6 个 SNR 的 feature CSV、all features、summary、metadata。

### 6.4 phase line vs GT payload 对比

```powershell
python scripts\experiments\compare_phase_line_to_gt_payload.py `
  --datasets 0_0_0_10_14_8 0_0_0_10_14_16 0_0_0_10_14_32 `
  --snr-db -22 -23 -24 -25 -26 -27 `
  --output-dir data\phase_line_gt_compare\m22_m27_all_datasets
```

结果：通过。

输出：

```text
phase_line_gt_compare_packets.csv
phase_line_gt_compare_summary.csv
```

### 6.5 STO phase-jump 补偿

先生成 noisy IQ：

```powershell
python scripts\experiments\run_low_snr_gt_bin_experiment.py `
  -i ..\data\USRP_IQ\0_0_0_10_14_16.bin `
  -g data\weak_sync_chain\header_first\0_0_0_10_14_16_header_first_symbols.csv `
  -o data\low_snr_gt_bin\0_0_0_10_14_16_m22_m27_sto_input `
  --target-snr-db -22 -23 -24 -25 -26 -27 `
  --cfo-correction-mode continuous `
  --no-plots `
  --overwrite
```

结果：通过，生成 noisy IQ 和 CSV。

第一次运行 STO 脚本：

```powershell
python scripts\experiments\run_low_snr_sto_phase_jump_experiment.py ...
```

结果：CSV 写完后，画图阶段失败：

```text
ModuleNotFoundError: No module named 'matplotlib'
```

随后新增 `--no-plots` 并重跑：

```powershell
python scripts\experiments\run_low_snr_sto_phase_jump_experiment.py `
  -d data\low_snr_gt_bin\0_0_0_10_14_16_m22_m27_sto_input `
  -g data\weak_sync_chain\header_first\0_0_0_10_14_16_header_first_symbols.csv `
  --snr-db -22 -23 -24 -25 -26 -27 `
  --cfo-correction-mode continuous `
  --tau-source sfo_cum_before `
  --phase-sign plus `
  --no-plots
```

结果：通过。

## 7. 还没解决的问题

### 7.1 尚未复刻论文 baseline

还没有正式阅读和复刻：

```text
A Low-Complexity Demodulation for Oversampled LoRa Signal
```

下一窗口应先读论文正文、公式、实验设置和复杂度口径，再决定如何对齐 baseline。

### 7.2 尚未定义“超越”的指标

需要明确：

```text
SER threshold?
CRC/PRR threshold?
同等复杂度下的 gain?
同等 oversampling factor 下的 gain?
同等低 SNR 下的 packet recovery?
是否要求实时/低复杂度？
```

### 7.3 尚未把当前 offset coherence 理论化为 oversampling enhancer

当前已有：

```text
multi-offset energy
offset coherence
Top-L candidate recall
lock/rerank
```

但还没有整理成面向论文对比的统一算法描述。

### 7.4 尚未做候选级 feature table

如果后面考虑学习型 reranker 或轻量模型，需要先做：

```text
row = one payload symbol candidate bin
label = candidate_bin == GT_bin
features = energy/coherence/offset phase/context
```

现在还没有这个标准 feature dataset。

### 7.5 大文件需要后续管理

`0_0_0_10_14_16_m22_m27_sto_input` 约 1.36 GB。后续如果空间紧张，可清理 noisy `.bin`，但清理前要确认：

- 是否还要用 `gr-lora` 环境补画 PNG
- 是否还要复跑 STO 对照
- 是否要保留完整可复现输入

## 8. 下一步建议

### 8.1 先读论文并复刻 baseline

下一窗口第一步：

```text
读 A Low-Complexity Demodulation for Oversampled LoRa Signal。
提取它的 oversampling demod 核心公式、复杂度、实验参数和指标。
在当前数据集上复刻其 baseline。
```

不要先凭记忆改代码。

### 8.2 把当前结果重命名为 oversampling evidence

建议用这样的概念表述：

```text
E_k[b] = multi-offset fused energy
H_k[b] = offset coherence / offset phase consistency
A_k[b] = normalized amplitude/energy evidence
```

其中 `H_k[b]` 是目前最有潜力超越论文 baseline 的部分。

### 8.3 设计新算法时优先低复杂度

可能方向：

```text
1. offset coherent/noncoherent hybrid combining
2. offset phase alignment before summation
3. per-offset reliability weighting
4. energy-gated coherence rerank
5. adaptive Top-L / confidence lock
6. offset-consistency-aware soft demod
```

重点是把增强发生在 oversampling FFT evidence 层，而不是 payload byte 或 CRC 层。

### 8.4 Feature engineering 先行，暂不训练 LSTM

如果要探索学习型方法，先做 feature table：

```text
candidate-level static features
local phase dynamics
offset-level coherence features
```

先用简单模型或统计验证：

```text
logistic regression
GBDT / random forest
pairwise ranker
small MLP
```

只有当这些证明 phase/offset dynamic features 有独立增益后，再考虑 LSTM/BiLSTM。

### 8.5 后续出图用 gr-lora 环境

如果要画图，使用：

```powershell
& 'D:\mysoft2\miniconda3\envs\gr-lora\python.exe' <script>
```

不要用默认 Python 直接跑需要 `matplotlib` 的脚本，除非加 `--no-plots`。

## 一句话交接

我们带着证据从 phase-line 强化解码撤退：phase line 有弱信息，但主证据不稳；下一阶段要把当前 multi-offset energy + offset coherence 重新组织成 oversampled LoRa weak-signal enhancement，先复刻 `A Low-Complexity Demodulation for Oversampled LoRa Signal`，再用更强的 offset phase/energy 融合方法超过它。
