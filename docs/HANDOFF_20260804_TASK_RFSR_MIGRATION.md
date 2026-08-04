# Task-aware RF-SR 开发迁移 Handoff（2026-08-04）

这份文件用于把当前工作从付费 AutoDL 服务器迁移到另一台机器。它优先记录“如何无损接手”，
完整实验解释见 [`docs/OFFICIAL_TASK_RFSR_20260804.md`](OFFICIAL_TASK_RFSR_20260804.md)。

## 0. 最重要的三件事

1. **当前新增代码尚未 commit。** 只 clone GitHub 会丢失本轮全部 task-aware 工作；离开旧服务器前
   必须 commit/push，或直接打包整个 worktree。
2. **模型和实验结果在仓库外。** `/root/autodl-tmp/rfsr-run/` 下的 checkpoint/JSON 需要单独复制。
3. **官方数据也在仓库外。** 当前数据根目录总占用约 118 GiB；可以复制完整目录，也可以只复制
   约 26.4 GiB 的 `.zst` 后在新服务器解压。

## 1. 当前 Git 状态

```text
repository: /root/lora-rfsr-savaux
branch: main
origin: https://github.com/hapusama/lora-rfsr-savaux.git
base commit: 4dd12c55a94d134083649e665addb5bd19199171
base subject: 神经网络重设计
vendored RFSR upstream: 00c135947f855790f458fdc25ae9533c70d77849
```

当前未提交文件：

```text
 M third_party/rfsr/rfsr/nn/__init__.py
?? docs/OFFICIAL_TASK_RFSR_20260804.md
?? docs/HANDOFF_20260804_TASK_RFSR_MIGRATION.md
?? third_party/rfsr/rfsr/nn/official_ota_dataset.py
?? third_party/rfsr/rfsr/nn/task_loss.py
?? third_party/rfsr/rfsr/nn/task_model.py
?? third_party/rfsr/rfsr/nn/task_pretraining_dataset.py
?? third_party/rfsr/tests/test_task_aware_rfsr.py
?? tools/evaluate_official_task_rfsr.py
?? tools/evaluate_task_rfsr_symbol_metrics.py
?? tools/pretrain_task_rfsr.py
?? tools/train_official_task_rfsr.py
```

推荐在旧服务器先建立独立分支再提交，避免直接污染 `main`：

```bash
cd /root/lora-rfsr-savaux
git switch -c task-aware-rfsr-20260804
git add \
  docs/OFFICIAL_TASK_RFSR_20260804.md \
  docs/HANDOFF_20260804_TASK_RFSR_MIGRATION.md \
  third_party/rfsr/rfsr/nn/__init__.py \
  third_party/rfsr/rfsr/nn/official_ota_dataset.py \
  third_party/rfsr/rfsr/nn/task_loss.py \
  third_party/rfsr/rfsr/nn/task_model.py \
  third_party/rfsr/rfsr/nn/task_pretraining_dataset.py \
  third_party/rfsr/tests/test_task_aware_rfsr.py \
  tools/evaluate_official_task_rfsr.py \
  tools/evaluate_task_rfsr_symbol_metrics.py \
  tools/pretrain_task_rfsr.py \
  tools/train_official_task_rfsr.py
git commit -m "Add two-stage task-aware RF-SR experiments"
git push -u origin task-aware-rfsr-20260804
```

上面只是建议命令；执行前仍应自行 review `git diff --cached`。

如果不方便 push，至少保存 Git bundle 和未跟踪文件；最简单可靠的是直接打包整个仓库：

```bash
tar -C /root -czf /root/autodl-tmp/lora-rfsr-savaux-worktree-20260804.tar.gz \
  lora-rfsr-savaux
sha256sum /root/autodl-tmp/lora-rfsr-savaux-worktree-20260804.tar.gz
```

## 2. 本轮到底实现了什么

### 第一版：按 `神经网络重设计` 文档实现

```text
同一 OTA 250 kS/s x
  -> low-grid TCN
  -> 线性 polyphase baseline + 三个缺失相位残差
  -> 同一 OTA received 1 MS/s y
```

- 58,790 参数，32 channels，dilation `1,2,4,8,16,32,64,128`；
- phase 0 硬复制，`output[..., ::4] == input`；
- complex RMS normalization；
- complex Charbonnier + 四支路非相干 concentration + margin；
- 从零训练，**没有预训练**。

8-reference/-15 dB 筛查：

| Decoder | 插值 | 第一版 scratch |
| --- | ---: | ---: |
| FFT | 23/320 | 24/320 |
| Savaux | 23/320 | 23/320 |
| Savaux+GLS | 23/320 | 37/320 |

原因是第一版 task loss 只优化支路功率，没有约束 Savaux 相干相位；网络还改变了 GLS 所依赖的
branch noise covariance。

### 第二版：保留论文价值后的两阶段重建

保留原 RF-SR 的三个合理思想：

1. 合法随机 payload synthetic pretraining；
2. 插值 prior + learned residual；
3. synthetic pretrain -> OTA fine-tune。

新增/修正：

- radius-16 Kaiser sinc polyphase baseline；
- 合法 gr-lora header/payload symbol 生成；
- 先在 1 MS/s 施加 multipath/CFO/STO/SFO/gain/phase/AWGN，再严格抽取 250 kS/s；
- 精确可微 Savaux Eq. (36)-(37) 八支路相干 loss；
- hard-observed 与 soft-observed 两种模型；
- residual strength、exact-bin、双 sync/decode strength 消融；
- 按 reference ID 绑定的 60/20/20 split 和 cluster-paired 统计；
- packet/method 原子评估缓存。

Torch Savaux 与 NumPy reference 的最大误差：随机波形 `3.1e-6`，clean chirp `6.4e-5`。

## 3. 当前实验结论（接手时不要误读）

### 最终 held-out test

测试使用全部 20 个 test reference，每个选一个最接近 -15 dB 的 OTA capture，共 800 symbols。
reference 99 没有 -15 dB capture，使用其最接近的 -8.41 dB。所有参数先在 validation 锁定。

Hard-observed，`residual_strength=0.5`：

| Decoder | 插值错误 | RF-SR 错误 | 插值 median margin | RF-SR median margin |
| --- | ---: | ---: | ---: | ---: |
| FFT | 121/800 | 121/800 | 4.899 dB | **5.448 dB** |
| Savaux | 121/800 | 121/800 | **7.414 dB** | 7.345 dB |
| Savaux+GLS | 121/800 | 121/800 | **7.447 dB** | 7.325 dB |

Soft-observed，`residual_strength=0.5`：

| Decoder | 插值错误 | RF-SR 错误 | 插值 median margin | RF-SR median margin |
| --- | ---: | ---: | ---: | ---: |
| FFT | 121/800 | 121/800 | 4.899 dB | **5.289 dB** |
| Savaux | 121/800 | 121/800 | 7.414 dB | **7.439 dB** |
| Savaux+GLS | 121/800 | 121/800 | **7.447 dB** | 7.155 dB |

两种模型与插值都同步 18/20。三个 decoder 的 reference-cluster paired SER difference 都为 0；
**当前没有证据表明网络超过插值。** 网络改善了一些 proxy loss/margin，但没有改变 hard decision。

### 已否决、不要直接重复的路线

- 第一版无预训练 scratch：GLS 从 23/320 恶化到 37/320。
- exact-bin 强监督：窗口 exact accuracy `66.4% -> 77.4%`，但整包只同步 7/8，Savaux
  变为 63/320。
- soft `strength=1`：validation 只同步 3/4。
- sync strength 0.5 / decode strength 1.0：validation FFT 产生 12/160 错误，margin 下降。
- 单纯继续增加 TCN、epoch 或 waveform 权重：目前没有下游证据支持。

### 下一步最值得做什么

不要把下一个目标继续定义成“生成更像 1 MS/s 的 IQ”。250 kS/s 对 125 kHz complex-baseband
LoRa 已保留主要信息，插值很强。更合理的下一步是：

1. 直接从 low-rate/branch spectrum 学 candidate posterior 或 LLR；
2. 联合同步、CFO/SFO、symbol ranking 和 CRC/PER；
3. 或把输入降到 125/62.5 kS/s，构造真正缺信息的 RF-SR；
4. 如果仍做 waveform supervision，先建立可靠的 OTA received -> aligned clean reference target，
   不要把未对齐 `signalout` 直接和硬观测约束混用。

## 4. 必须复制的实验产物

四个 run 目录合计不到 4 MiB，建议全部复制，而不是只拿 `best.pt`；目录内还有 config、history、
split manifest、summary 和关键 evaluation JSON。

| 用途 | 路径 | Checkpoint SHA-256 |
| --- | --- | --- |
| hard synthetic pretrain | `/root/autodl-tmp/rfsr-run/task_aware_pretrain_physical_savaux_seed42/` | `eddc3d50a15a919157b1e6f7b764937f025e2774d3395a9edabaeeee2b8c7371` |
| hard OTA fine-tune | `/root/autodl-tmp/rfsr-run/task_aware_official_pretrained_savaux_seed42/` | `a6f5398b90d9fd457a2f6047d9adc6b1686e78813c8a588a85b45a00f89aff25` |
| soft synthetic pretrain | `/root/autodl-tmp/rfsr-run/task_aware_pretrain_soft_savaux_seed42/` | `4fde0d526140561b5c998fa7a7d58c190632e691a93dacc86b13b5a4239fd989` |
| soft OTA fine-tune | `/root/autodl-tmp/rfsr-run/task_aware_official_soft_savaux_seed42/` | `b84e20fddf54579eae95cc37d4a461cb17a14121064833e4477fb055d26a8ae9` |

推荐打包：

```bash
tar -C /root/autodl-tmp/rfsr-run -czf \
  /root/autodl-tmp/task-aware-rfsr-artifacts-20260804.tar.gz \
  task_aware_pretrain_physical_savaux_seed42 \
  task_aware_official_pretrained_savaux_seed42 \
  task_aware_pretrain_soft_savaux_seed42 \
  task_aware_official_soft_savaux_seed42
sha256sum /root/autodl-tmp/task-aware-rfsr-artifacts-20260804.tar.gz
```

最重要的最终 JSON：

```text
# hard, strength=0.5, 20-reference test
/root/autodl-tmp/rfsr-run/task_aware_official_pretrained_savaux_seed42/
  eval_test20_alpha050.json

# soft, strength=0.5, 20-reference test
/root/autodl-tmp/rfsr-run/task_aware_official_soft_savaux_seed42/
  eval_test20_alpha050.json
```

## 5. 官方数据迁移

当前根目录：

```text
/root/autodl-tmp/.autodl/rfsr_ota_official
```

目录状态：

```text
metadata/*.json      100 packet/reference metadata
ota/*.cfile          2,668 个已解压 OTA
reference/*.cfile    100 个已解压 reference
ota/*.cfile.zst      相同 OTA 的压缩副本
reference/*.zst      相同 reference 的压缩副本
README.md
dataverse_manifest.json

总占用（raw + zst）：约 118 GiB
raw IQ：约 91.45 GiB
zst：约 26.4 GiB
```

这不是官方全部 10,000 OTA；是全部 100 reference 加当前下载的 2,668 OTA。

### 方案 A：新服务器空间足够，直接复制完整目录

```bash
rsync -aH --info=progress2 \
  OLD_HOST:/root/autodl-tmp/.autodl/rfsr_ota_official/ \
  /NEW_DATA_DISK/rfsr_ota_official/
```

需要至少约 120 GiB 空间。

### 方案 B：只复制压缩副本，再解压

只传 `.zst`、压缩 sidecar 和 metadata，大约 26.4 GiB：

```bash
rsync -a --info=progress2 \
  --include='*/' \
  --include='*.zst' \
  --include='*.zst.meta.json' \
  --include='README.md' \
  --include='dataverse_manifest.json' \
  --include='metadata/*.json' \
  --exclude='*' \
  OLD_HOST:/root/autodl-tmp/.autodl/rfsr_ota_official/ \
  /NEW_DATA_DISK/rfsr_ota_official/
```

在新服务器解压（先确认至少约 95 GiB 额外可用空间）：

```bash
find /NEW_DATA_DISK/rfsr_ota_official/ota \
     /NEW_DATA_DISK/rfsr_ota_official/reference \
  -type f -name '*.cfile.zst' -print0 \
  | xargs -0 -n1 -P4 zstd -d --keep
```

验证数量：

```bash
find /NEW_DATA_DISK/rfsr_ota_official -type f -name '*.cfile' | wc -l
# 预期 2768

find /NEW_DATA_DISK/rfsr_ota_official/ota -type f -name '*.cfile' | wc -l
# 预期 2668

find /NEW_DATA_DISK/rfsr_ota_official/reference -type f -name '*.cfile' | wc -l
# 预期 100
```

所有训练/评估入口都通过 `--official-root` 接收新路径，不需要把数据放回原来的绝对路径。

## 6. 旧服务器环境

已验证环境：

```text
OS: Linux
Python: 3.10.8
PyTorch: 2.1.2+cu121
CUDA runtime: 12.1
NumPy: 1.26.3
SciPy: 1.15.3
GPU: NVIDIA GeForce RTX 3090, 24 GiB
driver: 580.105.08
```

注意：旧环境没有安装 `pytest`，测试使用 `unittest`。`rg` 也不在当前系统里，但不影响运行。

新服务器最少需要：

```text
Python 3.10+
PyTorch（CUDA 版本按新机器驱动选择）
NumPy
SciPy
zstandard/zstd（只在解压数据时需要）
```

vendored RFSR 通过 `PYTHONPATH=.:third_party/rfsr` 使用，不要求先安装成 wheel。

## 7. 新服务器最短恢复流程

```bash
# 1. 获取含本轮提交的分支
git clone https://github.com/hapusama/lora-rfsr-savaux.git
cd lora-rfsr-savaux
git switch task-aware-rfsr-20260804

# 2. 解压小型实验产物
mkdir -p /NEW_DATA_DISK/rfsr-run
tar -C /NEW_DATA_DISK/rfsr-run -xzf task-aware-rfsr-artifacts-20260804.tar.gz

# 3. 放好或解压官方数据
export RFSR_OFFICIAL_ROOT=/NEW_DATA_DISK/rfsr_ota_official

# 4. 语法与单元测试
python -m py_compile \
  third_party/rfsr/rfsr/nn/official_ota_dataset.py \
  third_party/rfsr/rfsr/nn/task_loss.py \
  third_party/rfsr/rfsr/nn/task_model.py \
  third_party/rfsr/rfsr/nn/task_pretraining_dataset.py \
  tools/pretrain_task_rfsr.py \
  tools/train_official_task_rfsr.py \
  tools/evaluate_task_rfsr_symbol_metrics.py \
  tools/evaluate_official_task_rfsr.py

PYTHONPATH=.:third_party/rfsr \
  python -m unittest discover -s third_party/rfsr/tests -v
```

旧服务器最终测试结果：

```text
third_party/rfsr/tests: 22 passed
相关 Savaux/GLS/官方评估链: 50 passed
合计: 72 passed
```

## 8. 最小 smoke test

不需要 OTA 数据的预训练 smoke：

```bash
python tools/pretrain_task_rfsr.py \
  --run-dir /tmp/task-rfsr-smoke \
  --train-items 4 --validation-items 2 \
  --symbols-per-item 2 --epochs 1 \
  --batch-size 1 --workers 0 \
  --channels 8 --dilations 1 2 4 --no-amp
```

需要 OTA 数据的 checkpoint symbol gate：

```bash
python tools/evaluate_task_rfsr_symbol_metrics.py \
  --official-root "$RFSR_OFFICIAL_ROOT" \
  --output /tmp/task-rfsr-symbol-gate.json \
  --checkpoint hard=/NEW_DATA_DISK/rfsr-run/task_aware_official_pretrained_savaux_seed42/best.pt \
  --capture-limit 8 --symbols-per-capture 2 \
  --batch-size 1 --workers 0
```

## 9. 正式复现命令

### Hard synthetic pretraining

```bash
python tools/pretrain_task_rfsr.py \
  --run-dir /NEW_DATA_DISK/rfsr-run/task_aware_pretrain_physical_savaux_seed42 \
  --train-items 1000 --validation-items 200 \
  --symbols-per-item 4 --epochs 8 \
  --batch-size 2 --workers 4 --seed 42
```

### Hard official OTA fine-tune

```bash
python tools/train_official_task_rfsr.py \
  --official-root "$RFSR_OFFICIAL_ROOT" \
  --run-dir /NEW_DATA_DISK/rfsr-run/task_aware_official_pretrained_savaux_seed42 \
  --pretrained /NEW_DATA_DISK/rfsr-run/task_aware_pretrain_physical_savaux_seed42/best.pt \
  --epochs 8 --waveform-warmup-epochs 1 \
  --symbols-per-capture 4 --batch-size 2 --workers 4 \
  --validation-capture-limit 160 --seed 42 --split-seed 42
```

### 20-reference held-out test

```bash
python tools/evaluate_official_task_rfsr.py \
  --official-root "$RFSR_OFFICIAL_ROOT" \
  --checkpoint /NEW_DATA_DISK/rfsr-run/task_aware_official_pretrained_savaux_seed42/best.pt \
  --output /NEW_DATA_DISK/rfsr-run/task_aware_official_pretrained_savaux_seed42/eval_test20_alpha050.json \
  --split test --captures-per-reference 1 \
  --selection target_snr --selection-snr-db -15 \
  --calibration-captures 24 \
  --residual-strength 0.5 --sync-mode common_native \
  --bootstrap-repetitions 10000 --permutation-repetitions 50000 \
  --seed 20260804
```

## 10. 代码兼容性和容易踩的坑

- `TaskAwarePolyphaseTCN` 默认仍是 `baseline="linear", hard_observed=True`，保证旧 checkpoint
  config 能加载；新训练 CLI 默认使用 `sinc + savaux`。
- `TaskAwareRFSRLoss` 类默认 `spectral_mode="four_branch"` 用于旧结果兼容；正式新训练器显式/默认
  选择 `savaux`。独立调用 loss 时不要忘记指定。
- checkpoint schema 是 `task-aware-rfsr-v1`，通过 `load_task_aware_checkpoint()` 加载。
- official trainer 的 `--pretrained` 会以 checkpoint 内的 `model_config` 为准，CLI 中 channels、
  dilation、baseline 不会覆盖预训练结构。
- `target_source="received"` 才满足 hard coordinate；`reference` 模式不是 OTA input -> reference
  target，而是从 reference 同时构造 x/y。不要误用来声称复现论文 OTA->reference。
- `valid_mask` 会排除 hard phase 0 的 waveform loss；它是结构不变量，不是需要优化的输出。
- 评估默认 `common_native`：共享 native packet 候选，再在每个前端上做精同步验证。失败同步按整包
  symbol error 计入 SER。
- interpolation/native 的缓存与 checkpoint 无关；task cache 绑定 checkpoint SHA、residual strength、
  GLS 参数和源文件 size/mtime。换数据或参数后应允许它自动 miss，不要手改 JSON。
- full test 结果中 121/800 很大一部分与 2/20 sync failure 有关；不要只看已同步子集后宣称网络改善。

## 11. 相关文档

```text
docs/OFFICIAL_TASK_RFSR_20260804.md
  本轮网络、训练和全部 hard/soft 消融的正式实验记录

docs/HANDOFF_20260731_RFSR_OFFICIAL_OTA_VALIDATION.md
  官方数据来源、公开 checkpoint 和早期 OTA 验证背景

docs/ANALYSIS_20260730_OFFICIAL_RFSR_SYNTHETIC_CHAIN.md
  早期官方 synthetic/checkpoint 与 Savaux 链分析

docs/HANDOFF_20260728_RFSR_OTA.md
  更早的 OTA 数据处理背景
```

## 12. 接手后的建议顺序

1. 先确认 Git 分支包含本 handoff 列出的 11 个新增/修改文件。
2. 校验四个 checkpoint SHA-256。
3. 跑 72 个相关测试和最小 synthetic smoke。
4. 确认 official archive 数量是 2,668 OTA + 100 reference。
5. 用已有 checkpoint 跑 8-capture symbol gate，不要先重新训练。
6. 阅读最终 20-reference JSON，确认迁移前后统计一致。
7. 再开始 decoder-native LLR/同步联合学习的新分支。

完成以上步骤后，新服务器就已经恢复到本次 AutoDL 实验结束时的可复现状态。
