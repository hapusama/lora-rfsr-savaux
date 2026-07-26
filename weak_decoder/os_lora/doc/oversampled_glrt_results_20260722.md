# 结构保持的过采样 LoRa 解调：实现与首轮结果

## 1. 接收器实现

实现位于 `system/oversampled_glrt.py`，统一评估入口为
`experiments/evaluate_oversampled_glrt.py`。payload ground truth 只用于离线计分，
不参与同步状态、协方差、steering、候选生成或最终判决。

第一阶段把候选 `k` 的 Savaux polyphase branch 表示为 `R=OSR` 维复向量
`y(k)`，计算低维 GLS 分数

\[
\Lambda_{\mathrm B}(k)=
\frac{|a^H C_k^{-1}y(k)|^2}{a^H C_k^{-1}a}.
\]

默认协方差形状为 `N x R x R`，即每个候选一套 `R x R` 小矩阵，从不构造
`RN x RN` 稠密矩阵。`C_k` 由 off-packet windows 按当前 packet 的 CFO 重新
估计；使用更多 snapshot 的 pooled covariance 仅用于检测 branch 相关性和方差
不均衡。若噪声近似白噪声，门控使接收器严格退化到 Savaux。信号 steering 由
preamble 和 explicit header 联合估计，但白噪声门启用时同时恢复全 1 steering，
保证判决严格等于 Savaux。主链直接使用上述纯 GLS 分数；`0.25` 归一化 GLS 加
`0.75` Savaux 仅保留为 `branch_shrinkage` 消融，不参与 Proposed 判决。

第二阶段只检查第一阶段的 Top-8 候选，并从完整 `RN` 点 dechirped 序列中在

\[
K_1=k+f_s,\qquad K_2=k+f_s-N\pmod{RN}
\]

提取两个 fractional-bin 分量 \(A_{1,k}\) 和 \(A_{2,k}\)。两点在 Hz 域相隔
\(F_s-BW\)，在 `RN` 点 DFT 中相隔 \((R-1)N\) 个 bin；因此第二个位置也可写为
\(k+f_s+(R-1)N\)。按 preamble/header 估计出的 residual frequency 与 fold timing
预测候选相关相位 \(\hat\phi_{s,k}\)，再计算

\[
\rho(k)=
\frac{\left|A_{1,k}+A_{2,k}e^{-j\hat\phi_{s,k}}\right|^2}
{2\left(\left|A_{1,k}\right|^2+\left|A_{2,k}\right|^2\right)+\epsilon}.
\]

因子 2 只把分数归一化到 `[0,1]`，不改变候选排序。这里不估计双峰 `2 x 2`
协方差，也不使用双峰幅度 steering 或 whitening；旧 pair GLRT 仅作为
`savaux_dual` 消融保留。默认判决也不无条件选择最大 \(\rho\)：归一化相干度会让
低能量噪声候选偶然取得高分，因此只有当候选距 GLS 第一名不超过 `0.15 dB`，且
相干度至少增加 `0.30` 时才允许重判。`coherence`-only 与 joint-log 规则保留为
消融选项。当前冻结实验先使用 packet 内常数 timing；代码已保留按上游 `sfo_hat`
施加正/负 timing slope 的消融入口，但在其符号与尺度完成独立验证前不把它默认为
已正确的 SFO 相位轨迹。

此外，接收器利用 explicit header 检测整数 bin residue。若至少 8 个 header 观测
中有 75% 一致支持 `+1` 或 `-1` bin 修正，则在输出 hard symbol 前施加包级校准。
该步骤只依赖 explicit header 合法 FFT bin 的 `1 mod 4` 结构，不需要先知道 header
内容；它是同步状态校准，而不是 payload-aided 纠错。评估器同时输出
`savaux_header`，用于把该收益与 branch GLS、双折返重判明确分开。

## 2. 冻结参数后的有色噪声 holdout

开发使用 seeds `42--45`；下表使用未参与调参的 seeds `50/51/52`、全部 7 个包，
每个 SNR 共 735 个 payload symbols。注入噪声为 ADC-rate 截断复 AR(1)：

\[
h[\ell]=(0.85e^{j0.7})^\ell,\qquad 0\le \ell<65.
\]

这是一项可复现的 covariance stress test，不代表实测空口噪声。

| SNR (dB) | FFT | Savaux | Branch GLS | Branch shrinkage | Proposed | UniChirp | SymFEC | LoRaTrimmer |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -18 | 0.1823 | 0.1279 | **0.1211** | 0.1252 | 0.1224 | 0.2163 | 0.1483 | 0.1415 |
| -20 | 0.3810 | **0.1605** | 0.1701 | 0.1592 | 0.1687 | 0.2952 | 0.4707 | 0.2286 |
| -22 | 0.6857 | 0.3510 | **0.3388** | 0.3442 | **0.3388** | 0.4803 | 0.8082 | 0.5061 |

跨三个 SNR，Savaux、纯 Branch GLS、branch shrinkage 和 Proposed 的错误数分别为
`470/2205`、`463/2205`、`462/2205` 和 `463/2205`：

- 纯 Branch GLS 相对 Savaux 净减少 7 个错误；
- 冻结后的相干度重判相对纯 Branch GLS 有 3 fixes、3 breaks，净收益为 0；
- 完整系统相对 Savaux 减少 7 个错误，错误数降低 1.49%，该净收益来自第一阶段 GLS，
  不能归因于第二阶段。

开发 seeds `42--45` 上，相干度门控相对纯 GLS 有 5 fixes、1 break；但冻结参数后
在未见 seeds `50--52` 上没有净收益。因此当前结果支持“已经按预期提取并使用了
完整采样相位结构，且保守门控未造成总体退化”，尚不支持“第二阶段已稳定降低
SER”。纯 GLS 在 -18 和 -22 dB 优于 Savaux，但 -20 dB 略差；后续应增加独立
信道/capture 验证并改善 STO/SFO 相位轨迹，而不是继续在本 holdout 上调门限。

同一 holdout 上的 `coherence`-only 消融在 -18、-20、-22 dB 分别为
`118/735`、`238/735`、`406/735` 个错误，合计 `762/2205`，明显差于纯 GLS 的
`463/2205`。这验证了相干度适合作为近似等能候选之间的一致性证据，不能脱离
第一阶段能量单独充当最终似然。

## 3. AWGN 退化检查

新 seed `53`、全部 7 包在 -22、-23、-24、-25 和 -26 dB 的检查中，Savaux、
header calibration、纯 Branch GLS、branch shrinkage、旧 pair GLRT 和 Proposed
的 hard decisions 完全相同。白噪声门同时安装单位协方差和全 1 steering，因此这一退化
由实现逻辑保证，而不只是本次随机 seed 恰好相同。更早的 seeds `42/43/44` sweep
也得到相同结论。

## 4. 真实 CR=4/7 capture

外部 ground truth 来自同一固定帧的 high-SNR capture；待测 IQ 来自
`USRP_collector/data/branch4_fixed/low_snr`。这些 capture 的
`coding_rate_index=3`，即 LoRa CR=4/7。噪声协方差由独立 capture 的 off-packet
窗口估计。

### low2：header 可观测的整 bin 同步偏差

15 个严格同步帧，共 735 个 payload symbols：

| 方法 | errors | SER |
|---|---:|---:|
| Ordinary FFT | 98/735 | 0.1333 |
| Savaux | 98/735 | 0.1333 |
| UniChirp | 109/735 | 0.1483 |
| SymFEC | 39/735 | 0.0531 |
| LoRaTrimmer | 98/735 | 0.1333 |
| Savaux + header calibration | **0/735** | **0** |
| Proposed | **0/735** | **0** |

错误集中在 packet 4 和 6：两包各有 49 个 payload symbols，Savaux 全部比 ground
truth 小 1 bin。两包各自的 8 个 header 观测均一致给出 residue `-1`，因此接收器
施加 `+1` bin 校准并修复全部 98 个错误；其余 13 包的校准量为 0，判决不变。
`savaux_header` 与 Proposed 同为 0 错误，故这里的全部收益应归因于 header 同步
校准，不能归因于 branch GLS 或双折返模块。

该结果验证了方法能够在不查看 payload ground truth 的前提下修复一种真实、包级
系统性失效，但有效异常包只有 2 个，尚不足以证明对任意信道或硬件状态的统计泛化。

### low1、low3 和 low4：安全性检查

- low1：Savaux 与 Proposed 均为 `0/539`；
- low3：Savaux 与 Proposed 均为 `0/588`；
- low4：Savaux 与 Proposed 均为 `1/392`，而 FFT、UniChirp、SymFEC、
  LoRaTrimmer 分别为 `27/392`、`53/392`、`3/392`、`2/392`。

因此 low1--low3 合计时，Proposed 为 `0/1862`，Savaux 为 `98/1862`；low4 中则
与强 Savaux 基线持平，没有修复剩余的单个错误。真实数据结论应表述为“已修复
low2 的 header 可观测整 bin 偏差且未破坏其余已测 capture”，而不是“所有真实
弱包条件下都普遍大幅优于 Savaux”。

## 5. 复现

冻结参数的有色噪声完整 baseline 对比：

```powershell
python -m weak_decoder.os_lora.experiments.evaluate_oversampled_glrt `
  --datasets 0_0_0_10_14_32 --snrs -18 -20 -22 --seeds 50 51 52 `
  --max-packets 7 --noise-shape ar1 --noise-filter-taps 65 `
  --noise-color-magnitude 0.85 --noise-color-phase-rad 0.7 `
  --noise-windows 256 --output-dir `
  data\experiments\oversampled_coherent_fold_holdout_full50_52_20260722
```

主要结果目录：

- `data/experiments/oversampled_coherent_fold_candidate_phase_dev42_45_20260722`
- `data/experiments/oversampled_coherent_fold_holdout_full50_52_20260722`
- `data/experiments/oversampled_coherent_fold_coherence_only_holdout50_52_20260722`
- `data/experiments/oversampled_coherent_fold_awgn_seed53_20260722`
- `data/experiments/oversampled_glrt_header_cal_real_low1_20260722`
- `data/experiments/oversampled_coherent_fold_real_low2_20260722`
- `data/experiments/oversampled_glrt_header_cal_real_low3_20260722`
- `data/experiments/oversampled_coherent_fold_real_low4_20260722`

每个目录保存运行参数、逐 seed 汇总、逐符号候选诊断和协方差统计。完整采样实验
保留 `predicted_fractional_bin`、coherent ratio、phase residual、候选能量损失及
header calibration 诊断，便于逐次核查 fix/break，而不只保留最终 SER。
