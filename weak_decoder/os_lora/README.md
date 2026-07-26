# OS-LoRa 弱包解码模块

本目录按“系统实现不能依赖实验代码、实验入口不能互相依赖”的原则拆分。
后续接入实时或离线解码链时只需要依赖 `system/`；复现实验、生成表格和绘图时
才使用 `experiments/`。

## 目录职责

```text
os_lora/
├── system/       可复用的解码算法与数据结构
├── experiment_support/  多个实验共享的非在线基础设施
├── experiments/  可独立删除和运行的评估、消融、诊断、标定与绘图入口
├── doc/          算法说明与历史实验记录
└── __init__.py   稳定的公共接口
```

`system/` 当前包含：

- `nonuniform_sampling.py`：非均匀采样 pattern、频谱打分、GLS、条件检测器等核心算法。
- `chirp_svd.py`：ChirpSVD 的配置、训练与候选打分实现。
- `noise.py`：背景频点选择等共享噪声处理工具。
- `litenap_savaux.py`：LiteNap 式真欠采样 polyphase 观测、Savaux 候选相位
  合并和可选 phase-jump 指纹重排；`K=D` 时与完整 Savaux 数值等价。
- `oversampled_glrt.py`：Savaux branch 的低维 GLS、完整采样双折返分量提取、
  相干/非相干功率比、explicit-header 整 bin 校准和置信门控重判。主链只估计
  `OSR x OSR` 的 branch 协方差，不构造 `RN x RN` 稠密矩阵。模块中原有的
  `2 x 2` 双峰协方差与 pair GLRT 仅保留为 `savaux_dual` 历史消融，不参与
  `proposed` 判决。当离包 branch 噪声近似白噪声时，默认门控使判决退化到
  Savaux；`--allow-white-fold-overrides` 可用于对应消融。

`experiments/` 中的脚本可以依赖 `system/` 和 `experiment_support/`，但不得导入
另一个实验入口；因此删除任意一个实验脚本不会导致其他实验出现导入错误。
`system/` 不得依赖另外两个目录。实验脚本中的 CSV 字段名和算法标识仍保留英文，
以免破坏已有结果、绘图脚本和论文数据处理流程；源代码注释与说明统一使用中文。

## 系统代码的导入方式

新代码优先从稳定公共接口导入：

```python
from weak_decoder.os_lora import build_pattern_bank, conditional_lora_gls_detect
```

需要明确模块来源时也可以直接导入：

```python
from weak_decoder.os_lora.system.nonuniform_sampling import build_pattern_bank
from weak_decoder.os_lora.system.noise import select_background_bins
```

## 运行实验

请从 `weakPacket_decoding` 目录以模块方式运行，避免脚本移动后出现相对路径问题：

```powershell
python -m weak_decoder.os_lora.experiments.evaluate_real_capture_gls --help
python -m weak_decoder.os_lora.experiments.evaluate_low_complexity_gls --help
python -m weak_decoder.os_lora.experiments.evaluate_nonuniform_sampling --help
python -m weak_decoder.os_lora.experiments.evaluate_oversampled_glrt --help
python -m weak_decoder.os_lora.experiments.evaluate_litenap_savaux --help
```

多 SNR 的统一 baseline 对比示例：

```powershell
python -m weak_decoder.os_lora.experiments.evaluate_oversampled_glrt `
  --datasets 0_0_0_10_14_8 0_0_0_10_14_16 0_0_0_10_14_32 `
  --snrs -22 -23 -24 -25 -26 --seeds 42 43 44 `
  --output-dir data\experiments\oversampled_glrt
```

输出包括逐 seed 汇总 `summary_by_seed.csv`、跨 seed 的 `summary.csv`、逐符号
候选/fix/break 诊断 `symbols.csv`、协方差颜色统计 `covariance.csv` 和 SER 曲线。
评估器还支持 `--noise-shape lowpass|ar1` 的可复现 ADC-rate 有色噪声压力测试；
它只用于验证 covariance-aware 接收器，不能替代真实有色干扰 capture。
逐符号输出中的 `savaux_header` 用于隔离 header 整 bin 校准的贡献；
`branch_shrinkage` 是 GLS/Savaux 分数混合消融；`branch_gls` 是主链第一阶段，
`proposed` 表示纯 GLS 加完整采样双分量相干度重判及 header 校准；`savaux_dual`
只代表旧的 pair-GLRT 消融。当前冻结参数的结果、
真实 CR=4/7 验证和可复现命令见
`doc/oversampled_glrt_results_20260722.md`。

LiteNap-Savaux 的 clean-GT 后加白噪声比较使用 `noisy_iq` 的复高斯噪声约定。
冻结命令、样本预算、逐 SNR 结果和结论见
`doc/litenap_savaux_results_20260724.md`。当前数据未显示相对完整 Savaux 的 SER
提升；K1/K2 应解释为采样率/计算量交换，而不是增益结论。
`-16` 到 `-28 dB` 的逐 1 dB 错误归因见
`doc/litenap_savaux_error_modes_20260724.md`；结果显示主要瓶颈是 modulo-`N/D`
alias bin 本身判错，而不是 alias group 判错。

实验入口按用途大致分为：

- `evaluate_*.py`：性能评估与解码对比。
- `analyze_*.py`：矩阵、候选、噪声协方差和 oracle 上限分析。
- `calibrate_*.py`：阈值标定。
- `compare_*.py`：条件检测器基线对比。
- `diagnose_*.py`：候选失败诊断。
- `plot_*.py`：结果可视化。

新增可部署算法时放入 `system/` 并通过 `system/__init__.py` 和顶层
`__init__.py` 导出；多个实验共用但不属于在线解码器的代码放入
`experiment_support/`；一次性评估、数据扫描和画图入口放入 `experiments/`。

## 真实 capture 汇总表

`evaluate_real_capture_gls` 每次运行都会在输出目录生成 `capture_summary.csv`，
其中包含包检测时刻、检测数、strict sync 数、`dechirp_gt_packet_snr_db`、
dechirp PNR、FFT errors，以及普通 FFT、Savaux 和 GLS 的 SER。传入以下参数可以把多次运行
更新到同一张表中：

```powershell
python -m weak_decoder.os_lora.experiments.evaluate_real_capture_gls `
  <其余参数> `
  --capture-summary-csv data\experiments\capture_summary.csv
```

`dechirp_gt_packet_snr_db` 使用传统单 branch dechirp FFT：先在每个 payload symbol
中读取外部 GT bin 的能量，再按包分别累加 GT-bin 能量与其余 `2^SF - 1` 个 bin 的能量，
计算 `10*log10(sum(E_gt) / sum(E_other))`。`capture_summary.csv` 中的主字段取逐包值的
中位数，完整逐包结果保存在 `packet_snr.csv`。外部 GT 只用于三种解调器完成判决后的
离线评分，不参与同步或 hard-bin 判决。dechirp PNR 是传统单 branch dechirp FFT 的
峰值功率与背景频点中位功率之比，并在全部评分 symbol 上取中位数。
`gls_ser` 默认对应 `gls_crossfit`，可用 `--summary-gls-method gls_offpacket` 改为
固定包外协方差 GLS。

## 架构约束测试

目录依赖由自动测试固定下来：

```powershell
python -m unittest `
  weak_decoder.os_lora.tests.test_architecture `
  weak_decoder.os_lora.tests.test_capture_summary -v
```

测试会检查实验入口之间没有直接导入、`system/` 没有反向依赖、系统模块没有
根目录同名副本，并逐个导入所有现存实验入口。
