# 官方 RFSR-OTA checkpoint 网络缺陷审计交接（2026-08-01）

## 一句话结论

上游公开的 OTA checkpoint 在 7 个不同 payload 的官方 RFSR-OTA IQ 上，残差分支没有产生任何可观测的输入相关修正；除序列首尾各一个 zero-padding 点外，它在 float32 下逐样点输出同一个固定复数。

因此当前 checkpoint 的实际行为是：

```text
RFSR(x) = 官方 polyphase interpolation(x) + 固定复数偏置
```

它不是“整个网络输出为零”，也不是“残差严格为零”。准确说法是：

> 学到的、随输入变化的超分辨率残差已经塌缩；完整输出基本等于插值基线再加一个很小的 DC 偏置。

这已经由官方数据实验直接证明。大 L1 正则是最可疑的成因，但没有公开训练日志，现阶段不能把因果关系写死。

## 证据等级

### 已被直接证明

1. 使用的 checkpoint 是上游跟踪的公开 OTA checkpoint，而不是本地训练模型。
2. 7 个官方 OTA 文件具有 7 个不同的 16-byte payload，每个文件均通过官方 manifest MD5。
3. 原始幅度下，7 包的 residual 数组逐样点完全相同。
4. 每包去均值并缩放到单位 RMS 后，residual 与原始幅度结果仍逐样点完全相同。
5. 每个序列裁掉首尾各一个点后，residual 的复方差在 float32 输出上精确为 0。
6. 固定 residual 主要等于最后一层 bias，但还包含前面各层 bias 传播出的约 `2.43e-9` 固定偏移。
7. checkpoint 的全部 12,802 个参数都被压到很小的量级，和同结构随机初始化相差约四个数量级。

### 代码层面已确认的高风险设计

1. `model0v0lopenaltyhl` 同时启用 Hybrid loss 和额外 L1/Lasso penalty。
2. L1 penalty 的系数硬编码为 `1.0`，并对所有 weight 和 bias 求未归一化的绝对值总和。
3. 优化器还同时使用 `weight_decay=1e-5`。
4. 合并后的梯度再统一裁剪到 norm `1.0`。
5. Hybrid loss 的 FFT 没有使用 `norm="ortho"` 或显式长度归一化，因此频域项的数值尺度依赖序列长度。
6. 官方 OTA loader 不做中心化、标准差/RMS 标准化或幅度配准，损失各项的相对尺度直接依赖数据幅度。

### 尚未被证明

1. 不能仅凭最终 checkpoint 断言塌缩一定由 L1 penalty 单独造成。
2. 尚无该 checkpoint 对应的 loss history、各项梯度或参数轨迹。
3. 不知道发布者训练该 checkpoint 时是否使用了与当前 vendored 文件完全相同的未提交改动。
4. 尚未完成去掉 L1、降低 L1、排除 bias 正则等训练消融。
5. 本实验只证明“残差分支退化”，不直接给出 Native 与 RFSR 的 SER/SNR 性能差。

## checkpoint 身份与结构

公开 OTA checkpoint：

```text
third_party/rfsr/checkpoints/
  model_model0v0lopenaltyhl_bs1_osf4_ds250_lr0.0001_wd1e-05_ota_dsf8.pth

SHA-256:
  a4de5311d70a9c37618a89632d023abe46f847ba18b3d15fb439a513fcd0c398
```

上游记录：

```text
repository: https://github.com/AndreasKuster/RFSuperResolution
base commit: 00c135947f855790f458fdc25ae9533c70d77849
```

模型是 `SimpleComplexCNN0`：

```text
250 kS/s complex IQ
  -> 官方 polyphase interpolation，OSF=4
  -> 1 MS/s interpolated IQ
  -> 4 层 ComplexConv1D residual CNN
       1 -> 16 -> 32 -> 16 -> 1 complex channel
       kernel_size=3, stride=1
  -> interpolation + residual
```

四个 ComplexConv1D 都使用 bias，不是只有最后一层有 bias。最后一层 bias 为：

```text
I = -1.053180312738e-05
Q = +2.215706808784e-05
```

CNN 在插值后采样率上的名义感受野只有 9 个相邻点。它适合做局部波形修正，但不具备从整符号或整包上下文估计 CFO、SFO、长程同步偏移的能力。插值 FIR 本身有更长上下文，但学习残差网络只观察局部插值结果。

## 参数塌缩审计

公开 OTA checkpoint：

| 指标 | 数值 |
| --- | ---: |
| 参数总数 | 12,802 |
| 非零参数数 | 12,802 |
| `sum(abs(parameter))` | 0.1255883723 |
| 参数 L2 norm | 0.0014429725 |
| 最大绝对参数 | 2.8713521e-5 |

同结构 PyTorch 随机初始化，seed 0 到 4：

| 指标 | 随机初始化范围/均值 |
| --- | ---: |
| L1 sum | 592.7783 到 596.2417，均值 594.3125 |
| L2 norm | 6.6919 到 6.8143 |
| 最大绝对参数 | 约 0.407 |

checkpoint 的 L1 sum 约为随机初始化的 `1/4732`。参数并未变成数学上的零，但多层小权重相乘后，输入相关路径在 float32 下已经不可观测；bias 路径仍能产生固定输出。

## 官方数据样本

数据来源：

```text
RFSR-OTA
DOI: 10.21979/N9/C6ABM3
local root: data/rfsr_ota_official/
```

本次只保留 7 个不同 payload，每个 payload 一个 rx gain；没有把同一包的多个 `rxg` 当成独立样本。

| packet | rxg | payload hex | 官方 MD5 |
| ---: | ---: | --- | --- |
| 000000 | 15 | `66dce15fb33deacb5c0362f30e95f52e` | `90726843c208b28ea77c3a061ee95ff3` |
| 000001 | 12 | `6af463bb47d499c7bcae4199142ccb98` | `9899ad3a77540970a412033c6ff25a46` |
| 000002 | 0 | `66d6f02779182272d241ef27d6f49719` | `4745f47928060b82a6300f52c905c85b` |
| 000038 | 24 | `20f971c57a31209b0497f412e90d22ed` | `7b608e5b2c458fd88bb4714e8e6f9dd2` |
| 000072 | 21 | `704eeb48b7eb6bff506c7109bac24b44` | `fe83262f81578fad345d06ca0d7d7632` |
| 000073 | 27 | `70680c9c017003fa8192af80dbb83969` | `fbb11a7e7072cf4e6ead855bc64afe8c` |
| 000093 | 15 | `597935d16f626ffbf8a3218ef33f4284` | `06b25943b86d625a2668c8fc9e7bd7c1` |

目录审计结果：

```text
cfiles             = 7
unique packet ids  = 7
unique payloads    = 7
bytes per file     = 35,475,648
total bytes        = 248,329,536
*.part             = 0
```

这个集合用于 checkpoint 行为审计已经足够，因为问题是函数是否依赖输入；它不是用于估计低 SNR SER 曲线的统计样本，也不能声称是官方 held-out test split。

## residual 实验定义

没有使用 metadata 包边界、CFO 或同步真值。每个完整 IQ 文件都按公开 OTA loader 的机械语义输入：

```python
high = np.memmap(path, dtype="<c8", mode="r")  # complex64 little-endian, 2 MS/s
low = np.asarray(high[::8], dtype=np.complex64) # 250 kS/s
```

主实验使用原始幅度，不做中心化和标准化。诊断实验另外使用：

```python
low_centered = low - mean(low)
low_normalized = low_centered / rms(low_centered)
```

网络运行设置：

```text
device                 = CUDA
chunk_input_samples    = 65,536
overlap_input_samples  = 68
upsample_factor        = 4
residual source        = model.residual_vec
```

直接读取 `model.residual_vec`，避免用两个大 float32 输出相减来估计极小方差。为验证提取逻辑，额外比较了：

```text
direct residual vs enhance(x) - interpolate(x)
max absolute difference = 1.3642421e-11
RMS difference          = 2.1176379e-12
```

统计量定义如下。这里的 variance 是网络 residual 的复方差，不是 OTA 输入方差，也不是完整增强输出的方差：

```text
mu              = mean(r)
residual RMS    = sqrt(mean(|r|^2))
complex var     = mean(|r - mu|^2)
centered RMS    = sqrt(complex var)
```

## 官方 7 包实验结果

7 包原始输入 RMS 覆盖：

```text
2.4769509e-4 to 5.6263438e-4
```

每一包都得到完全相同的 residual 统计：

```text
residual mean
  -1.053378127765e-05 + j2.215566200961e-05

residual RMS
  2.453230333844e-05

complex variance
  1.142522173313e-24

centered residual RMS
  1.068888288510e-12

max |r - mean(r)|
  1.500516509905e-09
```

更强的逐样点检查：

```text
不同官方 packet 相对 packet 000000：
  max absolute residual difference = 0
  different float32 samples         = 0

原始幅度相对 centered + unit-RMS 输入：
  max absolute residual difference = 0
  different float32 samples         = 0
```

非零方差只来自两个点：

```text
index 0
index 2,217,227
```

也就是整个输出序列的第一个和最后一个样点。它们是卷积 zero-padding 的固定边界效应，不随 packet、payload、rxg 或输入尺度变化。首尾各裁掉一个样点后：

```text
complex variance  = 0
centered RMS      = 0
max centered      = 0
```

上述“0”指当前 float32 推理结果的逐位相等，不是对无限精度实数网络做数学恒等证明。

固定 residual 与最后一层 bias 的差为：

```text
abs(mean residual - final bias) = 2.4269599e-09
```

因此不能再表述为“只剩最后一层 bias”。更准确的是：

> 最后一层 bias 占主体；前面各层 bias 经过极小权重传播，再贡献一个固定的约 2.43e-9 偏移。

## 对现有评估结论的影响

任何使用这个 checkpoint 的 `official_ota_rfsr` 路径，实质上都非常接近：

```text
official polyphase interpolation + fixed DC
```

所以后续必须把“纯官方插值”提升为强制基线，而不能只比较：

```text
native_1msps vs official_ota_rfsr
```

如果 RFSR 与插值存在同步、Savaux 或 SER 差异，应先检查固定 DC、数值精度、裁剪窗口或同步阈值，而不能直接归因于网络恢复了高频信息。

这一发现不否定 RFSR 架构经过正确训练后可能有用；它否定的是“当前公开 OTA checkpoint 已经提供有效输入相关超分辨率修正”这一前提。

## loss 与正则代码审计

公开 checkpoint 名字中的：

```text
model0v0lopenaltyhl
```

会触发两条分支：

```text
hl          -> HybridDenoiserLoss
lopenalty   -> extra L1 parameter penalty
```

数据项为：

```text
L_data = 1.0 * L_time + 0.1 * L_freq

L_time = mean absolute error on I/Q samples
L_freq = mean absolute error on magnitudes of unnormalized full FFT
```

训练总 loss 为：

```text
L_total = L_data + 1.0 * sum(abs(all model parameters))
```

同时优化器文件名记录：

```text
learning rate = 1e-4
weight decay  = 1e-5
```

随后执行：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

这里有三个明显的数值风险。

第一，`sum(abs(parameter))` 没有除以参数数 12,802。随机初始化时这一项约为 594，而不是一个自然的小修正项。实际数据 loss 的历史没有公开，因此不能声称确切倍率，但这个量级足以构成严重塌缩风险。

第二，L1 对全部参数生效，包括每层 bias。只要插值 shortcut 已经能把数据 loss 降到一定程度，继续压缩 residual CNN 往往比学习细小修正更便宜。

第三，全部参数非零时，单独的 L1 subgradient norm 约为：

```text
sqrt(12,802) ~= 113
```

再把联合梯度裁到 1.0，可能使更新方向长期由 L1 的符号梯度主导。这个机制与最终所有参数都缩到 `3e-5` 以下的现象一致，但仍需训练消融验证因果。

Hybrid loss 也存在尺度问题：`torch.fft.fft` 默认不归一化，频域 magnitude MAE 的绝对数值随序列长度和输入幅度变化。固定 `lambda_freq=0.1` 不能保证在不同 packet 长度、幅度或裁剪方式下具有相同含义。

## 关于“中心化、RMS 标准化能否减小正则”的纠正

中心化或 RMS 标准化不会直接改变：

```text
sum(abs(model parameters))
```

因为正则只依赖参数，不依赖当前 batch。

标准化改变的是数据 loss 和数据梯度的尺度，从而改变它们相对于固定正则梯度的权重。如果输入和标签的幅度很小，数据梯度可能更容易被固定强度的参数正则压倒；把成对数据放到一致、可控的尺度，可能改善这个比例，但不能自动修复一个已经塌缩的 checkpoint。

本次验证中，即使把每包输入中心化并缩放到单位 RMS，残差仍与原始输入逐位相同。这说明当前 checkpoint 已经失去输入依赖，不是简单地在推理时补一个标准化就能恢复。

另外，当前官方 OTA loader 明确没有做中心化或 RMS 标准化。若后续训练要加入，必须同时规定：

1. x 和 y 使用哪个共同中心与尺度；
2. 推理后如何恢复物理幅度；
3. guard/noise 是否参与尺度估计；
4. 是否泄漏目标 y 的信息到推理归一化；
5. 训练和在线评估是否严格执行同一规则。

## 下一步建议：先做最小训练消融

不要立刻换大网络。先在同一数据、同一初始化和同一 batch 顺序下比较：

| 实验 | L1 penalty | bias penalty | FFT normalization | 目的 |
| --- | --- | --- | --- | --- |
| A | 当前 `sum`, lambda=1 | 是 | 无 | 复现塌缩基线 |
| B | 关闭 | 否 | 无 | 验证 L1 是否主因 |
| C | `mean(abs(theta))` | 否 | 无 | 消除参数数量依赖 |
| D | 小 lambda 网格 | 否 | 无 | 找不压死 residual 的范围 |
| E | 与最佳 B/C/D 相同 | 否 | `norm="ortho"` | 验证频域尺度问题 |

每个 batch 至少记录：

```text
L_time
L_freq
L1 penalty before weighting
weighted L1 penalty
data-gradient norm
regularization-gradient norm
combined gradient norm before clipping
clip coefficient
parameter L1/L2/max
residual mean
residual variance after removing boundaries
residual RMS / interpolation RMS
```

建议加入自动失败条件：如果多个不同输入的 residual 在去边界后逐样点相同，或 residual centered RMS 低于预设阈值，则标记 checkpoint collapsed，不继续用下游 SER 掩盖问题。

训练目标还应明确区分：

```text
received 250 kS/s -> received 1 MS/s
received 250 kS/s -> ideal reference 1 MS/s
```

前者主要是采样重建；后者同时要求网络消除通道、CFO/SFO/STO、增益和噪声。四层局部 CNN 无法可靠承担整包同步补偿，不能把标签未对齐造成的 loss 全部归因于“超分辨率能力不足”。

## 复现环境与关键入口

本次 Windows 环境：

```text
repository:
  D:\Desktop\proj\lora-rfsr-savaux

Python/PyTorch:
  D:\mysoft2\miniconda3\envs\MAML\python.exe
  torch 2.5.1
  CUDA available = True
```

关键实现：

```text
weak_decoder/rf_super_resolution/frontend.py
third_party/rfsr/rfsr/nn/nn.py
third_party/rfsr/rfsr/nn/dataset.py
third_party/rfsr/rfsr/nn/ota_dataset.py
```

重点代码位置：

```text
SimpleComplexCNN0.forward:
  interpolation + residual，以及 residual_vec

HybridDenoiserLoss:
  time I/Q L1 + full FFT magnitude L1

training loop:
  lopenalty -> l1_lambda = 1.0 -> sum(abs(all parameters))
```

后续复现 checkpoint 行为时应继续直接读取 `model.residual_vec`，并复用 frontend 的 chunk overlap/crop 语义。用 `enhance - interpolate` 可以做 parity check，但不适合把 float32 相消误差当作真实 residual 方差。

## 交接时的仓库与数据状态

```text
official checkpoint:
  已保留，且是 checkpoints/ 中唯一 OTA checkpoint

official data:
  data/rfsr_ota_official/ota/ 下 7 个 cfile
  每个 payload 一个 rxg
  全部官方 MD5 通过
  无 .part 文件

local non-official OTA checkpoint:
  已按用户要求删除
```

官方 IQ 数据体积较大且通常被 gitignore；不要把这些 `.cfile` 提交到 Git。handoff 中的 MD5、payload 和 packet id 用于在其他服务器重新抽取同一审计集合。

