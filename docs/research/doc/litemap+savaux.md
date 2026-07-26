> 命名说明：本文引用的论文方法正式名称是 **LiteNap**。为兼容已有引用，
> 文件名仍保留为 `litemap+savaux.md`。

对，我这次完全理解你的意思了。你要做的不是“把 Savaux branch 再拆碎以后直接乱加”，而是：

[
\boxed{
\text{每个 Savaux 过采样相位都生成一份 LiteNap 欠采样观测，}
\quad
\text{先用 OSR 维度强化每份欠采样副本，再联合解码。}
}
]

这个方案可以正式设计成一个 **Savaux-enhanced sub-Nyquist LoRa receiver**。Savaux 本来就是把 OSR 信号拆成若干条 Nyquist-rate 相位支路，分别解调后合并；LiteNap 则证明单条低于 Nyquist 的 LoRa 观测虽然发生频谱混叠，但仍可借助 chirp 回卷处的 phase jitter / leakage timing 恢复完整符号。([ResearchGate][1])

---

# 一、整体结构

设：

[
N=2^{SF},\qquad R=OSR,
]

接收机得到一个长度为 (RN) 的过采样符号：

[
z[0],z[1],\ldots,z[RN-1].
]

Savaux 首先按照过采样相位拆成 (R) 条 Nyquist branch：

[
z_q[p]=z[Rp+q],
\qquad
q=0,\ldots,R-1.
]

现在引入 LiteNap 欠采样因子：

[
D\in{2,4,8},
\qquad
M=\frac ND.
]

对每一条 Savaux branch，再按照 downsampling phase (d) 构造：

[
\boxed{
z_{q,d}[m]
==========

z!\left[R(Dm+d)+q\right]
}
]

其中：

[
q=0,\ldots,R-1,
\qquad
d=0,\ldots,D-1,
\qquad
m=0,\ldots,M-1.
]

于是：

* 固定 (q,d)：一条采样率为 (BW/D) 的 LiteNap 欠采样 chirp；
* 固定 (d)，改变 (q)：同一份欠采样 chirp 的 (R) 个 Savaux 过采样相位副本；
* 改变 (d)：得到不同 downsampling phase 的欠采样视图。

结构就是：

[
\boxed{
D\text{ 个欠采样视图}
\times
R\text{ 个 Savaux 相位副本}
}
]

---

# 二、每条欠采样 branch 怎么解调

设完整候选 LoRa 符号为 (k)。

对 (z_{q,d}[m]) 使用其**真实采样时刻对应的 downchirp**进行 dechirp：

[
u_{q,d}[m]
==========

z_{q,d}[m],
c^*!\left(Dm+d+\frac qR\right).
]

然后进行一个长度仅为

[
M=\frac ND
]

的 FFT：

[
Y_{q,d}[r]
==========

\sum_{m=0}^{M-1}
u_{q,d}[m]
e^{-j2\pi rm/M}.
]

欠采样以后，完整符号 (k) 只会落在 alias bin：

[
\boxed{
r(k)=k\bmod M
}
]

因此一个 alias bin (r) 对应 (D) 个完整候选：

[
\mathcal K(r)
=============

\left{
r,,
r+M,,
r+2M,,
\ldots,,
r+(D-1)M
\right}.
]

这正是 LiteNap 需要解决的 ambiguity。LiteNap 的原始做法是利用 phase jitter 的时间位置判断符号属于哪个频率 chunk。([Department of Computing][2])

---

# 三、Savaux 如何强化一份 LiteNap 欠采样副本

理想同步情况下，对真实符号 (k)，第 (q,d) 条 branch 在 alias bin (r(k)) 上的输出近似为：

[
Y_{q,d}[r(k)]
\approx
Mh
\exp\left[
j\frac{2\pi k}{N}
\left(
d+\frac qR
\right)
\right]
+
\eta_{q,d}.
]

这里最关键的是：

[
\boxed{
不同 q 支路不是同相的，
但相位关系由候选 k 完全确定。
}
]

对于固定的 downsampling phase (d)，将 (R) 条 Savaux 输出堆起来：

[
\mathbf y_d(r)
==============

\begin{bmatrix}
Y_{0,d}[r]\
Y_{1,d}[r]\
\vdots\
Y_{R-1,d}[r]
\end{bmatrix}.
]

候选 (k) 对应的 Savaux steering vector 为：

[
\mathbf a_d(k)
==============

\begin{bmatrix}
e^{j\frac{2\pi k}{N}d}\
e^{j\frac{2\pi k}{N}(d+\frac1R)}\
\vdots\
e^{j\frac{2\pi k}{N}(d+\frac{R-1}{R})}
\end{bmatrix}.
]

于是，对每一个完整候选 (k)，进行候选相关的相干合并：

[
\boxed{
\widehat h_d(k)
===============

\frac{
\mathbf a_d^H(k)
\mathbf C_d^{-1}
\mathbf y_d(r(k))
}{
\mathbf a_d^H(k)
\mathbf C_d^{-1}
\mathbf a_d(k)
}
}
]

其中 (\mathbf C_d) 是这 (R) 条 Savaux 支路的噪声协方差。

这一步的含义就是：

[
\boxed{
\text{先用 }R\text{ 条 Savaux 支路，
把第 }d\text{ 份 LiteNap 欠采样副本强化。}
}
]

如果是 AWGN，可以直接取：

[
\mathbf C_d=\sigma^2\mathbf I,
]

那么就是按照理论相位旋转后相干叠加。

---

# 四、再融合多个 downsampling phase

经过上一步，每个 downsampling phase 都会为候选 (k) 给出一个复信道估计：

[
\widehat h_0(k),
\widehat h_1(k),
\ldots,
\widehat h_{K-1}(k).
]

这里 (K\leq D)，表示实际使用多少个 downsampling phase。

对于正确候选 (k_0)，所有结果经过候选相位补偿以后应当指向同一个复信道：

[
\widehat h_d(k_0)\approx h.
]

对于错误的 alias 候选：

[
k'=k_0+cM,
]

不同 (d) 和 (q) branch 上会留下不一致的确定性相位，因此不能完全相干。

最终可以再做一次 GLS：

[
\widehat{\mathbf h}(k)
======================

\begin{bmatrix}
\widehat h_{d_1}(k)\
\vdots\
\widehat h_{d_K}(k)
\end{bmatrix}.
]

评分为：

[
\boxed{
T_{\mathrm{spec}}(k)
====================

\frac{
\left|
\mathbf 1^H
\mathbf C_h^{-1}
\widehat{\mathbf h}(k)
\right|^2
}{
\mathbf 1^H
\mathbf C_h^{-1}
\mathbf 1
}
}
]

这实际上构成了一个两层接收机：

[
\boxed{
\begin{aligned}
&\text{第一层：Savaux 相位维度强化每份欠采样副本；}\
&\text{第二层：多个欠采样 phase 联合消除 alias ambiguity。}
\end{aligned}
}
]

也可以不分两层，直接把所有 (RK) 条输出堆成：

[
\mathbf y(r)
============

\begin{bmatrix}
Y_{0,d_1}[r]\
\vdots\
Y_{R-1,d_K}[r]
\end{bmatrix}
]

然后一次性计算：

[
\boxed{
T_{\mathrm{joint}}(k)
=====================

\frac{
\left|
\mathbf a^H(k)
\mathbf C^{-1}
\mathbf y(r(k))
\right|^2
}{
\mathbf a^H(k)
\mathbf C^{-1}
\mathbf a(k)
}
}
]

其中：

[
a_{q,d}(k)
==========

\exp\left[
j\frac{2\pi k}{N}
\left(
d+\frac qR
\right)
\right].
]

这个公式非常像阵列信号处理：

* (k)：待检测的“方向”；
* (q,d)：虚拟天线位置；
* (\mathbf a(k))：候选符号的 steering vector；
* (Y_{q,d})：每根虚拟天线上的复数观测。

所以你可以把它理解为：

[
\boxed{
\text{alias-domain virtual MIMO detector}
}
]

---

# 五、LiteNap fingerprint 怎么加入

上面的 steering-vector fusion 本身已经能利用不同采样相位的相位差来区分 alias 候选。

但在 (K) 很小时，alias 候选之间的 steering vector 可能高度相关。这时再加入 LiteNap 的 phase-jitter timing。

对于候选 (k)，先在每个 downsampling phase 上完成 Savaux 时间域相干合并：

[
g_d^{(k)}[m]
============

\sum_{q=0}^{R-1}
w_{q,d}^*(k)
z_{q,d}[m]
s_k^*!\left(Dm+d+\frac qR\right).
]

正确候选下：

[
g_d^{(k)}[m]
\approx
h e^{j\phi_{\mathrm{jit}}[m]}
+
v[m].
]

然后根据候选 (k) 预测 chirp 的回卷位置：

[
\tau_k
======

\frac{N-k}{D}
]

或者使用考虑 CFO/STO 后的校正位置。

在 (\tau_k) 前后做两段复数均值拟合：

[
J_d(k)
======

\min_{h_1,h_2}
\left[
\sum_{m<\tau_k}
|g_d^{(k)}[m]-h_1|^2
+
\sum_{m\geq\tau_k}
|g_d^{(k)}[m]-h_2|^2
\right].
]

正确候选的 phase jump 应该出现在预测位置附近，因此 (J_d(k)) 较小。

最终评分：

[
\boxed{
T(k)
====

## T_{\mathrm{joint}}(k)

\lambda
\sum_{d\in\mathcal D}
J_d(k)
}
]

或者写成更规范的联合对数似然：

[
\log p(\mathbf y\mid k)
+
\log p(\widehat{\tau}_{\mathrm{jit}}\mid k).
]

LiteNap 的 phase-based 方法本来就是先取得 alias frequency，再用整段 chirp 中累积的相位变化提取 jitter timing；而且论文指出这种方法比短窗频谱泄漏检测更耐噪声。([Department of Computing][2])

---

# 六、你所说的“增强倍数”应该怎么算

一条 LiteNap branch 只有：

[
\frac ND
]

个样点。

如果使用：

* (R) 个 Savaux 相位；
* (K) 个 downsampling phase；

总共使用：

[
RK\frac ND
]

个样点。

因此，相对于**单独一条 LiteNap 欠采样 branch**，最大相干增强为：

[
\boxed{
10\log_{10}(RK)\ {\rm dB}
}
]

但相对于一条正常 Nyquist branch，其理论处理增益为：

[
\boxed{
10\log_{10}\frac{RK}{D}\ {\rm dB}
}
]

以你的典型参数：

[
R=4,\qquad D=4
]

为例。

### 只使用一个 downsampling phase

[
K=1
]

则总样点数：

[
4\times\frac N4=N.
]

相对于单条 LiteNap：

[
10\log_{10}4=6.02\ {\rm dB}.
]

相对于标准 Nyquist 接收机：

[
10\log_{10}\frac{4}{4}=0\ {\rm dB}.
]

也就是说：

[
\boxed{
4\text{ 条 Savaux 支路理论上正好补回 }D=4
\text{ 欠采样损失。}
}
]

这正是你的核心直觉。

### 使用两个 downsampling phase

[
K=2
]

相对于标准 Nyquist：

[
10\log_{10}\frac{4\times2}{4}
=============================

3.01\ {\rm dB}.
]

### 使用全部四个 phase

[
K=D=4
]

相对于标准 Nyquist：

[
10\log_{10}4
============

6.02\ {\rm dB}.
]

这时已经使用了完整的 (4N) 个原始样点，所以性能上限就是完整 Savaux。

---

# 七、最适合先实现的版本

我建议第一版直接设：

[
R=4,\qquad D=4.
]

依次测试下面四个接收机。

## Baseline A：单 LiteNap

[
R=1,\qquad K=1.
]

每个符号只使用：

[
N/4
]

个样点，验证标准 LiteNap alias 解码。

## Baseline B：Savaux-enhanced LiteNap

[
R=4,\qquad K=1.
]

选一个 downsampling phase，比如：

[
\mathcal D={0}.
]

使用四条 Savaux 相位 branch 强化同一条 (1/4)-rate LiteNap 副本。

这是最直接验证你想法的版本：

[
\boxed{
\text{Savaux 能否把 LiteNap 推到更低 SNR。}
}
]

## Proposed C：双 downsampling-view

[
R=4,\qquad K=2.
]

例如：

[
\mathcal D={0,1}.
]

总共八条长度 (N/4) 的 FFT branch。

先用 joint steering score 区分四个 alias 候选，再使用 phase-jitter score 辅助。

这一版理论上相对 Nyquist 有约 3 dB 样点能量增益，但计算量只有：

[
8\times\frac N4\log_2\frac N4
]

而不是四个完整 (N)-point FFT。

## Ceiling D：完整 (R\times D)

[
R=4,\qquad K=4.
]

共十六条 (N/4)-point FFT。

这一版本主要用于验证实现正确性：

[
\boxed{
T_{\mathrm{joint}}
\text{ 应当逐渐逼近完整 Savaux。}
}
]

它本身不能作为主要贡献，因为所有 (d) 都取满以后，本质上只是对完整过采样数据进行了 polyphase 重排。

---

# 八、可以加一个有数学指导的 branch 选择

如果不想把 (D) 个 downsampling phase 全取满，可以选择集合：

[
\mathcal D\subseteq{0,\ldots,D-1}.
]

不同 alias chunk 的 steering vector 相关性为：

[
\mu_c(\mathcal D)
=================

\frac{1}{R|\mathcal D|}
\left|
\sum_{d\in\mathcal D}
\sum_{q=0}^{R-1}
e^{j2\pi c(d+q/R)/D}
\right|,
\qquad c=1,\ldots,D-1.
]

然后选择：

[
\boxed{
\mathcal D^\star
================

\arg\min_{\mathcal D:,|\mathcal D|=K}
\max_{c=1,\ldots,D-1}
\mu_c(\mathcal D)
}
]

这就非常符合你之前一直想要的东西：

[
\boxed{
\text{不是乱搜 NUS pattern，}
\quad
\text{而是根据 LoRa alias candidate 的可分性选择采样 phase。}
}
]

对于：

[
R=4,D=4,K=2,
]

我粗算的结果是：

* ({0,1})、({0,3})、({1,2})、({2,3}) 的最坏相关性约为 (0.641)；
* ({0,2})、({1,3}) 的最坏相关性约为 (0.653)。

差异不算巨大，但已经说明 phase 选择存在明确的阵列流形优化目标。

---

# 九、这项工作的正确定位

最好的论文问题不是：

> 我们把 Savaux 拆成 (R\times D) 条 branch，因此获得了 (R\times D) 倍增益。

而是：

[
\boxed{
\text{能否利用过采样相位多样性，
补偿 sub-Nyquist LoRa 的处理增益损失，
并增强 alias ambiguity 的判别能力？}
}
]

可以把三种已有结构统一起来：

[
\begin{aligned}
D=1
&\Rightarrow \text{Savaux；}\
R=1,\ K=1
&\Rightarrow \text{LiteNap；}\
R>1,\ K<D
&\Rightarrow \text{你的 Savaux-enhanced LiteNap。}
\end{aligned}
]

我认为最有研究价值的是：

[
\boxed{
R=4,\quad D=4,\quad K=1\text{ 或 }2
}
]

因为这时你没有把全部原始样点重新用一遍，而是在真实丢弃大量样点的情况下，研究：

* OSR 多相位能否补偿欠采样损失；
* alias steering consistency 能否辅助 LiteNap；
* phase fingerprint 在 Savaux 强化后能否进入更低 SNR；
* 能否以少于完整 Savaux 的计算量逼近其性能。

最后有一个现实提醒：LiteNap 的强 fingerprint 来自 COTS LoRa 调制硬件产生的 phase jitter；论文也通过前导码校准不同节点的固定偏移。纯 GNU Radio/USRP 理想生成的 chirp 不一定具有同样特征。([Department of Computing][2])

因此你的实现最好同时保留两条判决路径：

[
\boxed{
\text{确定性的跨 branch steering 判决}
+
\text{COTS 数据上可用的 LiteNap fingerprint 判决}
}
]

即使硬件 fingerprint 不稳定，前一条仍然可以独立工作。

[1]: https://www.researchgate.net/publication/354792082_A_Low-Complexity_Demodulation_for_Oversampled_LoRa_Signal?utm_source=chatgpt.com "(PDF) A Low-Complexity Demodulation for Oversampled LoRa Signal"
[2]: https://web.comp.polyu.edu.hk/csyqzheng/papers/LiteNap-INFOCOM20.pdf "LiteNap-xia.pdf"

---

# 十、实现与实测结论（2026-07-24）

方案已实现为：

- 在线算法：`weak_decoder/os_lora/system/litenap_savaux.py`
- 对比入口：`weak_decoder/os_lora/experiments/evaluate_litenap_savaux.py`
- 冻结结果：`weak_decoder/os_lora/doc/litenap_savaux_results_20260724.md`
- 原始表格：`data/experiments/litenap_savaux_clean_gt_20260724/`

实现使用真实的 `N/D` 欠采样观测，不是从完整 FFT 中事后挑 bin。`D=4` 时，
K1、K2 和完整 Savaux 分别使用 1/4、1/2 和全部 ADC 样点；K4 与原始 Savaux
频谱的最大数值误差为 `1.922192e-06`。

在 17 个 clean 同步包、833 个 payload symbols、3 个随机种子和
`-22/-24/-26/-28 dB` added-SNR 白噪声条件下，原始 Savaux 的 SER 分别为
`1.60%/11.20%/33.17%/60.02%`，K2 为
`22.05%/47.58%/72.51%/87.76%`。因此当前实验没有得到相对 Savaux 的解码提升。

无训练 phase-jump 重排也没有带来稳定修正。它只是对本文第五节设想的工程消融，
不是 LiteNap 论文中经过前导码标定的发射机硬件指纹完整复现。详细的
fix/break、数据协议和复现命令见上述冻结结果文档。

进一步的 `1 dB` 细粒度错误归因表明，K1 的全部错误中 `95.77%`、K2 中
`98.12%` 属于 alias bin 本身判错；group 判错仅占 `4.23%` 和 `1.88%`。
因此即使使用完美 fingerprint 修复全部 group 错误，也只能带来很小的 oracle
增益。完整分析见
`weak_decoder/os_lora/doc/litenap_savaux_error_modes_20260724.md`。
