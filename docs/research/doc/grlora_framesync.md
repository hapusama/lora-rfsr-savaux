# gr-lora_sdr 同步策略

## 0. 基本信号模型

gr-lora_sdr 不是把 CFO、STO、SFO 放进一个大优化问题里联合求解，而是在 `frame_sync` 里按以下顺序逐步消掉：

**粗定时 → CFO_frac → STO_frac → CFO_int → SFO → 重新估 STO_frac → payload 采样插删**

Hi²LoRa 里把 CFO/STO/SFO 作为硬件 offset 特征建模：

- **CFO** 造成频率漂移
- **STO** 同时造成频率和相位漂移
- **SFO** 让 STO 随时间变化

这个物理认识和 gr-lora_sdr 的处理是一致的；区别是 gr-lora_sdr 更工程化，直接调整相位旋转和采样点。

---

设 LoRa 一个 symbol 有：

$$N = 2^{\text{SF}}$$

个 chip。标准 upchirp 可以写成：

$$u[n] = \exp\left\{ j2\pi \left( \frac{n^2}{2N} + \frac{n}{2} \right) \right\}, \qquad 0 \le n < N$$

接收端存在三个主要 offset：

$$\kappa = \kappa_I + \kappa_F$$

表示 CFO，单位是 frequency bin；其中 $\kappa_I$ 是整数 CFO，$\kappa_F$ 是小数 CFO。

$$\mu = \mu_I + \mu_F$$

表示 STO，单位是 chip；其中 $\mu_I$ 是整数 STO，$\mu_F$ 是小数 STO。

$$\epsilon$$

表示采样频率相对误差，即 SFO。

接收信号可以粗略写成：

$$r[n] \approx h \cdot u\big((1-\epsilon)n - \mu\big) \cdot \exp\left(j2\pi\frac{\kappa n}{N}\right) + w[n]$$

这个式子里：

- CFO 是额外乘上的线性相位旋转
- STO 是时间平移
- SFO 是时间轴缩放

如果先不考虑 SFO，即 $\epsilon = 0$，dechirp 后：

$$z[n] = r[n]u^*[n] \approx h \exp\left\{ j2\pi \left[ \frac{(\kappa-\mu)n}{N} + \phi_{\text{STO}} \right] \right\}$$

所以 LoRa 里有一个很烦的点：

$$\text{dechirp 后的峰位置} \approx \kappa - \mu$$

也就是说，FFT peak 的偏移不是纯 CFO，也不是纯 STO，而是两者混在一起。

---

## 1. 粗同步：先找到 preamble 大致起点

gr-lora_sdr 先通过 preamble upchirp 找一个粗窗口位置。这个阶段主要得到一个粗略的 $\hat{k}$，代码里叫 `k_hat`。它来自 preamble upchirp 的频域峰位置。

数学上可以理解为：

$$\hat{k} \approx \kappa_I + \mu_I$$

或者在不同 chirp 符号约定下是类似的整数 bin 组合。关键是：preamble peak 给的是 CFO/STO 混合量，不足以单独确定 CFO。所以后面还要借助 downchirp/SFD 去拆。

既然当前 peak 显示我偏了 $\hat{k}_{\text{coarse}}$，那我把窗口起点往反方向挪 $N - \hat{k}_{\text{coarse}}$ 个 chip，使得新的窗口重新落到 chirp 起点。

---

## 2. 小数 CFO 估计：利用连续 preamble 的相位旋转

gr-lora_sdr 当前代码里调用的是 `estimate_CFO_frac_Bernier()`，在 SYNC 阶段估计 `m_cfo_frac`，随后构造 CFO 小数补偿向量。源码中可以看到，它对连续 preamble 的同一频率 bin 做复数相关，然后用相位差估计 CFO_frac；随后构造 `CFO_frac_correc[n] = exp(-j2π m_cfo_frac n/N)`。

数学上，假设第 $q$ 个 preamble dechirp+FFT 后，目标 bin 的复数值是：

$$P_q[k^\star] \approx A e^{j(\theta_0 + 2\pi q\kappa_F)}$$

连续两个 preamble 的相位差为：

$$P_q[k^\star] P_{q+1}^*[k^\star] \approx A^2 e^{-j2\pi\kappa_F}$$

所以：

$$\hat{\kappa}_F = -\frac{1}{2\pi} \angle \left( \sum_q P_q[k^\star] P_{q+1}^*[k^\star] \right)$$

估出小数 CFO 后，乘补偿向量：

$$c_{\text{CFO},F}[n] = \exp\left(-j2\pi\frac{\hat{\kappa}_F n}{N}\right)$$

补偿后：

$$r_F[n] = r[n] \cdot c_{\text{CFO},F}[n]$$

这个动作的物理意义是：CFO 小数部分会让每个 chip 的相位持续旋转；乘一个反向线性相位，把这个旋转抵消掉。

---

## 3. 小数 STO 估计：CFO_frac 去掉后，看剩余 peak 偏移

小数 CFO 补偿后，preamble dechirp 得到的 residual frequency peak 主要来自小数 STO。代码里的 `estimate_STO_frac()` 会对多个 upchirp 分别 dechirp，然后做长度 $2N$ 的 FFT，并把多个 preamble 的 FFT power 累加；最后通过三点插值获得一个小数 peak 位置。

### 为什么用 2N 长度 FFT？

FFT 的"长度"不是说原始信号真的有 $2N$ 个有效采样点，而是说 DFT 计算时取多少个频率采样点。

LoRa 一个 symbol 本来有 $N$ 个 chip。对一个 upchirp dechirp 之后，我们得到一段长度为 $N$ 的序列 $x[0], x[1], \ldots, x[N-1]$。

正常做 N-point FFT：

$$X_N[k] = \sum_{n=0}^{N-1} x[n] e^{-j2\pi kn/N}, \quad k = 0, 1, \ldots, N-1$$

这时 FFT 的频率间隔是 $\Delta f_{\text{FFT}} = 1/N$。

现在的问题是：STO_frac / CFO_frac 会让峰落在两个整数 bin 中间。比如理想情况下 dechirp 后 tone 应该落在 $k=0$，但因为小数 STO，它可能实际落在 $k=0.37$。

如果只做 N-point FFT，FFT 只会看 $0, 1, 2, 3, \ldots, N-1$ 这些点。它看不到 $0.37$ 这个位置，只能看到 0 号 bin 最大、1 号 bin 次大。这样估计就比较粗。

所以 gr-lora_sdr 做长度 $2N$ 的 FFT。数学上等价于先补零：

$$x[0], \ldots, x[N-1], \underbrace{0, \ldots, 0}_{N \text{ 个}}$$

然后做：

$$X_{2N}[k] = \sum_{n=0}^{N-1} x[n] e^{-j2\pi kn/(2N)}, \quad k = 0, 1, \ldots, 2N-1$$

注意求和还是只到 $N-1$，因为后面的 $N$ 个点是 $0$。

这时频率采样间隔变成 $\Delta f_{\text{FFT}} = 1/(2N)$。也就是说，频率网格变细了一倍。原来只能看 $0, 1, 2, 3, \ldots$，现在能看 $0, 1/2, 1, 3/2, 2, 5/2, \ldots$。

所以长度 $2N$ 的 FFT 本质上是在做 **zero-padding frequency interpolation**，不是说物理上多出来了一倍采样点。

### 补零后的 0.5 bin 是真实值还是插值？

$0.5$、$1.5$ 这些 bin 位置的值是真实计算出来的频谱采样点，但它们不是新的物理信息。

更准确地说：它是真实的 DTFT 曲线采样点；但这条 DTFT 曲线本身已经由原来的 $N$ 个采样点决定了，所以它也可以理解成由原始数据"插值"出来的。

原来 $N$ 点 FFT：

$$X_N[k] = \sum_{n=0}^{N-1} x[n] e^{-j2\pi kn/N}$$

它只在这些频率上看：$k = 0, 1, 2, \ldots, N-1$

如果补零到 $2N$：

$$X_{2N}[k] = \sum_{n=0}^{N-1} x[n] e^{-j2\pi kn/(2N)}$$

注意求和还是只到 $N-1$，因为后面补的都是 $0$。

这时 $k = 0, 1, 2, 3, \ldots, 2N-1$，对应到原来的 bin 坐标就是 $0, 0.5, 1, 1.5, 2, \ldots$

所以：

$$X_{2N}[2m] = X_N[m]$$

也就是说，偶数点就是原来的整数 bin。而奇数点：

$$X_{2N}[2m+1] = \sum_{n=0}^{N-1} x[n] e^{-j2\pi (m+0.5)n/N}$$

它就是在 $m+0.5$ 这个频率位置上，拿原始信号和对应复指数做相关。这个值是实打实算出来的，不是随便画出来的。

但它又可以说是"插值出来的"。因为有限长度序列的连续频谱是：

$$X(e^{j\omega}) = \sum_{n=0}^{N-1} x[n] e^{-j\omega n}$$

这是一条连续的频谱曲线。$N$ 点 FFT 只是采样这条曲线上的 $N$ 个点；$2N$ 点 FFT 是采样这条曲线上的 $2N$ 个点；$4N$ 点 FFT 是采样这条曲线上的 $4N$ 个点。

所以补零后的 $0.5$ bin、$1.5$ bin 是：**同一条频谱曲线上更密的采样点。**

它不是简单地取相邻两个 FFT bin 的平均值，而是严格由原始序列通过复指数相关算出来的；数学上等价于对原来频谱进行一种 sinc/Dirichlet kernel 插值。

### 回到 STO_frac 估计

可以写成：

$$D_q[k] = \mathrm{FFT}_{2N}\left\{ r_F[qN+n] \cdot u^*[n] \right\}$$

多个 preamble 累加能量：

$$E[k] = \sum_q |D_q[k]|^2$$

找到峰值：

$$k_0 = \arg\max_k E[k]$$

再做三点插值：

$$\hat{k} = k_0 + k_a$$

因为 FFT 长度是 $2N$，所以小数 STO 估计写成代码形式就是：

$$\hat{\mu}_F = \mathrm{wrap}_{[-0.5,0.5]}\left( \mathrm{frac}\left( \frac{\hat{k}}{2} \right) \right)$$

STO_frac 的"补偿"不是简单乘相位，而是重新选择采样点。设过采样倍数是 $L$，那么从原始过采样流里取样时，代码做的是类似：

$$\tilde{r}[i] = r_{\text{raw}}\left[ L(N - \hat{k}_{\text{coarse}} + i) - \mathrm{round}(L\hat{\mu}_F) \right]$$

源码里对应的是 `corr_preamb[i] = preamble_raw_up[m_os_factor * (m_number_of_bins - k_hat + i) - round(m_os_factor * m_sto_frac)]`，也就是把小数 STO 转换成过采样点偏移，再重新抽取对齐后的 chip。

**单位链条理解：**

$$\text{chip index} \xrightarrow{\times L} \text{raw sample index}$$

其中：

$$L = \frac{\text{raw samples}}{\text{chip}}$$

也就是每个 LoRa chip 对应多少个原始采样点。乘进去之前，括号里的量都可以理解成 chip 单位的索引/偏移；乘上 $L$ 之后，就变成 raw IQ 里的 sample 单位索引。

物理意义：STO 是时间窗口偏移，所以最直接的补偿不是乘一个复指数，而是把采样窗口挪回去。

---

## 4. 整数 CFO 估计：利用 downchirp/SFD 拆整数 CFO

前面 preamble upchirp 的峰里面，整数 CFO 和整数 STO 混在一起。gr-lora_sdr 后面利用 downchirp 的结果估计 `m_cfo_int`。源码中 `down_val` 来自 downchirp 的解调值，然后：

$$\hat{\kappa}_I = \begin{cases} \left\lfloor \dfrac{k_d}{2} \right\rfloor, & k_d < \dfrac{N}{2}, \\ \left\lfloor \dfrac{k_d - N}{2} \right\rfloor, & k_d \ge \dfrac{N}{2}. \end{cases}$$

这就是代码里的 `m_cfo_int = floor(down_val / 2)` 或 `floor((down_val - N)/2)`。

### 为什么有个除以 2？

对 upchirp 做 dechirp 后，频率峰位置近似是：

$$k_u \equiv \kappa_I - \mu_I \pmod{N}$$

这和 Hi²LoRa 里写的逻辑一致：dechirp 后 frequency drift 近似是 CFO 减 STO，即 $f_s \approx -\tau_s + \delta_s$。

而对 downchirp 来说，STO 的符号反过来，所以峰位置近似是：

$$k_d^{\text{raw}} \equiv \kappa_I + \mu_I \pmod{N}$$

注意：这里的 $k_d^{\text{raw}}$ 是还没做 upchirp 粗同步之前的 downchirp 观测。

g r-lora_sdr 在 DETECT 阶段做的粗同步，是先从连续 upchirp 里找最频繁的 peak：

$$\hat{k} = k_u$$

代码里就是 `k_hat = most_frequent(...)`，然后通过 `items_to_consume = os_factor * (N - k_hat)` 调整后续窗口位置。也就是说，它不是知道了真正的 $\mu_I$，而是用 upchirp peak 去移动窗口。

这一步的真实效果是：**让后续 upchirp 看起来回到 0 bin。**

因为原来 upchirp peak 是 $k_u = \kappa_I - \mu_I$，粗同步把时间窗口移动后，相当于把 residual timing 改成一个新的 $\mu_I'$，使得 upchirp peak 变成：

$$k_u' = \kappa_I - \mu_I' \approx 0$$

所以：

$$\mu_I' \approx \kappa_I$$

**这句话很重要：粗同步之后，接收窗口并不一定是真正物理对齐了；它只是让 upchirp 的 FFT peak 对齐了。** 由于 upchirp peak 里混着 CFO，这个"对齐"会留下一个等价于 CFO 的 residual timing。

然后再看 downchirp。downchirp 的 peak 近似是：

$$k_d \equiv \kappa_I + \mu_I' \pmod{N}$$

而刚才粗同步后 $\mu_I' \approx \kappa_I$，所以：

$$k_d \equiv \kappa_I + \kappa_I = 2\kappa_I \pmod{N}$$

因此：

$$\hat{\kappa}_I \approx \frac{k_d}{2}$$

这就是为什么源码里 `down_val` 要除以 2。

得到整数 CFO 后，补偿向量为：

$$c_{\text{CFO},I}[n] = \exp\left(-j2\pi\frac{\hat{\kappa}_I n}{N}\right)$$

完整 CFO 补偿是：

$$c_{\text{CFO}}[n] = \exp\left(-j2\pi\frac{(\hat{\kappa}_I + \hat{\kappa}_F)n}{N}\right)$$

代码里整数 CFO 还伴随一个 rotate 操作，用来修正整数 bin 级别的循环偏移；然后再乘整数 CFO 相位补偿向量。

---

## 5. SFO 估计：不是单独拟合，而是从 CFO 推出来

gr-lora_sdr 的 changelog 明确说，SFO 估计利用 CFO 和 SFO 都由同一个 reference clock 引起这一关系；补偿方法包括 preamble 中两步细化估计，以及 payload 阶段的样点 puncturing/insertion。

### 晶振如何产生 CFO 和 SFO

载波频率和采样频率都由同一个 reference clock 产生。

假设发射端参考时钟有相对误差 $\epsilon_{\text{tx}}$，接收端参考时钟有相对误差 $\epsilon_{\text{rx}}$。

那么发射端实际 RF carrier 近似是：

$$f_{c,\text{tx}} = f_c (1 + \epsilon_{\text{tx}})$$

接收端本振近似是：

$$f_{c,\text{rx}} = f_c (1 + \epsilon_{\text{rx}})$$

所以下变频之后看到的 CFO 是：

$$\Delta f_{\text{CFO}} = f_{c,\text{tx}} - f_{c,\text{rx}} = f_c (\epsilon_{\text{tx}} - \epsilon_{\text{rx}})$$

于是：

$$\frac{\Delta f_{\text{CFO}}}{f_c} = \epsilon_{\text{tx}} - \epsilon_{\text{rx}}$$

这个量就是收发两端参考时钟的相对 ppm 误差。

如果 LoRa 芯片/SDR 的采样时钟也由这个 reference clock 派生，那么采样频率偏差也近似由同一个相对误差决定：

$$\epsilon_{\text{SFO}} \approx \epsilon_{\text{tx}} - \epsilon_{\text{rx}} \approx \frac{\Delta f_{\text{CFO}}}{f_c}$$

这就是为什么 gr-lora_sdr 可以用 CFO 推 SFO。

设载波中心频率为 $f_c$，带宽为 $B$，CFO bin 估计为 $\hat{\kappa} = \hat{\kappa}_I + \hat{\kappa}_F$。

一个 LoRa frequency bin 对应的实际频率间隔是：

$$\Delta f_{\text{bin}} = \frac{B}{N}$$

所以 CFO 的 Hz 估计是：

$$\widehat{\Delta f} = \hat{\kappa} \frac{B}{N}$$

如果 CFO 和 SFO 来自同一个参考时钟偏差，则相对时钟误差为：

$$\hat{\epsilon} = \frac{\widehat{\Delta f}}{f_c} = \frac{\hat{\kappa} B}{N f_c}$$

代码里写成：

$$\hat{s}_{\text{SFO}} = \frac{\hat{\kappa} B}{f_c}, \qquad \hat{\epsilon} = \frac{\hat{s}_{\text{SFO}}}{N}$$

也就是：

```cpp
sfo_hat = ((m_cfo_int + m_cfo_frac) * m_bw) / m_center_freq;
clk_off = sfo_hat / m_number_of_bins;
fs_p = m_bw * (1 - clk_off);
```

这里 `sfo_hat` 不是普通意义上"无量纲 SFO"，而更像：

$$\hat{s}_{\text{SFO}} = N\hat{\epsilon}$$

也就是每个 LoRa symbol 累积多少 chip 的 timing drift。这点很重要。源码正是用 `sfo_hat` 更新后续 STO 漂移。

### SFO 的线性影响

设采样频率相对误差是 $\epsilon$。那么经过一个 LoRa symbol，也就是 $N$ 个 chip，累计 timing drift 大约是：

$$s_{\text{SFO}} = N\epsilon$$

所以第 $q$ 个 symbol 的 STO 可以写成：

$$\mu_q = \mu_0 + q \cdot s_{\text{SFO}}$$

这里 $s_{\text{SFO}}$ 才是 gr-lora_sdr 里 `sfo_hat` 更接近的东西。

所以不是说 $\mu_q = \text{constant}$，而是说：

$$\mu_{q+1} - \mu_q = s_{\text{SFO}} \approx \text{constant}$$

也就是 STO 线性漂移。

这和 Hi²LoRa 的说法一致：SFO 会改变每个 chip 的采样时间，导致不同 symbol 的 STO 随时间变化；论文也明确把 SFO 解释成 time-varying STO 的来源。

---

## 6. Preamble 内 SFO 相位补偿：修正 chirp 斜率失配

有 SFO 时，实际采样率相当于：

$$f_s' = B(1 - \hat{\epsilon})$$

理想采样率是：

$$f_s = B$$

对 preamble 的全局样点编号 $n$，定义：

$$q = \left\lfloor\frac{n}{N}\right\rfloor, \qquad r = \operatorname{mod}(n, N)$$

其中 $q$ 表示第几个 preamble chirp，$r$ 表示当前 chirp 内第几个 chip。

gr-lora_sdr 构造的 SFO 补偿向量是：

$$c_{\text{SFO}}[n] = \exp\left(-j2\pi \Psi_{\text{SFO}}(q,r)\right)$$

其中：

$$\Psi_{\text{SFO}}(q,r) = \frac{r^2}{2N}\left[\left(\frac{B}{f_s'}\right)^2 - \left(\frac{B}{f_s}\right)^2\right] + \left[q\left(\left(\frac{B}{f_s'}\right)^2 - \frac{B}{f_s'}\right) + \frac{B}{2}\left(\frac{1}{f_s} - \frac{1}{f_s'}\right)\right]r$$

这就是源码里 `sfo_corr_vect[n] = exp(-j2π(...))` 的数学形式。代码在 correct SFO in the preamble upchirps 后面先计算 `sfo_hat`、`clk_off`、`fs_p`，再构造这个向量并乘到 `preamble_upchirps` 上。

### 公式来源推导

这个公式可以从**连续 LoRa upchirp 相位 + 采样率失配导致的时间轴缩放**直接推出来。关键不是凭空凑项，而是比较两件事：

- 用理想采样率 $f_s$ 采到的 chirp 相位
- 用偏移采样率 $f_s'$ 采到的 chirp 相位

两者相减，就是 gr-lora_sdr 里要乘掉的 SFO 相位误差。

先从连续域的 LoRa base upchirp 写起。设一个 chirp 时长为：

$$T = \frac{N}{B}$$

基带 upchirp 从 $-B/2$ 扫到 $B/2$，所以瞬时频率可以写成：

$$f(t) = -\frac{B}{2} + \frac{B}{T}t$$

相位是频率积分：

$$\Phi(t) = \int_0^t f(\lambda)\,d\lambda = -\frac{B}{2}t + \frac{B}{2T}t^2$$

因为 $T = N/B$，所以：

$$\Phi(t) = -\frac{B}{2}t + \frac{B^2}{2N}t^2$$

这里的 $\Phi(t)$ 是以 cycles 为单位的相位，真正的复指数是 $e^{j2\pi\Phi(t)}$。

如果采样率是 $f_s$，第 $r$ 个 chip 对应时间 $t = r/f_s$，代进去：

$$\Phi_s(r) = -\frac{B}{2}\frac{r}{f_s} + \frac{B^2}{2N}\frac{r^2}{f_s^2}$$

整理成：

$$\Phi_s(r) = \frac{r^2}{2N}\left(\frac{B}{f_s}\right)^2 - \frac{B}{2f_s}r$$

这就是一个 chirp 内的理想采样相位。

现在设全局样点编号是 $n = qN + r$，其中 $q = \lfloor n/N \rfloor$，$r = \operatorname{mod}(n,N)$。

理想情况下，第 $q$ 个 chirp 的起点是 $qT = qN/B$。但如果采样率是 $f_s'$，接收端第 $qN+r$ 个采样点对应的真实时间是 $(qN+r)/f_s'$。

所以它相对于第 $q$ 个理想 chirp 起点的局部时间是：

$$t'_{q,r} = \frac{qN+r}{f_s'} - \frac{qN}{B} = qN\left(\frac{1}{f_s'} - \frac{1}{B}\right) + \frac{r}{f_s'}$$

把它乘上 $B$，写成归一化时间更清楚：

$$Bt'_{q,r} = qN\left(\frac{B}{f_s'} - 1\right) + r\frac{B}{f_s'}$$

令 $a = B/f_s'$，则 $Bt'_{q,r} = qN(a-1) + ra$。

连续 chirp 相位可以写成 $\Phi(t) = \frac{(Bt)^2}{2N} - \frac{Bt}{2}$，所以偏移采样率下，第 $q$ 个 chirp 内第 $r$ 个点的相位是：

$$\Phi_{s'}(q,r) = \frac{[qN(a-1)+ra]^2}{2N} - \frac{qN(a-1)+ra}{2}$$

展开后，只和 $q$ 有关、不和 $r$ 有关的项是当前 chirp 的公共相位，不会影响 FFT peak 位置和 chirp 内相位斜率，所以源码的补偿向量里没有保留这类纯常数项。

留下和 $r$ 有关的部分：

$$\Phi_{s'}(q,r) \doteq \frac{r^2}{2N}\left(\frac{B}{f_s'}\right)^2 + qr\left[\left(\frac{B}{f_s'}\right)^2 - \frac{B}{f_s'}\right] - \frac{B}{2f_s'}r$$

理想采样率 $f_s$ 下，一个 chirp 内第 $r$ 个点的相位是：

$$\Phi_s(r) = \frac{r^2}{2N}\left(\frac{B}{f_s}\right)^2 - \frac{B}{2f_s}r$$

所以 SFO 引起的相位误差就是：

$$\Psi_{\text{SFO}}(q,r) = \Phi_{s'}(q,r) - \Phi_s(r)$$

把两者相减，把后两个 $r$ 项合并，就得到上面的式子。

### 各项物理意义

- **第一项** $\frac{r^2}{2N}\left[\left(\frac{B}{f_s'}\right)^2 - \left(\frac{B}{f_s}\right)^2\right]$：来自 **chirp 内二次相位项**。采样率错了，等价于 chirp 的时间轴被拉伸/压缩，所以 chirp slope 不匹配。
- **第二项** $q\left[\left(\frac{B}{f_s'}\right)^2 - \frac{B}{f_s'}\right]r$：来自 **跨 chirp 累积的时间漂移**。第 $q$ 个 chirp 的起点已经因为 SFO 偏了，所以它会给当前 chirp 内的 $r$ 方向引入额外线性相位斜率。
- **第三项** $\frac{B}{2}\left(\frac{1}{f_s} - \frac{1}{f_s'}\right)r$：来自 **起始频率 $-B/2$ 的线性相位项**。

所以 SFO 补偿不是只补二次项，而是：

$$\text{SFO correction} = \text{chirp 内 slope 失配} + \text{起始频率线性项失配} + \text{跨 chirp 累积漂移}$$

---

## 7. SFO 补偿后，重新估 STO_frac

SFO 会污染 STO_frac 估计，所以 gr-lora_sdr 做完 preamble SFO 相位补偿以后，会重新调用 `estimate_STO_frac()`。

源码里直接写着：

```cpp
float tmp_sto_frac = estimate_STO_frac();
```

注释是 "better estimation of sto_frac in the beginning of the upchirps"。也就是：先把 SFO 造成的 chirp 斜率/相位畸变压下去，再重新估一个更干净的小数 STO。

### 为何要再重新估计一次 STO？

更准确地说，gr-lora_sdr 的 STO_frac 初估是这样的：

对每个 preamble upchirp，dechirp 后做 $2N$-point FFT：

$$X_q[k] = \mathrm{FFT}_{2N}\{x_q[n]\}$$

然后累加 power：

$$E[k] = \sum_q |X_q[k]|^2$$

再找：

$$k^\star = \arg\max_k E[k]$$

最后把 $k^\star$ 换成 fractional STO。

这个估计成立的隐含条件是：这些 preamble chirp 的 residual peak 位置大体一致。如果没有明显 SFO，那么每个 preamble 的峰差不多在同一个位置，累加 power 会把峰增强、把噪声平均掉。

但是如果有 SFO，则第 $q$ 个 preamble 的 STO 其实是：

$$\mu_q = \mu_0 + q \cdot s_{\text{SFO}}$$

那么每个 chirp 的峰会轻微移动：$k_q \approx -\mu_q$。于是你累加 $E[k] = \sum_q |X_q[k]|^2$，得到的峰就不是严格的 $\mu_0$，而更像是多个 chirp 的**有效平均峰 / 被 SFO 拉宽后的峰**。

所以第一次 `estimate_STO_frac()` 只是一个初估，不是最终干净的 STO。

**既然已经在估计 STO_frac 的时候对齐完时间窗口了，干啥在对齐完 SFO 之后还要再对齐一次 STO？**

因为第一次 STO_frac 估计的时候，信号里还带着 SFO 的影响。SFO 对 STO_frac 估计有两层污染：

**第一层是跨 chirp 的峰位置漂移：** $\mu_q = \mu_0 + q \cdot s_{\text{SFO}}$，多个 preamble 的 peak 不完全重合，所以 power sum 的峰会被拉宽甚至偏移。

**第二层是 chirp 内部斜率不匹配。** 采样率不对时，接收端看到的 chirp 不是和本地 downchirp 完全同斜率的 chirp。dechirp 后残留的不是一个特别干净的 tone，而是带一点 residual chirp。这样 FFT peak 也会变宽、变歪。

所以第一次估计流程其实是：

$$r[n] \xrightarrow{\text{dechirp}} \mathrm{2N\text{-}FFT} \rightarrow \hat{\mu}_F^{(0)}$$

但这里的 $\hat{\mu}_F^{(0)}$ 是带 SFO bias 的。

做完 SFO correction 后，preamble 被修成更接近理想 chirp：

$$r_S[n] = r[n] \cdot e^{-j2\pi\Psi_{\text{SFO}}(q,r)}$$

这时每个 preamble 的 peak 会更集中：$k_q^{\text{after}} \approx \text{common peak}$。

于是再做一次：

$$E_S[k] = \sum_q |X_{S,q}[k]|^2$$

得到 $\hat{\mu}_F^{(1)}$，这个才是更干净的 fractional STO。

---

## 8. 把 STO 推进到 NetID 和 Payload 起点

SFO 会让 STO 随时间累积，所以 preamble 起点处的 STO 不能直接用于后面的 NetID / payload。代码里会更新：

$$\hat{\mu} \leftarrow \hat{\mu} + \hat{s}_{\text{SFO}} \cdot L_{\text{preamble}}$$

源码是：

```cpp
m_sto_frac += sfo_hat * m_preamb_len;
```

后面进入 payload 前，还会再加上 SFD/NetID 相关的 symbol 数，比如：

```cpp
m_sto_frac += sfo_hat * 4.25;
```

物理意义：STO 是一个随时间变化的状态，而不是一个固定常数。如果每个 symbol 漂移 $\hat{s}_{\text{SFO}}$，那么经过 $Q$ 个 symbol 后：

$$\mu_Q = \mu_0 + Q \cdot \hat{s}_{\text{SFO}}$$

**为什么补偿了 SFO 了还存在 STO 漂移？** 因为之前补偿 SFO 的操作只针对前导码，payload 部分是没有被补偿的，这一步补偿只是为了得到更加干净的初始 STO。

---

## 9. 当前代码实现范围

当前 `weakPacket_decoding` 里的实现做到 **帧定界 + gr-lora_sdr 风格同步参数估计**，暂时不进入 payload 解码。

入口脚本为：

```text
scripts/run_weak_sync_chain.py
```

当前链路为：

```text
raw complex64 IQ
  -> 滑窗前导码检测
  -> chirp 起点粗对齐
  -> sync word + SFD 帧定界
  -> 按 preamble_ref_bin 做 gr-lora_sdr 式粗同步挪窗
  -> chip-rate 抽样前导码
  -> Bernier 相位差估计 CFO_frac
  -> 2N FFT 估计 STO_frac
  -> SFD downchirp 估计 CFO_int
  -> 由 CFO_int + CFO_frac 推 SFO
  -> SFO 相位补偿后重新估计 STO_frac
  -> 用 STO/CFO/SFO 重新抽取并复检两个 sync word 符号
  -> 根据 netid_offset、CFO_int、payload 处 STO_frac 计算 payload 起点
  -> 验证同步后前导码 dechirp+FFT peak 是否集中在 signed bin 0
```

新增模块为：

```text
weak_decoder/synchronization/grlora_frame_sync.py
```

输出 CSV 里和 gr-lora 同步相关的关键字段包括：

```text
grlora_coarse_offset_chips
grlora_coarse_offset_samples
grlora_synced_preamble_start_sample
grlora_synced_sfd_start_sample
grlora_synced_payload_start_sample
grlora_fine_preamble_start_sample
grlora_fine_payload_start_sample
grlora_preamble_peak_mean_signed_bin
grlora_preamble_peak_max_abs_signed_bin
grlora_preamble_bin0_count
grlora_preamble_peak_count
grlora_sync1_peak_signed_bin
grlora_sync2_peak_signed_bin
grlora_sfd_mean_signed_bin
grlora_up_symbols_used
grlora_cfo_frac_est
grlora_sto_frac_initial
grlora_sto_frac_refined
grlora_sto_frac_used
grlora_sto_sample_correction
grlora_cfo_int_est
grlora_down_val_signed_bin
grlora_cfo_total_est
grlora_cfo_hz_est
grlora_sfo_hat
grlora_sfo_samples_per_symbol
grlora_clk_off
grlora_fs_p
grlora_netid_sto_frac_est
grlora_payload_sto_frac_est
grlora_payload_sto_sample_correction
grlora_netid1_est
grlora_netid2_est
grlora_netid_offset
grlora_netid_valid
grlora_sfo_cum_initial
grlora_fine_preamble_peak_mean_signed_bin
grlora_fine_preamble_peak_max_abs_signed_bin
grlora_fine_preamble_bin0_count
grlora_fine_preamble_peak_count
```

其中 `located_preamble_start_sample` 是 SFD 定界反推出的物理前导码起点；`grlora_synced_preamble_start_sample` 是仿照 gr-lora_sdr 用前导码 peak 挪窗之后的粗同步起点。`grlora_fine_payload_start_sample` 进一步加入整数 CFO 对应的等效采样偏移、`netid_offset` 和 payload 起点处的 STO_frac 采样修正，更接近 gr-lora_sdr 进入 payload 符号输出时使用的起点。

`grlora_sfo_hat` 的单位是 **chip / symbol**，和 gr-lora_sdr 原版一致；如果要换成原始 IQ 的 sample / symbol，需要乘以过采样倍数 `os_factor`，对应 CSV 里的 `grlora_sfo_samples_per_symbol`。`grlora_payload_sto_frac_est` 也是 chip 单位的小数 STO，真正挪原始采样点时使用 `round(grlora_payload_sto_frac_est * os_factor)`，对应 `grlora_payload_sto_sample_correction`。

注意：gr-lora_sdr 的 `frame_sync` 输出 payload 时主要消除 STO/SFO 造成的采样点偏差，但 CFO 会随 `cfo_int`、`cfo_frac` 一起传给后面的 `fft_demod`，由 CFO-aware downchirp 继续处理。因此，当前频谱图里的“after framesync”仍展示粗同步挪窗后的前导码 peak 是否回到 bin0；CFO/STO/SFO 的数值验证主要看 CSV 字段。

