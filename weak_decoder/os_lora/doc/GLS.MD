# 系统分成离线和在线两部分
## 离线部分：确定“用哪 8 条 pattern”
离线阶段不是训练神经网络，而是从 96 个人工构造的非均匀 pattern 中，选出对当前接收机噪声环境最有价值的 8 个。

完整过采样符号有：

$ NR
=
1024\times 4
=
4096 $

个 IQ 点。

每条 pattern 在每个 chip 内只选择一个过采样点：

$ n_b[p]
=
4p+c_b[p],
\qquad
c_b[p]\in\{0,1,2,3\} $

所以每条 pattern 最终从 4096 个点中取出 1024 个点，形成一条虚拟 branch。

96 个 pattern 包含不同时间尺度的 offset 切换：

+ 短尺度 $ w=2,4 $：快速切换 offset；
+ 长尺度 $ w=8,\ldots,256 $：慢速切换 offset；
+ block pattern：主要观察块间变化；
+ multiscale pattern：同时包含块内和块间变化。

这些 pattern 并不是新的物理采样，也不是新的独立信号副本，而是对同一组 OSR $ =4 $ IQ 数据进行不同的软件投影。前面的实验已经证明：在 AWGN 下，不管构造多少 pattern，扣除相关性后，有效自由度不会超过 OSR $ =4 $。

---

## 离线 pattern 选择过程
### 白化后信号方向的能量
从真实 IQ 的 off-packet 区域取出很多纯噪声窗口。

每个噪声窗口都经过和正式解调相同的：

$ \text{pattern 取样}
\rightarrow
\text{LoRa 候选模板校正}
\rightarrow
\text{候选 bin 投影} $

然后将 96 个 pattern 的输出堆成向量：

$ \mathbf{y}_{\ell}[m]
\in
\mathbb{C}^{96} $

利用**多个 off-packet 窗口估计噪声协方差**：

$ \widehat{\mathbf{C}}_{\mathrm{off}}
=
\frac{1}{L}
\sum_{\ell=1}^{L}
\mathbf{y}_{\ell}[m]
\mathbf{y}_{\ell}^H[m] $

如果已经对样本做了去均值，更严格地应写成：

$ \widehat{\mathbf{C}}_{\mathrm{off}}
=
\frac{1}{L-1}
\sum_{\ell=1}^{L}
\left(
\mathbf{y}_{\ell}[m]
-
\bar{\mathbf{y}}[m]
\right)
\left(
\mathbf{y}_{\ell}[m]
-
\bar{\mathbf{y}}[m]
\right)^H $

由于正确候选经过完整 LoRa 相位校正后，所有 pattern 的信号响应相同，因此：

$ \mathbf{a}
=
\sqrt{N}\,\mathbf{1} $

:::info
选择 pattern 的目标，**是让这组 pattern 对信号方向的白化后信息量尽可能大：**

$ J(\mathcal{I})
=
\mathbf{a}_{\mathcal{I}}^H
\widehat{\mathbf{C}}_{\mathcal{I}}^{-1}
\mathbf{a}_{\mathcal{I}} $or$ J(\mathcal{I})
=
\left\|
\mathbf{C}_{\mathcal{I}}^{-1/2}
\mathbf{a}_{\mathcal{I}}
\right\|^2 $

表示的是：假设 $ |x|^2=1 $ 时，**白化后的信号能量。或者说，是信号以噪声标准差为单位的坐标系中的能量**

:::

---

### 贪心实现
接着从空集合开始，贪心最大化：

$ J(\mathcal{I})
=
\mathbf{a}_{\mathcal{I}}^H
\widetilde{\mathbf{C}}_{\mathcal{I}}^{-1}
\mathbf{a}_{\mathcal{I}} $

:::info
假设已经选中了 pattern 集合 $ \mathcal{I} $，那么对每个尚未被选中的候选 pattern $ j $，计算它带来的边际信息增益：

$ \Delta J_j
=
\frac{
\left|
a_j
-
\mathbf{C}_{j\mathcal{I}}
\mathbf{C}_{\mathcal{I}\mathcal{I}}^{-1}
\mathbf{a}_{\mathcal{I}}
\right|^2
}{
C_{jj}
-
\mathbf{C}_{j\mathcal{I}}
\mathbf{C}_{\mathcal{I}\mathcal{I}}^{-1}
\mathbf{C}_{\mathcal{I}j}
} $

其中，分子表示：

$ \left|
a_j
-
\mathbf{C}_{j\mathcal{I}}
\mathbf{C}_{\mathcal{I}\mathcal{I}}^{-1}
\mathbf{a}_{\mathcal{I}}
\right|^2 $

即**新 pattern 的信号响应中，扣除已有 pattern 可以解释的部分后，剩余的新增信号信息。**

分母表示：

$ C_{jj}
-
\mathbf{C}_{j\mathcal{I}}
\mathbf{C}_{\mathcal{I}\mathcal{I}}^{-1}
\mathbf{C}_{\mathcal{I}j} $

即**新 pattern 的噪声中，扣除与已有 pattern 重复部分后，剩余的条件噪声方差。**

:::

<details class="lake-collapse"><summary id="ud60f807c"><span class="ne-text">举个具体的例子</span></summary><p id="ucfaace22" class="ne-p"><span class="ne-text">协方差矩阵由所有 pattern 在背景 bin 上的输出估计：</span></p><p id="uebd5e371" class="ne-p" style="text-align: center"><span id="PFuXN" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/66253ffab9eb2a462152d790ca85cf42.svg"></span></p><p id="u71aaa966" class="ne-p"><span class="ne-text">其中：</span></p><ul class="ne-ul"><li id="u39fdfde0" data-lake-index-type="0"><span id="azzcN" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/a0deb98d2a744dcdbc1fa3a8dfbb3f9b.svg"></span><span class="ne-text">：pattern </span><span id="XPJzp" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/d4cd21d60552e207f237e82def9029b6.svg"></span><span class="ne-text"> 的噪声功率；</span></li><li id="u2f1dfb86" data-lake-index-type="0"><span id="m5Bdk" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/cb482e8d0d2a6a604a3647a70b6b8907.svg"></span><span class="ne-text">：pattern </span><span id="SkM5C" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/d4cd21d60552e207f237e82def9029b6.svg"></span><span class="ne-text"> 与 pattern </span><span id="fzImq" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/34c7b563b30bde3c748139530686798e.svg"></span><span class="ne-text"> 的噪声相关性；</span></li><li id="u49bfb30d" data-lake-index-type="0"><strong><span class="ne-text">实际使用时还会加入对角加载，以提高协方差求逆的稳定性。</span></strong></li></ul><p id="u8f0df417" class="ne-p"><span class="ne-text">假设完整协方差矩阵为：</span></p><p id="ubff82307" class="ne-p" style="text-align: center"><span id="qRvWH" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/3d710d952b5e34a9153da534a5de14e5.svg"></span></p><p id="u1746b24f" class="ne-p"><span class="ne-text">当前已经选择：</span></p><p id="uc55fb8f0" class="ne-p" style="text-align: center"><span id="dThmB" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/93c928e7def0e9d993509f535fb664ae.svg"></span></p><p id="u8448e4ba" class="ne-p"><span class="ne-text">现在准备考察候选 pattern </span><span id="YCkuj" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/8248971b60839fc0a02c1ab952029333.svg"></span><span class="ne-text">。</span></p><p id="u27b0b2ac" class="ne-p"><span class="ne-text">那么：</span></p><p id="u10b30f53" class="ne-p" style="text-align: center"><span id="p4B2Y" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/98a27df7c2324027c5b330c713a19719.svg"></span></p><p id="uaf8cbeff" class="ne-p"><span class="ne-text">表示已选 pattern 之间的协方差，即取完整矩阵的第 </span><span id="fTxsW" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/6f7e1460f638cdb85d4d6a406684e2ca.svg"></span><span class="ne-text"> 行和第 </span><span id="t3KEn" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/6f7e1460f638cdb85d4d6a406684e2ca.svg"></span><span class="ne-text"> 列。</span></p><p id="uaec96780" class="ne-p" style="text-align: center"><span id="x6WVP" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/183538cd6add49d8658acf5a5463d6c2.svg"></span></p><p id="u43e95d85" class="ne-p"><span class="ne-text">表示候选 pattern 4 与已选 pattern 之间的相关性，即取完整矩阵的第 4 行和第 </span><span id="twoFJ" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/6f7e1460f638cdb85d4d6a406684e2ca.svg"></span><span class="ne-text"> 列。</span></p><p id="u3ccf2d5b" class="ne-p"><span class="ne-text">对应的列向量为：</span></p><p id="uef64da41" class="ne-p" style="text-align: center"><span id="CFhzP" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/18b3f82356e94eef2109d77a34a896d2.svg"></span></p><p id="u19cdcc40" class="ne-p"><span class="ne-text">然后计算：</span></p><p id="uc667cf34" class="ne-p" style="text-align: center"><span id="lEQ8W" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ec76df8e87106289b957376e7b8ed0a3.svg"></span></p><p id="u1d527e83" class="ne-p"><span id="I99Go" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/643ebff21ea62f96ba8331522d3d33ef.svg"></span><span class="ne-text"> 是利用已有 pattern 的噪声来线性预测候选 pattern 4 噪声的最优系数：</span></p><p id="u884812f7" class="ne-p" style="text-align: center"><span id="C134c" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/c0751ab0b28b328b73312264fc60328f.svg"></span></p><p id="ueabc5f03" class="ne-p"><span class="ne-text">因此，pattern 4 在去除与已有 pattern 重复的部分后，真正带来的新观测为：</span></p><p id="ub5491086" class="ne-p" style="text-align: center"><span id="QVec7" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/1f9439a366eba4bc32a185ba48da352c.svg"></span></p><p id="uf0dc1ff4" class="ne-p"><span class="ne-text">假设所有 pattern 对目标信号的响应均为 1：</span></p><p id="u1bfcc297" class="ne-p" style="text-align: center"><span id="oJ2tl" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/22b8556cf7ed38aa2d73c35cf8fe8e80.svg"></span></p><p id="uec7bbbac" class="ne-p"><span class="ne-text">消除重复噪声的同时，也会消除相应的重复信号成分。因此，剩余的信号响应为：</span></p><p id="u8e8a4900" class="ne-p" style="text-align: center"><span id="SNPxp" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/828591d39289d07280d218e43efd17de.svg"></span></p><p id="ud0ff0812" class="ne-p"><span class="ne-text">剩余的条件噪声方差为：</span></p><p id="u8005ffee" class="ne-p" style="text-align: center"><span id="smUvB" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/6dcfad6ba9aac3c5e5b3dfb66e356b14.svg"></span></p><p id="u1f0464d2" class="ne-p"><span class="ne-text">因此，候选 pattern 4 带来的边际信息增益为：</span></p><p id="u41ec1f72" class="ne-p" style="text-align: center"><span id="CLpbS" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/22b76f613c9dd26f4869b15768231bb0.svg"></span></p><p id="u7ef6f2eb" class="ne-p"><span class="ne-text">而：</span></p><p id="u7bfef90a" class="ne-p" style="text-align: center"><span id="nyG9d" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/f6dfbe2c42dc2f66175721738cd7b870.svg"></span></p><p id="u0461dd61" class="ne-p"><span class="ne-text">表示当前已选 pattern 经过最优 GLS 合并后能够获得的归一化输出信噪比。</span></p><p id="u5735e483" class="ne-p"><span class="ne-text">更准确地说，当目标信号功率满足 </span><span id="Jo5xQ" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/df2813fec5e9dd41ec974a1ffa7a191d.svg"></span><span class="ne-text"> 时，最大输出信噪比就是 </span><span id="ABdU7" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ed40f14737ced37e79fb0c4a98fe773c.svg"></span><span class="ne-text">；一般情况下则为：</span></p><p id="u58a47067" class="ne-p" style="text-align: center"><span id="Od7RC" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/eb1c567f1dbf800d4e3dc5cc45bbcd3a.svg"></span></p><p id="uf114362e" class="ne-p"><span class="ne-text">在这个例子中：</span></p><p id="u55245f23" class="ne-p" style="text-align: center"><span id="eFrQT" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/78a66e28e4c409225f4a10163ce2fa29.svg"></span></p><p id="ud4aaa060" class="ne-p"><span class="ne-text">加入 pattern 4 后：</span></p><p id="ubf9e271e" class="ne-p" style="text-align: center"><span id="VWz3o" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/419279d77cd7b936597070f5a7d87ad4.svg"></span></p><p id="u1fb937c0" class="ne-p"><span class="ne-text">因此：</span></p><p id="u89abc9c9" class="ne-p" style="text-align: center"><span id="PKwYM" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/f21294fed9d0abc07ec24ffdfea3cb77.svg"></span></p><p id="u117e0fb9" class="ne-p"><span class="ne-text">一句话总结：</span></p><p id="ua1baa48b" class="ne-p" style="text-align: center"><span id="TdldY" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/94ef55b18ce9117f39adc29742a0b709.svg"></span></p></details>
<details class="lake-collapse"><summary id="u421fb23d"><span class="ne-text">解释</span></summary><h2 id="bzzIj"><span class="ne-text">1. 这个式子并不是直接计算候选 pattern 的信号幅度</span></h2><p id="ufebda6a8" class="ne-p"><span class="ne-text">先写出观测模型：</span></p><p id="uc093c06c" class="ne-p" style="text-align: center"><span id="ojPxJ" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/0773a36f081dc16efaabaccfff5bd388.svg"></span></p><p id="ub473ddfe" class="ne-p"><span class="ne-text">对于候选 pattern </span><span id="EX6Ax" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/036441a335dd85c838f76d63a3db2363.svg"></span><span class="ne-text">：</span></p><p id="u24c6b986" class="ne-p" style="text-align: center"><span id="QMgp3" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/1a620979f06f3352698a9e752f39ed38.svg"></span></p><p id="uea65d81d" class="ne-p"><span class="ne-text">我们希望</span><strong><span class="ne-text">从候选观测 </span></strong><span id="MoZ9h" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/31fee971c83cea982db74c1cea622fd7.svg"></span><strong><span class="ne-text"> 中减去已有 pattern 能够预测的噪声部分</span></strong><span class="ne-text">：</span></p><p id="u9177823e" class="ne-p" style="text-align: center"><span id="AFmiT" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/b76edceff092910851c4db7c1be7cc02.svg"></span></p><p id="ub3eca905" class="ne-p"><span class="ne-text">关键问题是：预测系数 </span><span id="FZqZN" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/cb4030c433eacf966cb867a05bf3a8a5.svg"></span><span class="ne-text"> 应该取多少？</span></p><p id="u0c4087cb" class="ne-p"><span class="ne-text">它由下面的线性最小均方误差噪声预测问题决定：</span></p><p id="uf53d3ded" class="ne-p" style="text-align: center"><span id="FItW8" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/5145dabba4f26ce33509b3d7c007e8e6.svg"></span></p><p id="u04baa804" class="ne-p"><span class="ne-text">最优解为：</span></p><p id="u6c400eba" class="ne-p" style="text-align: center"><span id="lwUHb" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/70b80175aa710d4290b1c8405dd400ae.svg"></span></p><p id="uc843e56a" class="ne-p"><span class="ne-text">因此：</span></p><p id="u9350f7df" class="ne-p" style="text-align: center"><span id="pCSoW" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/064273588fa22e67f5a770044400ee77.svg"></span></p><p id="u9dd7c630" class="ne-p"><span class="ne-text">这一项不是候选 pattern 本身的信号幅度，而是：</span></p><p id="u360d992f" class="ne-p" style="text-align: center"><span id="eYoJh" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/6c97f7f42d7cbe49ab2b9155c56f76a2.svg"></span></p><p id="u0a210844" class="ne-p"><span class="ne-text">将观测模型代入：</span></p><p id="u289b540c" class="ne-p" style="text-align: center"><span id="sVME3" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/88fa714d6b97a466c9f8fd2268047441.svg"></span></p><p id="ue1137766" class="ne-p"><span class="ne-text">所以，候选 pattern 去除重复部分后剩余的信号响应为：</span></p><p id="ua633d0b7" class="ne-p" style="text-align: center"><span id="GL94W" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/e4f6b7fb1c63f5169afd627a581f1bef.svg"></span></p><hr id="EpIvn" class="ne-hr"><p id="u4edf6324" class="ne-p"><span class="ne-text">这个最优解最直观的推导方法是</span><strong><span class="ne-text">正交原理</span></strong><span class="ne-text">：</span></p><div class="ne-quote"><p id="u2c5c0f06" class="ne-p"><span class="ne-text">最优线性预测得到的误差，必须与所有用于预测的噪声分量都不相关。</span></p></div><p id="u6cc343ba" class="ne-p"><span class="ne-text">设已有噪声向量为：</span></p><p id="ua14cd2af" class="ne-p" style="text-align: center"><span id="Smiq9" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/672771c0597258e9cdd354a055033b5a.svg"></span></p><p id="ue62ce97a" class="ne-p"><span class="ne-text">利用它预测候选噪声 </span><span id="qILaz" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/15983af38f7df14c9b00330381774843.svg"></span><span class="ne-text">：</span></p><p id="ue3e5c89a" class="ne-p" style="text-align: center"><span id="MWW1X" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/166496a1c2da41ccfb043161eb3c7b06.svg"></span></p><p id="u1d8fa355" class="ne-p"><span class="ne-text">其中，</span><span id="HJ3sw" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/cb4030c433eacf966cb867a05bf3a8a5.svg"></span><span class="ne-text"> 是一个行向量。</span></p><p id="u13b90f4e" class="ne-p"><span class="ne-text">预测误差为：</span></p><p id="u5b6f4501" class="ne-p" style="text-align: center"><span id="zYbaS" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/4c5e1b717fa77d2cc31d0d39a38de2c5.svg"></span></p><p id="u4fbd16f0" class="ne-p"><span class="ne-text">目标是最小化均方误差：</span></p><p id="ubd599f7e" class="ne-p" style="text-align: center"><span id="svQyH" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/fceeac39997ad20f7cd63cf744c24323.svg"></span></p><h2 id="jKt4s"><span class="ne-text">1. 为什么最优误差必须与已有噪声正交？</span></h2><p id="ua7bf51a5" class="ne-p"><strong><span class="ne-text">如果最优误差 </span></strong><span id="ufKq5" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/60edacdef43a82b0deee4fe93260d5d0.svg"></span><strong><span class="ne-text"> 仍然与某个已有噪声分量 </span></strong><span id="BXVw0" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/77db1de84c57b9099d7206fb2c0f1ca3.svg"></span><strong><span class="ne-text"> 相关，就说明误差中还有一部分可以利用 </span></strong><span id="V6z64" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/77db1de84c57b9099d7206fb2c0f1ca3.svg"></span><strong><span class="ne-text"> 继续预测并消除，因此当前预测器还不是最优的。</span></strong></p><p id="u32427dce" class="ne-p"><span class="ne-text">所以，最优点必须满足：</span></p><p id="u6b3e6f3a" class="ne-p" style="text-align: center"><span id="mLA71" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/37b1c89ba9944d065f45168e15a1210b.svg"></span></p><p id="uec3c7c2d" class="ne-p"><span class="ne-text">将误差表达式代入：</span></p><p id="ua6c1e217" class="ne-p" style="text-align: center"><span id="Qn0hd" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/49544da4b879fadb29aff77ae0c5dfd6.svg"></span></p><p id="ufe290690" class="ne-p"><span class="ne-text">展开得到：</span></p><p id="u2725f322" class="ne-p" style="text-align: center"><span id="fP9Ro" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/5f2793de4d072a6ba6b808026d52321f.svg"></span></p><p id="uc8588d2e" class="ne-p"><span class="ne-text">根据协方差的定义：</span></p><p id="u0c497aca" class="ne-p" style="text-align: center"><span id="W6wGP" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/77da1ac3bcbe24a924306b814be9d6f8.svg"></span></p><p id="u66dc3457" class="ne-p"><span class="ne-text">以及：</span></p><p id="ue9fc08b9" class="ne-p" style="text-align: center"><span id="sbIzK" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/3a7e5671f98ae2d88d77b4134a632be7.svg"></span></p><p id="u6aa1c1d9" class="ne-p"><span class="ne-text">因此：</span></p><p id="u32c5c185" class="ne-p" style="text-align: center"><span id="Q2Quw" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/331d5bdcb1a62f28ae4dfd196240fc46.svg"></span></p><p id="u6678f8ac" class="ne-p"><span class="ne-text">也就是下面这组线性方程：</span></p><p id="u6bb49939" class="ne-p" style="text-align: center"><span id="o91Ai" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/b41bb9c6f41c5caa834ae3ec6df717f9.svg"></span></p><p id="u647ca627" class="ne-p"><span class="ne-text">如果 </span><span id="RWMgz" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/4ad6af1c1588197acc573bf071ef8273.svg"></span><span class="ne-text"> 可逆，在等式两侧右乘其逆矩阵，得到：</span></p><p id="u68e5fa0e" class="ne-p" style="text-align: center"><span id="Fg5F8" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/70b80175aa710d4290b1c8405dd400ae.svg"></span></p><p id="ubee59453" class="ne-p"><span class="ne-text">这就是最优噪声预测系数的来源。</span></p><h2 id="PF937"><span class="ne-text">2. 关于 </span><span id="ijpBX" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/dfd2e91fdb2d65cf5f97615513d11262.svg"></span><span class="ne-text">，这个质疑是成立的</span></h2><p id="u4a360e90" class="ne-p"><span class="ne-text">文档中假设：</span></p><p id="udb4ca571" class="ne-p" style="text-align: center"><span id="IUegN" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/e0c87b6718390c8b2e69a0af270f7eff.svg"></span></p><p id="u8b528278" class="ne-p"><span class="ne-text">其依据是理想情况下：</span></p><ul class="ne-ul"><li id="ub1f26351" data-lake-index-type="0"><span class="ne-text">每条 pattern 使用相同数量、相同归一化方式的采样点；</span></li><li id="u1349e605" data-lake-index-type="0"><span class="ne-text">在正确候选 bin 上，匹配滤波已经将所有采样点的相位完全对齐；</span></li><li id="u8bfcd4f9" data-lake-index-type="0"><span class="ne-text">每条 pattern 在正确 bin 上具有相同的相干增益。</span></li></ul><p id="u4a7fce0c" class="ne-p"><span class="ne-text">因此，理想模型认为：</span></p><p id="ud1c9e563" class="ne-p" style="text-align: center"><span id="Cvian" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ca8c54e48aa1f28861b5c93c4dabdf70.svg"></span></p><p id="u331a5d04" class="ne-p"><span class="ne-text">但这不是数学上必须成立的结论，而是一个工程建模假设。</span></p><p id="u03ffc94e" class="ne-p"><span class="ne-text">真实设备中可能存在：</span></p><ul class="ne-ul"><li id="u84ff0df4" data-lake-index-type="0"><span class="ne-text">CFO、STO 和 SFO；</span></li><li id="ufd7133fa" data-lake-index-type="0"><span class="ne-text">接收滤波器的幅相失真；</span></li><li id="ufab6598a" data-lake-index-type="0"><span class="ne-text">多径和时变信道；</span></li><li id="u67438db4" data-lake-index-type="0"><span class="ne-text">IQ 不平衡与相位噪声；</span></li><li id="udce6c6fa" data-lake-index-type="0"><span class="ne-text">不同 pattern 对采样误差的敏感性不同；</span></li><li id="u881ae0f9" data-lake-index-type="0"><span class="ne-text">wrap 和边界修正不完全准确。</span></li></ul><p id="ub17d7a05" class="ne-p"><span class="ne-text">因此，真实的信号响应向量更一般地应该写成：</span></p><p id="u252b1431" class="ne-p" style="text-align: center"><span id="J2Mq0" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/85a202440723c3a17f266c61f128c446.svg"></span></p><p id="u96ea5446" class="ne-p"><span class="ne-text">其中每个 </span><span id="T7OG7" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/1b259c5c2275b6297476dcfcb1ded459.svg"></span><span class="ne-text"> 都可能具有不同的幅度和相位。</span></p><p id="u78d987e7" class="ne-p"><span class="ne-text">这时可以利用校准包、训练包或高信噪比导频估计实际响应：</span></p><p id="u8c13090a" class="ne-p" style="text-align: center"><span id="woyEI" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/d94c2628ba117190010738fe764f7a3a.svg"></span></p><p id="ue0176edc" class="ne-p"><span class="ne-text">其中，</span><span id="WFKga" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/4dbe8c2c65bd6c3f9f60b6719f29505a.svg"></span><span class="ne-text"> 表示第 </span><span id="fwMiX" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/08843ad184698b2d78d7a39e060b9574.svg"></span><span class="ne-text"> 个训练样本中，pattern </span><span id="xQXsX" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/d4cd21d60552e207f237e82def9029b6.svg"></span><span class="ne-text"> 在真实 bin 上的相干输出。</span></p><p id="uee1f3bf5" class="ne-p"><span class="ne-text">随后将实际估计的响应代入 pattern 选择目标：</span></p><p id="u8dc77d50" class="ne-p" style="text-align: center"><span id="rjtvg" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/b2155fff4d9225ef86ec319cf103c38f.svg"></span></p><p id="u6d98db6b" class="ne-p"><span class="ne-text">对应的边际增益为：</span></p><p id="ud353b8cf" class="ne-p" style="text-align: center"><span id="xOxzQ" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/faa9e90cb16c5a7e78f01889d60684f3.svg"></span></p></details>
<details class="lake-collapse"><summary id="u67b9b0a8"><span class="ne-text">为什么这个优化目标可以不带x</span></summary><p id="ub53a553b" class="ne-p" style="text-align: center"><span id="GBL6Q" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/d657c771c2354fe1da9f1f55a8005997.svg"></span></p><p id="u1d4b3da4" class="ne-p"><span class="ne-text">之前的式子严格来说并不是完整的输出 SNR，而是去掉了与合并权重 </span><span id="dAF8M" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/31598feb6f8efa426da0e9fb43a9537e.svg"></span><span class="ne-text"> 无关的 </span><span id="radtA" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/59b5e00faa73e030f3d1567a8ad24704.svg"></span><span class="ne-text"> 后，用于寻找最优合并权重的</span><strong><span class="ne-text">归一化目标函数</span></strong><span class="ne-text">。</span></p><hr id="lXd46" class="ne-hr"><h2 id="r91kT"><span class="ne-text">为什么寻找最优 </span><span id="ChQhU" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/31598feb6f8efa426da0e9fb43a9537e.svg"></span><span class="ne-text"> 时可以省略 </span><span id="siUbD" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/59b5e00faa73e030f3d1567a8ad24704.svg"></span><span class="ne-text">？</span></h2><p id="u0d21d3c0" class="ne-p"><span class="ne-text">真正要求解的是：</span></p><p id="uc25c986c" class="ne-p" style="text-align: center"><span id="sGKZz" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/052a20b2a421af0bdf993a52f8d06a56.svg"></span></p><p id="u5adb9e82" class="ne-p"><span class="ne-text">对于当前符号而言，</span><span id="v8FHy" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/712ecf7894348e92d8779c3ee87eeeb0.svg"></span><span class="ne-text"> 是一个固定的复幅度，因此 </span><span id="sv83J" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/59b5e00faa73e030f3d1567a8ad24704.svg"></span><span class="ne-text"> 不随 </span><span id="EOQv9" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/31598feb6f8efa426da0e9fb43a9537e.svg"></span><span class="ne-text"> 改变。</span></p><p id="ue4750d64" class="ne-p"><span class="ne-text">所以：</span></p><p id="u9ddd95b5" class="ne-p" style="text-align: center"><span id="jsvtL" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/9a550125e83c4ff2ccedcfdf17c3da4b.svg"></span></p><p id="uedceebbd" class="ne-p"><span class="ne-text">也就是说：</span></p><p id="uac6aeca6" class="ne-p" style="text-align: center"><span id="kxBHK" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/f79105b8f66aa7f7040a9b4d9fb4c579.svg"></span></p><h3 id="VlQzP"><span class="ne-text">完整输出 SNR</span></h3><p id="u34f2f284" class="ne-p" style="text-align: center"><span id="Scv47" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/06c1494fe59d0307836f8d537fcae8fc.svg"></span></p><h3 id="ctoge"><span class="ne-text">用于寻找最优权重的归一化目标</span></h3><p id="u7246a90a" class="ne-p" style="text-align: center"><span id="x1PSY" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/28d538047589d0596c444ea423bd2b70.svg"></span></p><p id="u967712ee" class="ne-p"><span class="ne-text">后者等价于在假设：</span></p><p id="uc6b3ed06" class="ne-p" style="text-align: center"><span id="yxoQ2" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/df2813fec5e9dd41ec974a1ffa7a191d.svg"></span></p><p id="u3284ee72" class="ne-p"><span class="ne-text">时的输出 SNR。</span></p></details>
因此：

$ \Delta J_j
=
\frac{
\text{新增独立信号能量}
}{
\text{新增条件噪声方差}
} $

每一轮选择：

$ j^\star
=
\arg\max_{j\notin\mathcal{I}}
\Delta J_j $

然后更新：

$ \mathcal{I}
\leftarrow
\mathcal{I}\cup\{j^\star\} $

重复这一过程，直到选出 8 条 pattern。

所以这确实就是：

$ \boxed{
\text{从 96 个候选中，选择白化后新增信号信息量最大的 8 个 pattern。}
} $

---

# 在线解调的真实执行流程
## 正常运行 Savaux
每个 payload 符号到来以后，系统首先只运行标准 OSR $ =4 $ Savaux。

四条固定 offset branch 分别是：

$ z_q[p]
=
z[4p+q],
\qquad
q=0,1,2,3 $

每条 branch dechirp、FFT，然后按照候选 bin $ m $ 做 branch 间相位对齐：

$ S_{\mathrm{S}}[m]
=
\sum_{q=0}^{3}
Y_q[m]
e^{-j2\pi qm/(NR)} $

Savaux 分数为：

$ P_{\mathrm{S}}[m]
=
\left|
S_{\mathrm{S}}[m]
\right|^2 $

随后寻找最大峰和次大峰，计算：

$ \Delta_{\mathrm{S}}
=
10\log_{10}
\frac{
P_{(1)}
}{
P_{(2)}
} $

如果：

$ \Delta_{\mathrm{S}}
>
0.5\ \mathrm{dB} $

系统认为 Savaux 已经足够确定，直接输出：

$ \widehat{m}
=
\arg\max_m
P_{\mathrm{S}}[m] $

此时不会计算 8-pattern 频谱，也不会运行 GLS 或 CG。

所以绝大多数容易符号的处理路径仍然是：

$ \boxed{
\text{OSR=4 Savaux}
\rightarrow
\text{输出}
} $

---

## 判断当前是否真的存在有色噪声
只有当：

$ \Delta_{\mathrm{S}}
\leq
0.5\ \mathrm{dB} $

也就是 Savaux 最大峰和次大峰比较接近时，系统才继续。

但这时候仍然不会立刻运行非均匀 pattern，而是先检测当前符号的噪声结构。

系统利用已经计算出来的** 4 条 Savaux branch 频谱**，构造：

$ \mathbf{y}_{\mathrm{br}}[m]
=
\begin{bmatrix}
Y_0[m]\\
Y_1[m]\\
Y_2[m]\\
Y_3[m]
\end{bmatrix} $

为了尽量**避免真实信号峰污染噪声估计**，**会排除：**

+ 最强的 8 个 bin；
+ 这些 bin 左右的邻居。

剩余背景 bin 构成集合 $ \mathcal{B} $，然后估计：

$ \widehat{\mathbf{C}}_{\mathrm{br}}
=
\frac{1}{|\mathcal{B}|}
\sum_{m\in\mathcal{B}}
\mathbf{y}_{\mathrm{br}}[m]
\mathbf{y}_{\mathrm{br}}^H[m] $

**在理想 AWGN 下，4 条固定 branch 使用互不重叠的采样点，因此理论上应该接近：**

$ \widehat{\mathbf{C}}_{\mathrm{br}}
\approx
\alpha\mathbf{I}_4 $

也就是：

+ 四路**噪声功率相近**；
+ branch **之间没有明显相关性。**

于是定义偏离“白噪声协方差”的程度：

$ \eta_{\mathrm{br}}
=
\frac{
\left\|
\widehat{\mathbf{C}}_{\mathrm{br}}
-
\alpha\mathbf{I}_4
\right\|_F
}{
\left\|
\alpha\mathbf{I}_4
\right\|_F
} $

其中：

$ \alpha
=
\frac{1}{4}
\operatorname{tr}
\left(
\widehat{\mathbf{C}}_{\mathrm{br}}
\right) $

纯 AWGN Monte Carlo 标定得到门限：

$ \eta_{\mathrm{th}}
=
0.107121 $

对应约：

$ P_{\mathrm{FA}}
=
10^{-4} $

因此：

+ 若 $ \eta_{\mathrm{br}}<0.107121 $，说明当前符号接近 AWGN；
+ 若 $ \eta_{\mathrm{br}}\geq0.107121 $，说明存在明显 branch 相关性、功率不均衡或结构化干扰。

AWGN 时，非均匀 pattern 没有额外自由度，复杂后端很可能只会引入协方差估计误差，因此直接保留 Savaux 输出。

只有检测到有色结构，才允许后端介入。

这也符合你之前“非均匀 pattern 不能创造更多 AWGN 独立副本”的理论结论。

---

# 计算 8 条 LoRa-aware pattern 频谱
满足两个条件后：

$ \Delta_{\mathrm{S}}
\leq
0.5\ \mathrm{dB} $

并且：

$ \eta_{\mathrm{br}}
\geq
0.107121 $

系统才真正启动 8-pattern 后端。

对每条选中的 pattern $ b $，取样点：

$ n_b[p]
=
4p+c_b[p] $

:::info
然后对每一个候选 bin $ m $，不是直接对取出的 1024 点做普通 FFT，而是使用完整的 LoRa 候选模板：

$ Y_b[m]
=
\frac{1}{\sqrt{N}}
\sum_{p=0}^{N-1}
z[n_b[p]]
s_m^*[n_b[p]] $

:::

这里的 $ s_m[n] $ 不只是一个普通复单音，还包含：

1. 候选 bin $ m $ 的逐点相位；
2. 非均匀 offset $ c_b[p] $ 对应的分数采样相位；
3. 候选 $ m $ 的 LoRa wrap 位置；
4. wrap 后的 offset 相关相位修正。

因此它本质上是：

$ \boxed{
\text{在非均匀采样位置上，对完整 LoRa 候选波形做匹配滤波}
} $

而不是简单的 NUDFT。

对于正确 bin $ m=m_0 $，所有 pattern 的理想响应都相同：

$ Y_b[m_0]
=
x\sqrt{N}
+
\eta_b[m_0] $

把 8 条 pattern 堆起来：

$ \mathbf{y}[m]
=
\begin{bmatrix}
Y_{b_1}[m]\\
\vdots\\
Y_{b_8}[m]
\end{bmatrix} $

正确候选对应模型：

$ \mathbf{y}[m_0]
=
\mathbf{a}x
+
\boldsymbol{\eta}[m_0],
\qquad
\mathbf{a}
=
\sqrt{N}\,\mathbf{1} $

---

# pattern权重计算
## 两折 cross-fit
**如果用包含候选 bin **$ m $** 自身的数据估计协方差，再用该协方差给 **$ m $** 打分，就会产生自适应过拟合。**

因此系统把 1024 个 bin 划分为两个 fold，例如：

$ \mathcal{F}_0,
\qquad
\mathcal{F}_1 $

**当给 **$ \mathcal{F}_0 $** 中的候选打分时，只使用 **$ \mathcal{F}_1 $** 的背景 bin 估计协方差；反之亦然。**

> **packet没来的地方可能就一点接收机噪声，但是packet一来可能还会有点和lora相关的结构化干扰**
>

:::info
记另一个 fold 的背景 pattern 输出矩阵为：

$ \mathbf{Y}_{-f} $

则协方差估计是：

$ \widehat{\mathbf{C}}_{-f}
=
\frac{
\mathbf{Y}_{-f}\mathbf{Y}_{-f}^H
}{
|\mathcal{B}_{-f}|
}
+
\lambda\bar{P}\mathbf{I} $（分母是样本协方差）

其中：

$ \lambda
=
0.05 $

:::

对角加载的作用是避免：

+ 样本协方差病态；
+ 某些特征值过小；
+ GLS 权重过度放大估计误差；
+ pattern 高相关造成数值不稳定。

因此这里实际使用的不是裸协方差逆，而是正则化协方差。

---

## 什么是GLS
可以把 GLS 理解成：

$ \boxed{
\text{已知噪声“哪儿大、哪儿小、哪些观测互相重复”以后，再做最小二乘拟合}
} $

它首先是一个**参数估计方法**，不是专门为 branch 合并发明的，本质其实就是白化然后分析噪声

---

### GLS 到底在估计什么
一般模型写成：

$ \mathbf{y}
=
\mathbf{D}\boldsymbol{\theta}_0
+
\boldsymbol{\eta} $

其中：

+ $ \mathbf{y} $：实际观测；
+ $ \mathbf{D} $：已知设计矩阵；
+ $ \boldsymbol{\theta}_0 $：真实但未知的固定参数；
+ $ \boldsymbol{\eta} $：噪声。

这里特意写成 $ \boldsymbol{\theta}_0 $，表示它是真实值，已经由客观环境决定，当然不会被我们改变。

接收机不知道 $ \boldsymbol{\theta}_0 $，所以拿一个候选参数 $ \boldsymbol{\vartheta} $ 去解释观测：

$ \mathbf{D}\boldsymbol{\vartheta} $

残差为：

$ \mathbf{r}(\boldsymbol{\vartheta})
=
\mathbf{y}
-
\mathbf{D}\boldsymbol{\vartheta} $

GLS 做的只是：

$ \boxed{
\widehat{\boldsymbol{\theta}}_{\mathrm{GLS}}
=
\arg\min_{\boldsymbol{\vartheta}}
\left(
\mathbf{y}
-
\mathbf{D}\boldsymbol{\vartheta}
\right)^H
\mathbf{C}^{-1}
\left(
\mathbf{y}
-
\mathbf{D}\boldsymbol{\vartheta}
\right)
} $

注意，优化变量是候选值 $ \boldsymbol{\vartheta} $，而不是改变真实的 $ \boldsymbol{\theta}_0 $。

它问的是：

> 哪个候选参数产生的理论观测 $ \mathbf{D}\boldsymbol{\vartheta} $，最接近实际观测 $ \mathbf{y} $？
>

---

### GLS 本质上就是先白化，再做普通最小二乘
令：

$ \widetilde{\mathbf{y}}
=
\mathbf{C}^{-1/2}\mathbf{y} $

$ \widetilde{\mathbf{D}}
=
\mathbf{C}^{-1/2}\mathbf{D} $

那么：

$ \mathbf{C}^{-1/2}
\left(
\mathbf{y}
-
\mathbf{D}\boldsymbol{\vartheta}
\right)
=
\widetilde{\mathbf{y}}
-
\widetilde{\mathbf{D}}\boldsymbol{\vartheta} $

因此，GLS 目标可以写成：

$ \left(
\mathbf{y}
-
\mathbf{D}\boldsymbol{\vartheta}
\right)^H
\mathbf{C}^{-1}
\left(
\mathbf{y}
-
\mathbf{D}\boldsymbol{\vartheta}
\right)
=
\left\|
\widetilde{\mathbf{y}}
-
\widetilde{\mathbf{D}}\boldsymbol{\vartheta}
\right\|_2^2 $

所以：

$ \boxed{
\text{GLS}
=
\text{先用 }\mathbf{C}^{-1/2}\text{ 白化观测}
+
\text{再做普通最小二乘}
} $

这就是它最核心的本质。

---

### GLS 的解是什么
一般解为：

$ \boxed{
\widehat{\boldsymbol{\theta}}_{\mathrm{GLS}}
=
\left(
\mathbf{D}^H
\mathbf{C}^{-1}
\mathbf{D}
\right)^{-1}
\mathbf{D}^H
\mathbf{C}^{-1}
\mathbf{y}
} $

这个结果的含义就是：**根据噪声协方差对所有观测重新加权，然后估计未知参数**。

---

### 放到 branch 模型里
固定候选列的模型为：

$ \mathbf{y}
=
\mathbf{a}x_0
+
\boldsymbol{\eta} $

其中：

+ $ x_0 $：真实但未知的公共复幅度；
+ $ \mathbf{a} $：已知的 pattern 信号响应方向；
+ $ \mathbf{y} $：$ B $ 条 pattern 的实际输出。

这对应一般 GLS 中：

$ \mathbf{D}
=
\mathbf{a},
\qquad
\boldsymbol{\theta}_0
=
x_0 $

因为未知量只有一个复标量，GLS 变成：

$ \widehat{x}_{\mathrm{GLS}}
=
\arg\min_{\xi}
\left(
\mathbf{y}
-
\mathbf{a}\xi
\right)^H
\mathbf{C}^{-1}
\left(
\mathbf{y}
-
\mathbf{a}\xi
\right) $

其中，$ \xi $ 只是接收机尝试的候选复幅度。

求解得到：

$ \boxed{
\widehat{x}_{\mathrm{GLS}}
=
\frac{
\mathbf{a}^H
\mathbf{C}^{-1}
\mathbf{y}
}{
\mathbf{a}^H
\mathbf{C}^{-1}
\mathbf{a}
}
} $

---

归一化 branch 合并权重为：

$ \mathbf{w}_{\mathrm{GLS}}
=
\frac{
\mathbf{C}^{-1}\mathbf{a}
}{
\mathbf{a}^H\mathbf{C}^{-1}\mathbf{a}
} $

维度为：$ \mathbf{C}
\in
\mathbb{C}^{B\times B},
\qquad
\mathbf{C}^{-1}
\in
\mathbb{C}^{B\times B} $，$ \mathbf{a}
\in
\mathbb{C}^{B\times 1},
\qquad
\mathbf{C}^{-1}\mathbf{a}
\in
\mathbb{C}^{B\times 1} $，因此：$ \mathbf{w}_{\mathrm{GLS}}
\in
\mathbb{C}^{B\times 1} $

将它作用在 B 条 branch 输出上：$ \mathbf{y}
=
\begin{bmatrix}
y_1\\
\vdots\\
y_B
\end{bmatrix} $得到：

$ \begin{aligned}
\mathbf{w}_{\mathrm{GLS}}^H\mathbf{y}
&=
\left(
\frac{
\mathbf{C}^{-1}\mathbf{a}
}{
\mathbf{a}^H\mathbf{C}^{-1}\mathbf{a}
}
\right)^H
\mathbf{y}
\\
&=
\frac{
\mathbf{a}^H\mathbf{C}^{-1}\mathbf{y}
}{
\mathbf{a}^H\mathbf{C}^{-1}\mathbf{a}
}
\end{aligned} $

因此：$ \widehat{x}_{\mathrm{GLS}}
=
\mathbf{w}_{\mathrm{GLS}}^H\mathbf{y} $，所以它们分别表示：

$ \mathbf{w}_{\mathrm{GLS}}
=
\text{各条 branch 应该如何加权} $

$ \widehat{x}_{\mathrm{GLS}}
=
\text{按照这组权重合并后得到的复数结果} $

$ \widehat{x}_{\mathrm{GLS}} $ 最终是一个复数标量，而不是向量；它的模表示合并后的幅度，辐角表示合并后的相位。

---

### 总结
GLS 并不是在优化物理信道，也不是改变真实的 $ x_0 $。

真实模型是：

$ \mathbf{y}
=
\mathbf{a}x_0
+
\boldsymbol{\eta} $

GLS 做的是：

$ \boxed{
\text{根据 }\mathbf{C}\text{ 给不同观测加权，找出最能解释 }\mathbf{y}\text{ 的 }\widehat{x}
} $

在多branch问题中，这一过程恰好表现为：

$ \boxed{
\text{利用噪声协方差设计 branch 合并权重，然后输出对共同信号幅度的估计}
} $

所以最直白的一句话是：

$ \boxed{
\text{GLS 是“懂噪声结构的加权拟合”；放到你的模型里，就是“懂噪声结构的 branch 合并”。}
} $

## GLS 权重是怎么求的
理论上最优合并方向为：

$ \mathbf{g}_f
=
\widehat{\mathbf{C}}_{-f}^{-1}
\mathbf{a} $

也就是求解线性方程：

$ \widehat{\mathbf{C}}_{-f}
\mathbf{g}_f
=
\mathbf{a} $

但代码**不显式计算矩阵逆。**

虽然这里只是 $ 8\times 8 $ 矩阵，但当前实现采用 matrix-free CG，为后续更大 pattern 数目或嵌入式实现保留统一计算形式。

:::info
对于任意输入向量 $ \mathbf{u} $，只需要实现：

$ \widehat{\mathbf{C}}_{-f}\mathbf{u}
=
\frac{
\mathbf{Y}_{-f}
\left(
\mathbf{Y}_{-f}^H\mathbf{u}
\right)
}{
|\mathcal{B}_{-f}|
}
+
\lambda\bar{P}\mathbf{u} $（从前面的式子得出来的，多x了一个向量而已）

也就是按顺序执行：

$ \mathbf{u}
\rightarrow
\mathbf{Y}_{-f}^H\mathbf{u}
\rightarrow
\mathbf{Y}_{-f}(\cdot)
\rightarrow
\text{加对角加载} $

目前 CG 只迭代：$ I_{\mathrm{CG}}
=
4 $次，**<font style="background-color:#FBDE28;">得到近似解 </font>**$ \mathbf{g}_f $**<font style="background-color:#FBDE28;">。</font>**

:::

---

## GLS 分数在做什么
对 fold $ f $ 内的候选 $ m $，使用：

$ P_{\mathrm{GLS}}[m]
=
\frac{
\left|
\mathbf{g}_f^H\mathbf{y}[m]
\right|^2
}{
\mathbf{a}^H\mathbf{g}_f
} $

其中：

$ \mathbf{g}_f
\approx
\widehat{\mathbf{C}}_{-f}^{-1}
\mathbf{a} $

所以等价地，它在近似计算：

$ P_{\mathrm{GLS}}[m]
\approx
\frac{
\left|
\mathbf{a}^H
\widehat{\mathbf{C}}_{-f}^{-1}
\mathbf{y}[m]
\right|^2
}{
\mathbf{a}^H
\widehat{\mathbf{C}}_{-f}^{-1}
\mathbf{a}
} $

表示：候选 m 对应的 B 维 branch 响应向量 y，**经过基于噪声协方差设计的 GLS 最优权重合并后，再对合并结果取模平方并归一化得到的能量分数，用于与其他候选 bin 进行比较**。

---

## LoRa wrap 一致性重排
GLS 主要处理 pattern 之间的有色噪声，但某个错误候选仍可能因为某一段局部噪声而获得很高分数。

LoRa 的一个特殊结构是：候选 bin $ m $ 决定其 wrap 点：

$ p_{\mathrm{wrap}}
=
N-m $

因此对于候选 $ m $，系统把匹配输出分成：

+ wrap 前部分 $ H[m] $；
+ wrap 后部分 $ T[m] $。

若 $ m $ 是正确候选，前后两段经过各自的 LoRa wrap 相位校正后，应该由同一个复幅度 $ x $ 解释。

因此二者应当同相、尺度一致。

一致性指标为：

$ \gamma[m]
=
\frac{
|H[m]+T[m]|^2/N
}{
|H[m]|^2/(N-m)
+
|T[m]|^2/m
} $

并截断到：

$ 0
\leq
\gamma[m]
\leq
1 $

它类似于“前后两段相干能量”与“分别计算能量”的比值：

+ 如果前后段同相，$ \gamma[m] $ 接近 1；
+ 如果前后段相互冲突，$ \gamma[m] $ 变小；
+ 如果错误候选只靠某一段噪声冒头，也会被惩罚。

最终：

$ P_{\mathrm{final}}[m]
=
P_{\mathrm{GLS}}[m]
\gamma[m]^{0.5} $

然后选择：

$ \widehat{m}
=
\arg\max_m
P_{\mathrm{final}}[m] $

所以 wrap 一致性不是重新解调，而是对 GLS 候选进行 LoRa 结构约束下的重排。

# GLS pattern存在的缺陷
$ \widehat{x}_{\mathrm{GLS}}(m)
=
\frac{
\mathbf{a}_m^H
\mathbf{C}_m^{-1}
\mathbf{y}(m)
}{
\mathbf{a}_m^H
\mathbf{C}_m^{-1}
\mathbf{a}_m
} $

解决的是：

$ \boxed{
\text{在“候选 }m\text{ 就是真符号”的假设下，如何最优合并各个 branch 来估计它的复幅度}
} $

对应权重为：

$ \mathbf{w}_m
=
\frac{
\mathbf{C}_m^{-1}\mathbf{a}_m
}{
\mathbf{a}_m^H
\mathbf{C}_m^{-1}
\mathbf{a}_m
},
\qquad
\mathbf{w}_m^H\mathbf{a}_m
=
1 $

这个单位增益约束保证的是：**如果 **$ m $** 确实正确，它的目标响应不会在合并时被压掉**；同时，$ \mathbf{C}_m^{-1} $ **尽量降低该候选合并后的噪声方差**。

但它确实没有直接保证：

$ \left|
\widehat{x}_{\mathrm{GLS}}(K_0)
\right|
>
\left|
\widehat{x}_{\mathrm{GLS}}(m)
\right|,
\qquad
\forall m\neq K_0 $

## 为什么不能保证？
假设真实符号是 $ K_0 $。对于测试候选 $ m $，实际模型应写成：

$ \mathbf{y}(m)
=
\mathbf{a}(m,K_0)x
+
\boldsymbol{\eta}(m) $

GLS 却按照“$ m $ 正确”时的响应：

$ \mathbf{a}_m
\equiv
\mathbf{a}(m,m) $

去估计，得到：

$ \widehat{x}_{\mathrm{GLS}}(m)
=
\frac{
\mathbf{a}_m^H
\mathbf{C}_m^{-1}
\mathbf{a}(m,K_0)
}{
\mathbf{a}_m^H
\mathbf{C}_m^{-1}
\mathbf{a}_m
}
x
+
\frac{
\mathbf{a}_m^H
\mathbf{C}_m^{-1}
\boldsymbol{\eta}(m)
}{
\mathbf{a}_m^H
\mathbf{C}_m^{-1}
\mathbf{a}_m
} $

当 $ m=K_0 $ 时，前面的系数为 $ 1 $，因此：

$ \widehat{x}_{\mathrm{GLS}}(K_0)
=
x
+
\text{残余噪声} $

但当 $ m\neq K_0 $ 时，第一项未必为零：

$ \frac{
\mathbf{a}_m^H
\mathbf{C}_m^{-1}
\mathbf{a}(m,K_0)
}{
\mathbf{a}_m^H
\mathbf{C}_m^{-1}
\mathbf{a}_m
}
\neq
0 $

这就是 NUS 非正交匹配产生的**错误候选泄漏**。再加上某个错误候选可能碰巧获得很大的噪声投影，它的分数依然可能超过正确候选。

所以，原 GLS 主要优化的是：

$ \boxed{
\text{固定候选列内部的 branch 合并 SNR}
} $

而不是：

$ \boxed{
\text{正确候选相对于所有错误候选的判决间隔}
} $

你现有文档中其实也已经指出：这一优化只作用于完整候选响应矩阵的一列，没有联合利用其他候选的响应，也没有显式优化正确候选与错误候选之间的间隔。

真正面向解调判决的目标应该类似：

$ \max_{\mathbf{w}}
\left[
\left|
\mathbf{w}^H
\mathbf{a}(K_0,K_0)
\right|^2
-
\max_{m\neq K_0}
\left|
\mathbf{w}^H
\mathbf{a}(m,K_0)
\right|^2
\right] $

或者采用带旁瓣约束的形式：

$ \begin{aligned}
\min_{\mathbf{w}_m}\quad
&
\mathbf{w}_m^H
\mathbf{C}_m
\mathbf{w}_m
+
\lambda
\sum_{K\neq m}
\left|
\mathbf{w}_m^H
\mathbf{a}(m,K)
\right|^2
\\
\text{s.t.}\quad
&
\mathbf{w}_m^H
\mathbf{a}(m,m)
=
1
\end{aligned} $

这才同时要求：

+ 正确候选无失真保留；
+ 加性噪声被压低；
+ 其他符号泄漏到该候选的响应也被压低。

所以一句话总结：

$ \boxed{
\text{GLS 保证“假设 }m\text{ 正确时，对 }m\text{ 的估计最干净”，但不保证“}m\text{ 比其他候选更像真符号”。}
} $

## 我们真正希望GLS带来收益的来源
**真正的解码增益来源了**：如果**错误高峰主要由噪声造成**，而这些噪声**在不同 NUS pattern 上具有可利用的相关结构，那么 GLS 确实能够压低错误高峰**，从而改善正确 bin 与错误 bin 的排序，带来解码增益。

但有一个关键修正：

$ \boxed{
\text{GLS 不是直接把噪声 }\mathbf{B}\text{ 变小，而是选择一个投影，使合并后的噪声方差最小。}
} $

### 放到每个候选 bin 上看
对于候选 $ m $，多条 pattern 的输出向量为：

$ \mathbf{y}(m)
=
\mathbf{a}_m x_m
+
\boldsymbol{\eta}(m) $

GLS 选择：

$ \mathbf{w}_m
=
\frac{
\mathbf{C}_m^{-1}\mathbf{a}_m
}{
\mathbf{a}_m^H
\mathbf{C}_m^{-1}
\mathbf{a}_m
} $

使得：

$ \mathbf{w}_m^H\mathbf{a}_m
=
1 $

同时最小化：

$ \mathbf{w}_m^H
\mathbf{C}_m
\mathbf{w}_m $

所以合并输出为：

$ \widehat{x}(m)
=
\mathbf{w}_m^H\mathbf{y}(m)
=
x_m
+
\mathbf{w}_m^H\boldsymbol{\eta}(m) $

正确候选的信号响应被保留，而输出噪声方差变为：

$ \operatorname{Var}
\left[
\mathbf{w}_m^H\boldsymbol{\eta}(m)
\right]
=
\frac{
1
}{
\mathbf{a}_m^H
\mathbf{C}_m^{-1}
\mathbf{a}_m
} $

这就是 GLS 真正压制的东西。

---

### 错误 bin 的噪声峰当然可能被压下去
假设真实符号是 $ K_0 $，某个错误候选 $ m\neq K_0 $ 的高峰主要来自噪声：

$ \mathbf{y}(m)
\approx
\boldsymbol{\eta}(m) $

GLS 后：

$ \widehat{x}(m)
\approx
\mathbf{w}_m^H
\boldsymbol{\eta}(m) $

如果这个错误高峰来自某个：

+ 高噪声 pattern；
+ pattern 间相关的噪声方向；
+ 有色干扰方向；
+ 不同 pattern 输出不一致的偶然假峰；

那么 $ \mathbf{C}_m^{-1} $ 会降低这些方向的权重，**多个 pattern 之间也可能发生噪声相消**。于是，**错误 bin 的峰值会下降**。

与此同时，正确 bin 满足：

$ \widehat{x}(K_0)
\approx
x
+
\mathbf{w}_{K_0}^H
\boldsymbol{\eta}(K_0) $

其信号 $ x $ 被单位增益约束保留。

所以最终确实可能出现：

$ \boxed{
\left|
\widehat{x}(K_0)
\right|
\text{ 基本保持},
\qquad
\left|
\widehat{x}(m)
\right|
\text{ 被压低}
} $

从而增大：

$ \left|
\widehat{x}(K_0)
\right|^2
-
\max_{m\neq K_0}
\left|
\widehat{x}(m)
\right|^2 $

这当然就是解码增益。

现有文档中“GLS 只优化固定候选列 SNR”的说法，并不表示它不会改善判决；只是表示它**没有显式保证**改善所有候选之间的间隔。

---

### 为什么之前还说它“不保证正确候选更高”？
因为“有望压制”和“数学保证压制”是两回事。

GLS **最小化的是错误 bin **输出的**统计方差**：

$ \mathbb{E}
\left[
\left|
\mathbf{w}_m^H
\boldsymbol{\eta}(m)
\right|^2
\right] $

而解码时面对的是某一次具体的噪声 realization：

$ \left|
\mathbf{w}_m^H
\boldsymbol{\eta}(m)
\right|^2 $

平均方差下降以后，错误高峰出现的概率通常会下降，但不能保证每一次都下降。

例如，**某个错误 bin 的噪声刚好沿着信号方向**：

$ \boldsymbol{\eta}(m)
\approx
\gamma\mathbf{a}_m $

因为 GLS 必须满足：

$ \mathbf{w}_m^H\mathbf{a}_m
=
1 $

所以：

$ \mathbf{w}_m^H
\boldsymbol{\eta}(m)
\approx
\gamma $

**这种“长得和真实信号一样”的噪声，GLS 无法压制，否则也会把真实信号一起压掉**。

因此：

$ \boxed{
\text{GLS 能压制位于信号方向之外的结构化噪声，但不能压制与信号响应完全同向的假峰。}
} $

---

### “整个频谱都会得到增益”需要稍微改一下
> GLS 面向的观测模型可以写为：
>
> $ \mathbf{Y}
=
\mathbf{H}\mathbf{X}
+
\mathbf{B} $
>
> 其核心目标是在保留目标信号响应的同时，尽可能减小合并后的噪声项 $ \mathbf{B} $。因此，对每个候选符号进行 GLS 合并后，**整个候选响应谱都应获得一定程度的噪声抑制。**
>
> 由此产生一个问题：如果某些非均匀匹配响应 bin 的异常高峰主要由噪声造成，那么 GLS 是否也能够压低这些错误高峰，从而增大正确候选与错误候选之间的判决间隔，并最终带来实际的解码增益？
>

对每个候选 $ m $，都可以应用 GLS：

$ T_{\mathrm{GLS}}[m]
=
\frac{
\left|
\mathbf{a}_m^H
\mathbf{C}_m^{-1}
\mathbf{y}(m)
\right|^2
}{
\mathbf{a}_m^H
\mathbf{C}_m^{-1}
\mathbf{a}_m
} $

于是，整个候选频谱都会被重新计算。

但并不是所有 bin 都乘上**同一个固定的“增益系数”。**更准确地说：

$ \boxed{
\text{整个频谱中的每个候选，都经过了针对 pattern 噪声结构的最优投影。}
} $

正确 bin 得到“信号保真 + 降噪”；错误 bin 通常只有“降噪”，因此在统计意义上，正确 bin 会更加突出。

---

### 为什么 AWGN 下可能没有额外解码收益？
若原始噪声是理想白噪声，而且 Savaux 已经利用了全部 OSR 独立样本，**那么 NUS pattern 只是同一批样本的重复投影。**

**正确考虑相关性后，**$ \mathbf{C} $** 会把这些重复信息扣除，GLS 最终不会创造新的噪声自由度。**在这种情况下，错误噪声峰和正确 bin 的噪声都只能被压到 Savaux 已经达到的水平，NUS 后端没有额外空间。

:::info
现有文档也强调了：AWGN 下，pattern 数量增加并不会突破 OSR 的有效自由度。

但是，在有色噪声、结构化干扰或 pattern 间噪声功率不均衡时，GLS 可以利用 Savaux 等权合并没有利用的协方差结构，因此：

$ \boxed{
\text{错误噪声峰出现概率下降}
\Rightarrow
\text{正确候选胜出的概率提高}
\Rightarrow
\text{SER 获得增益}
} $

所以我的直觉没错：**GLS 虽然没有直接优化“正确峰减去次大峰”，但如果错误峰确实主要由可预测、可相关建模的噪声造成，它自然会通过压低这些噪声峰带来解码增益。**只是这个增益是统计性的，并且依赖噪声是否具有 Savaux 尚未利用的结构。

:::

---

# 完整数据流
整个在线实现可以压缩成：

$ \boxed{
\begin{aligned}
\text{OSR=4 IQ}
&\rightarrow
\text{4-branch Savaux}\\
&\rightarrow
\Delta_{\mathrm{S}}>0.5\ \mathrm{dB}
\ ?
\ \text{直接输出}\\
&\rightarrow
\text{估计 4-branch 背景协方差}\\
&\rightarrow
\eta_{\mathrm{br}}<0.107121
\ ?
\ \text{输出 Savaux}\\
&\rightarrow
\text{惰性计算 8 条 LoRa-aware pattern 频谱}\\
&\rightarrow
\text{2-fold 背景协方差估计}\\
&\rightarrow
\text{4 次 CG 求解 GLS 权重}\\
&\rightarrow
\text{GLS 候选分数}\\
&\rightarrow
\text{wrap 前后一致性重排}\\
&\rightarrow
\arg\max_m
\end{aligned}
} $

**然后GLS的部分实际上就是两个内容：1. 贪心选择最合适的pattern 2. 相干叠加权重需要确认一下**


