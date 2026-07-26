# NUS 研究主线与下一阶段 Handoff

**日期：** 2026-07-21  
**状态：** 研究方向收束；当前只验证 NUS 的存在必要性，暂不设计 pattern bank  
**优先级：** 本文档中的决定覆盖旧 handoff、`GLS.MD` 和历史实验中与之冲突的路线

## 1. 当前唯一主线

这项工作的核心不是 GLS、GSC、候选加权或 pattern 搜索，而是一个更清楚的接收机解释：

> 将一个过采样 LoRa 符号中的采样相位划分为时间域虚拟天线。Savaux 是固定虚拟阵列上的相干合并；NUS 则允许软件重新组织这些采样点，形成可重构的虚拟阵列。

设

$$
N=2^{\mathrm{SF}},\qquad R=\mathrm{OSR},
$$

一个符号的过采样 dechirp 序列为 $z[n]$。将它写成

$$
z_{p,q}=z[Rp+q],
\qquad
p=0,\ldots,N-1,
\quad
q=0,\ldots,R-1.
$$

这里，$q$ 不再只是“一个 chip 内的第几个采样点”，而被解释为固定阵列的第 $q$ 个虚拟接收通道。对候选 LoRa bin $k$，记经过精确 LoRa 相位补偿后的单点贡献为

$$
x_{p,q}(k)
=
z[Rp+q]e^{-j\phi_k(Rp+q)}.
$$

Savaux 的固定相干积累可抽象为

$$
A_{\mathrm{Savaux}}(k)
=
\sum_{p=0}^{N-1}
\sum_{q=0}^{R-1}
x_{p,q}(k).
$$

NUS pattern 则用

$$
c[p]\in\{0,\ldots,R-1\}
$$

在每个 chip 位置选择一个采样相位：

$$
A_c(k)
=
\sum_{p=0}^{N-1}
x_{p,c[p]}(k).
$$

因此，NUS 的论文意义不是“多生成几个 FFT”或“增加几个统计特征”，而是：

> 当固定虚拟阵列中的观测质量并不均匀时，重新配置采样点到虚拟支路的映射，使接收机减少坏采样对相干积累和假峰形成的影响。

这条主线必须坚定保留。当前不清楚的只是怎样构造 pattern，而不是是否还要研究 NUS。

## 2. 必须承认的数学边界

NUS 不制造新采样点，也不制造新的物理天线。所有 pattern 都来自同一个长度为 $NR$ 的波形。

如果 $R$ 条 NUS pattern 在每个 $p$ 上恰好把 $q=0,\ldots,R-1$ 各使用一次，并且最后仍然等权相干相加，那么

$$
\sum_{b=0}^{R-1}A_{c_b}(k)
=
A_{\mathrm{Savaux}}(k).
$$

此时只是重新排列了 Savaux 已经使用的样本，不可能产生增益。

在理想 AWGN、同步正确、模板正确且所有采样点同质时，Savaux 已经相当于使用全部 OSR 波形的匹配相干积累。NUS 不应稳定超过它。若在这种条件下出现稳定增益，应首先检查样本范围、同步、归一化、候选映射和测试泄漏，而不能立即宣称 NUS 有效。

所以 NUS 存在必要性的前提不是“OSR 大于 1”，而是下面这个可检验命题：

$$
\boxed{
\text{真实接收波形中的采样可靠性随 }(p,q)\text{ 非均匀，且这种非均匀性能够在 payload 判决前被预测。}
}
$$

可能造成这种现象的因素包括硬件 polyphase 幅相不一致、接收滤波后的相关噪声、窄时脉冲、ADC clipping、AGC transient，以及 residual CFO/STO/SFO 或 LoRa wrap 附近的模型失配。但这些现在都只是待验证的解释，不能先写进数据模型再用合成实验证明自己。

## 3. 当前证据等级

截至 2026-07-21，尚无被认可的证据证明 NUS 在自然弱包数据上超过 Savaux，也尚未证明真实数据中存在足够稳定、可预测的“坏采样位置”。

已有自然数据的主要事实是：

- `real_low_snr_20260718_low4_low6` 中，low4 共有 392 个可评分 payload symbols，Savaux 仅错误 1 个，SER 为 `1/392 = 0.002551`。这足以做回归检查，不足以证明新方法的统计收益。
- 三组 `USRP_IQ` 的冻结数据共有 840 个可评分 symbols，clean Savaux 错误为 0。额外注入 AWGN 后可以做受控回归，但不能证明自然采样非均匀性确实存在。
- low5--low7 当前主要受限于 detection/frame-sync，尚未形成可靠的弱 payload symbol 数据集。
- 旧的 96-pattern、Top-K、GLS/GSC、人工 colored noise、人工延迟副本和各种综合分数，都不能回答 NUS 是否在自然数据中有存在必要。

因此，下一阶段的正式里程碑不是“设计出稳定超过 Savaux 的 pattern bank”，而是：

$$
\boxed{
\text{NUS necessity / opportunity audit}
}
$$

先从数据判断 NUS 值不值得做，再决定 pattern 应该怎样构造。

## 4. 第一阶段要回答的四个问题

### 4.1 自然数据中是否真的存在采样可靠性非均匀

利用 packet 外噪声、preamble 和 header，估计采样位置或采样相位的残差统计。可以从最简单的量开始：

$$
r_{p,q}
=
\mathbb{E}
\left[
\left|
z_{p,q}-\widehat{\alpha}s_{p,q}(\widehat{\boldsymbol\theta})
\right|^2
\right],
$$

其中 $\widehat{\boldsymbol\theta}$ 只允许由 off-packet、preamble 或 header 估计，可包含 residual CFO/STO/SFO。还应记录：

- 不同 $q$ 的残差方差、幅度和相位偏差；
- $p$ 方向是否存在局部 burst、clipping 或 wrap 邻域异常；
- 不同采样相位之间的相关性，而不只看各自方差；
- 异常是否跨 chirp、跨 packet、跨 capture 重复出现；
- 观察到的差异是否显著大于有限样本估计误差。

仅发现有色噪声或协方差非对角，并不自动证明 NUS 必要。真正重要的是异常是否定位在 NUS 能够重新配置的 $(p,q)$ 采样几何上。

### 4.2 这种非均匀性能否在 payload 判决前预测

若“哪个采样点坏”只能看 payload GT、Savaux 是否判错或当前候选峰之后才知道，它就不能形成合法接收机。

需要验证：

$$
\text{preamble/header reliability map}
\longrightarrow
\text{payload sample reliability}
$$

是否能够跨符号、跨 packet，最好还能跨 capture 保持相关。训练和测试必须按 capture 或采集批次隔离，不能随机打散相邻 symbols。

### 4.3 非均匀采样是否真的参与了 Savaux 的错误形成

对每个可评分 payload symbol，只在离线诊断中使用 GT。记真实 bin 为 $k^\star$，Savaux 最强错误竞争 bin 为 $k'$。定义 Savaux 的 true-vs-false margin：

$$
M
=
|A_{\mathrm{Savaux}}(k^\star)|^2
-
|A_{\mathrm{Savaux}}(k')|^2.
$$

然后做单点或预先固定小区域的 influence/ablation 诊断，观察移除某个 $(p,q)$ 后 margin 的变化。这个分析只用于回答“坏采样是否在拉低真峰或抬高假峰”，不能作为在线 pattern 选择器，也不能把诊断上界当成算法 SER。

关键证据不是“总能事后找到几个应该删除的点”，而是：

> 被 preamble/header 独立判为低可靠的采样位置，在 held-out payload 中也系统性地产生负 margin influence，并显著集中出现在 Savaux 错误或低 margin symbols 中。

需要用随机置换采样标签或同规模随机 mask 作为零假设，排除“任何长序列总能事后找到异常点”的假象。

### 4.4 NUS 的动作空间是否覆盖这些异常

即使数据存在失配，也不代表 NUS 一定适合。还要检查异常是否能通过合法约束

$$
c[p]\in\{0,\ldots,R-1\}
$$

被避开或隔离，并且保留下来的样本仍能按照精确 LoRa 相位律相干积累。

`full colored-ML > Savaux` 只能说明完整波形中存在可利用的统计结构，不能单独证明 NUS 有必要；因为增益也可能来自一般加权、同步修正或更精确模板。只有当异常明确落在 NUS 可控制的采样几何上，NUS 才是有针对性的解决方案。

## 5. 数据与实验规范

### 5.1 主证据必须来自自然采集

- 不在真实 IQ 上人为加入专门有利于 NUS 的窄带、脉冲、回波或 colored noise。
- Added-AWGN 只能做灵敏度曲线和回归检查，不能证明 NUS 的自然必要性。
- 若现有数据没有足够多的 Savaux 错误，应明确写“数据不足”并补采，而不是调整噪声模型。
- 建议在仍能可靠切出 payload 的 waterfall 区域补采多个独立 capture/session，并使用发送端已知 payload 或独立 codec/CRC 路径产生 GT。
- 统计单位至少同时报告 symbol、packet、capture；置信区间和 train/test 隔离以 capture 为单位。

现有 `1/392` 的错误数远远不够。下一轮采集前应预先确定统计功效；作为工程上的最低目标，建议累计至少 100 个 Savaux symbol errors，且分布在不少于 3 个独立 captures，而不是集中在一个异常包中。

### 5.2 所有比较必须配对

Savaux 和任何诊断方法必须使用同一份 IQ、同一 symbol 切片、同一同步参数、同一 GT 和全部 $2^{\mathrm{SF}}$ 个候选 bin。必须输出 paired fixes/breaks 和逐 symbol 明细，不能只给平均 SER。

### 5.3 payload GT 的权限边界

GT 可以用于：

- 计算最终 SER；
- 分析 Savaux 错误；
- 进行明确标注为 post-hoc 的 influence 诊断。

GT 不可以用于：

- 选择 pattern、阈值、权重或触发条件；
- 选择只对 NUS 有利的 symbols/captures；
- 在测试集上决定 reliability 特征或结构超参数。

## 6. 第一阶段的最小实现，不做 pattern bank

若继续实现，只新增一个与在线系统隔离的 natural-data audit runner，例如：

```text
weak_decoder/os_lora/experiments/audit_nus_necessity.py
```

它只做测量，不提出复杂解码器。建议固定输出：

```text
data/experiments/nus_necessity_<date>/
  config.json
  dataset_summary.csv
  savaux_symbols.csv
  sample_reliability.csv
  reliability_transfer.csv
  margin_influence.csv
  capture_summary.csv
  RESULTS.md
```

最低输出内容：

1. 每个 capture 的可评分 symbols、Savaux errors、SER 和 margin 分布；
2. 按 $q$、按归一化 $p$、按 LoRa wrap 相对位置统计的残差/可靠性；
3. preamble/header 估计量对 held-out payload 的转移相关性；
4. 低可靠采样与 Savaux true/false margin influence 的关系；
5. Savaux errors 中可归因于局部采样异常的比例及其置信区间；
6. 与随机 mask/随机标签零假设的对照；
7. 明确的 `GO / NO-GO / DATA INSUFFICIENT` 结论。

第一阶段禁止加入：

- 96 个或任意大规模 pattern 候选库；
- Top-K 候选触发、payload oracle 或按 Savaux 错误挑 pattern；
- GLS、GSC、复杂白化检测器或多个手工指标加权；
- 人工结构化噪声与针对该噪声调参；
- 对 `weak_decoder/os_lora/system/` 的新功能扩张。

## 7. 进入 pattern 设计阶段的门槛

只有以下结论在 held-out 自然 capture 上同时成立，才进入下一阶段：

1. **非均匀：** $(p,q)$ 可靠性差异稳定存在，并有可重复的 effect size；
2. **可预测：** preamble/header/off-packet 能在 payload 判决前预测低可靠区域；
3. **与错误相关：** 预测出的低可靠区域确实系统性降低 Savaux 的 true-vs-false margin；
4. **NUS 可作用：** 这些区域能够通过合法 $c[p]$ 采样几何被避开或隔离；
5. **跨采集复现：** 结论不依赖单一 packet、单一 capture 或单一人工噪声 realization。

若其中任一项失败，应先报告失败原因。尤其是：若非均匀性不可从 payload 外预测，那么事后存在“坏点”也不足以支持一个可实现的 NUS 接收机。

## 8. 通过门槛后，pattern 设计才回答什么

pattern bank 当前没有答案，这不是本阶段要掩盖的问题。通过必要性审计后，再让数据决定模型：

- 若异常主要固定在采样相位 $q$，研究 branch/phase-aware 的确定性 NUS；
- 若异常沿 $p$ 局部突发，研究具有连续性或切换次数约束的 NUS；
- 若异常与 LoRa wrap 或候选 steering 有关，将精确 chirp 相位和 wrap 分段写入 pattern 目标；
- 若只观察到一般 colored covariance、却没有采样位置可预测性，则这更像一般 colored-ML 问题，不应硬包装成 NUS；
- 若自然数据与同质 AWGN 无显著差别，则 NUS 不具备超过 Savaux 的数据依据。

最终的 pattern 应直接来自已验证的失配结构和 LoRa 相位模型，而不是先造候选库再按 SER、margin、Fisher information 或复合分数搜索。最终接收机仍应保留 Savaux 作为固定阵列基线，NUS 是在数据证明固定几何失配时提供的可重构观测，而不是为了显得复杂而附加的模块。

## 9. 当前代码与历史材料的处理原则

- 已删除的 `weak_decoder/os_lora/experimental/` 保持删除，不恢复其中的 GSC、GLRT、CFR covariance NUS 等支线。
- `weak_decoder/os_lora/experiments/` 中现存的 Top-K、GLS 和旧 pattern 脚本只作为历史材料，不是下一阶段入口。
- `weak_decoder/os_lora/system/nonuniform_sampling.py` 中已有 pattern-bank/selection API 暂不扩展，也不把其输出作为 NUS 已有效的证据。
- `data/experiments/` 中的历史结果保留用于审计，但人工 structured-noise 结果不得进入论文主结论。
- Savaux 正式基线继续使用 `weak_decoder/baselines/savaux_oversampled/paper_oversampled_demod.py`。

## 10. 一句话交接

> 坚定 NUS，但暂停 pattern 发明。先在自然弱包数据中证明：固定 Savaux 所假设的采样同质性确实被破坏，这种破坏能从 preamble/header 提前预测，并且它确实参与了假峰和 Savaux 错误的形成。只有这三件事被数据证实，pattern 设计才有明确对象，NUS 才不是为了方法而方法。
