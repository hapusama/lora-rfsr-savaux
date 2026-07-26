bin选择方案设计
实则就是要融合相位信息设计一套打分准则，选择分数最高的bin作为结果，现在能想到的就是：
● 相位一致性
● rank
● 幅度
● 离argmax多远（我觉得如果上面那个实验发现了是有一定规律的话，这一点是要比rank要合理的）
第一个肯定得用，幅度也得用但是得加个权，因为低SNR情况下基本上不可信了，因此目前核心矛盾就转移到了相位咋用呢？

1.  输入是什么  
第 $k$ 个 payload symbol 做 dechirp+FFT，得到复数频域结果：
$Z_k(s), \quad s = 0, 1, \ldots, N-1$
其中 $N = 2^{\mathrm{SF}}$。
传统 LoRa 是：
$\hat{s}_k = \arg\max_s |Z_k(s)|$
但现在右下角已经看到 argmax acc 只有 0.06，所以不能这么选。
现在要用：
$\phi_k(s) = \angle Z_k(s)$
也就是每个候选 bin 的复数相位。

2. 第一版算法：Phase-Line Guided FFT Selector  
主趋势模型先用最简单的线性相位线：
$\theta_k = \theta_0 + \omega k$
这里：
● $\theta_0$：payload 起点处的包级相位
● $\omega$：每推进一个 symbol 的相位漂移
● 你图里的 $\omega \approx -0.129\pi/\text{sym}$
注意：算法里不要先 unwrap 所有候选 bin 的相位。因为候选 bin 还没选出来，强行 unwrap 会乱。第一版应该一直用 wrapped phase distance：
$d_k(s) = \mathrm{wrap}\left(\angle Z_k(s) - \theta_0 - \omega k\right)$
如果某个候选 bin 是对的，那么它的 $d_k(s)$ 应该接近 0。

3. 每个符号只保留 Top-L 候选，不看全 N 个 bin  
每个符号构造候选集合：
$S_k = \mathrm{TopL}_s |Z_k(s)|$
建议第一版直接做 $L = 16$ 或 $32$。
如果你全 $N$ 个 bin 都参与相位匹配，会出大问题：随机噪声 bin 太多，总会有一些相位偶然贴近主趋势。所以幅度虽然不能直接判决，但仍然要作为候选裁剪器。
更稳一点可以这样取候选：
$S_k = \mathrm{TopL}(|Z_k(s)|) \cup \{\text{Top peaks 的 } \pm 1 \text{ 邻居}\}$
因为实际 residual CFO/STO 会让真实能量在相邻 bin 泄漏。

4. 给候选 bin 打分  
你这个判断是对的。幅度项和 rank penalty 都可能反向伤害。
尤其你现在已经观察到 argmax acc 很低，如果再把幅度做成连续加分项，容易出现两种问题：
1. 正确 bin 相位贴线，但幅度不够高，被幅度项压下去；
2. 错误强峰相位略微贴近，也会因为幅度优势被保留下来。
所以这里更合理的设计不是"相位 + 弱幅度加权"，而是：
幅度只负责限定候选搜索区域；最终选择主要由相位一致性决定。
也就是把幅度从 score term 改成 candidate gate。

对于第 $k$ 个符号，先找到传统 FFT 的最大峰位置：
$s_k^{\max} = \arg\max_s |Z_k(s)|$
虽然 $s_k^{\max}$ 本身不一定正确，但在存在 residual CFO/STO、频谱泄露和低 SNR 扰动时，真实 bin 往往可能落在最大峰附近。因此，不直接在全频域或 Top-$L$ 中搜索，而是构建 argmax 附近的局部候选集合：
$\mathcal{S}_k = \left\{ s : \operatorname{dist}_N(s, s_k^{\max}) \le \Delta_b \right\}$
其中，$\operatorname{dist}_N(\cdot, \cdot)$ 是考虑 FFT bin 环形结构的 circular distance，$\Delta_b$ 是允许的 bin 偏移范围。第一版可以取：
$\Delta_b = 2 \sim 4$
如果担心真实 bin 偶尔不在 argmax 附近，可以再额外加入少量局部峰：
$\mathcal{S}_k = \mathcal{N}_{\Delta_b}(s_k^{\max}) \cup \mathcal{N}_1(\text{top-}P\text{ local peaks})$
在候选集合内部，不再使用幅度加分，也不再使用全局 rank penalty，而是主要计算候选 bin 相对于当前相位线的相位残差：
$d_k(s) = \operatorname{wrap}\left(\angle Z_k(s) - \theta_k\right)$
其中 $\theta_k = \theta_0 + \omega k$。
然后定义相位一致性分数：
$B_k(s) = \cos d_k(s)$
最终选择：
$\hat{s}_k = \arg\max_{s \in \mathcal{S}_k} B_k(s)$
也可以写成等价形式：
$\hat{s}_k = \arg\min_{s \in \mathcal{S}_k} \left| \operatorname{wrap}\left(\angle Z_k(s) - \theta_k\right) \right|$
这个版本更符合你的直觉：只在 FFT 最大峰附近找候选，但在局部范围内让相位一致性决定最终 bin。

如果你还是想保留"离 argmax 不能太远"的约束，可以不要用 rank，而是用距离惩罚：
$B_k(s) = \cos d_k(s) - \lambda_b \cdot \operatorname{dist}_N(s, s_k^{\max})$
不过我建议第一版先不要加这个惩罚，直接用 hard window：
$\operatorname{dist}_N(s, s_k^{\max}) \le \Delta_b$
因为 hard window 已经表达了"必须在 argmax 附近"，再加距离惩罚可能会重新把算法拉回 argmax。

这版的优势是比较干净：
$\text{amplitude} \rightarrow \text{define candidate region}$
$\text{phase} \rightarrow \text{choose final bin}$
而不是把 amplitude、rank、phase 全部混在一个分数里。

但是这里有一个必须先验证的实验：
$\Pr\left(\operatorname{dist}_N(s_k^{\mathrm{GT}}, s_k^{\max}) \le \Delta_b\right)$
也就是 GT-bin 是否真的经常落在 argmax 附近。
你可以画一个 histogram：
$\delta_k = \operatorname{dist}_N(s_k^{\mathrm{GT}}, s_k^{\max})$
如果大多数 $\delta_k \le 2$ 或 $\le 4$，那这个设计很稳。如果大量 GT-bin 离 argmax 很远，那 argmax-neighborhood gate 会把正确答案直接裁掉，这时就不能只围绕 argmax，而要改成 top-$P$ local peaks neighborhood。
所以第一版我建议这样做：
$\mathcal{S}_k = \mathcal{N}_{\Delta_b}(s_k^{\max})$
作为主实验；
然后做一个增强版：
$\mathcal{S}_k = \bigcup_{p=1}^{P} \mathcal{N}_{\Delta_b}(s_{k,p}^{\text{peak}})$
其中 $s_{k,p}^{\text{peak}}$ 是幅度排名前 $P$ 的局部峰。比如 $P = 3, \Delta_b = 1 \text{ or } 2$。
这样仍然不是全 Top-$L$ 乱搜，而是"围绕少数强峰附近做相位选择"。

一句话收敛：
原来的幅度加权分数改掉。幅度不进入最终打分，只用于构造候选区域；候选 bin 必须位于 argmax 或少数局部强峰附近，最终在局部候选中选择与 packet-level 相位线 residual 最小的 bin。这样既避免相位在全频域随机误锁，也避免幅度项反向压制相位判据。
这里存在隐患 ，我认为用弱幅度先验可能有反向提升，以及这个rank也是

5. 整包相位线怎么找  
可以从前面看到这个相位线特别重要，那该从哪里来呢？
本文不直接从所有 payload 候选 bin 中盲目搜索整包相位线，而是优先利用前导码、sync word 等已知符号构建高可信 anchor 集合。由于这些符号的理论 bin 已知，接收端可以在 dechirp+FFT 后直接读取对应 bin 的复数相位，并对该相位序列进行 unwrap 和线性拟合，从而得到初始 packet-level 相位趋势：
$\theta_k = \theta_0 + \omega k$
其中，$\theta_0$ 表示参考起点处的相位，$\omega$ 表示相邻符号之间的平均相位漂移。该相位漂移主要反映 residual CFO、STO/SFO 漂移以及帧同步残差对 payload 符号相位的连续影响。相比于直接在低 SNR payload 候选 bin 中搜索相位线，基于已知 anchor 的方式可以避免随机噪声 bin 形成虚假的相位一致性。
在 payload 解码阶段，初始相位线作为 phase-based bin choice 的先验，用于判断候选 bin 的相位是否与整包相位趋势一致。对于后续解出的符号，只有当其满足较高置信度条件——例如相位残差较小、最优候选与次优候选分数间隔较大、幅度排名不过低时——才将其加入可信符号集合，并作为 pseudo-anchor 参与相位线的保守重拟合。未通过置信度门控的符号只保留软信息，不参与相位模型更新。
因此，相位线的构建过程采用"已知 anchor 初始化，高置信 payload 符号逐步修正"的策略。当前阶段主要采用线性模型描述整包主相位趋势；未来如果实际实验中发现相位漂移存在更复杂的非线性变化，可进一步考虑分段线性模型、二次模型，或引入轻量级学习模块预测相位漂移，但该部分暂不作为当前方案的核心设计。
但是不好说，这个不知道合不合理啊
搞不好还不如直接通过前导码 SFD这些算出来呢，应该重新改成可靠的anchor和pseudo anchor，前者是前导码和SFD那些，后者可以是高置信度的符号

6.  找到相位线后怎么选 bin  
得到最优相位线 $(\theta_0^\star, \omega^\star)$后，每个符号独立选：
$\hat{s}_k = \arg\max_{s \in S_k} B_k(s; \theta_0^\star, \omega^\star)$
这一步就是最终 FFT bin choice。
注意这里的"独立选"是在整包相位线已经确定之后做的，不是传统 argmax 那种每个符号孤立判断。

7.  再做 2–3 轮 refinement  
初选出来的 $\hat{s}_k$ 可能有错误点。可以用它们重新拟合相位线。
先根据当前相位线做局部 unwrap：
$\tilde{\phi}_k = \theta_0^\star + \omega^\star k + \mathrm{wrap}\left(\angle Z_k(\hat{s}_k) - \theta_0^\star - \omega^\star k\right)$
然后对 $(k, \tilde{\phi}_k)$ 做加权线性拟合，更新 $\theta_k = \theta_0 + \omega k$。
权重可以设成：
$w_k = \exp\left(-|d_k(\hat{s}_k)|\right)$
或者更简单：phase residual 小，权重大；best 和 second-best score 差距大，权重大；幅度 rank 靠前，权重大。
然后重新选 bin，再重新拟合。做 2–3 轮就够。

8.  需要加一个 robust score，防止少数坏符号拖垮  
低 SNR 下某几个符号可能根本没救，所以总分不要简单 sum，可以用 trimmed sum。
比如每条相位线算出每个符号的 best score：
$q_k = \max_{s \in S_k} B_k(s; \theta_0, \omega)$
然后丢掉最差的 10%–20%：
$J_{\mathrm{trim}}(\theta_0, \omega) = \sum_{k \in \text{top 80\% scores}} q_k$
这样 $k=12$、$k=23$ 这种异常点不会把整条 phase line 拖偏。

9. 每个符号输出置信度，不要只输出硬判决
对最终每个 $k$，保留：
$s_{k,1} = \arg\max_s B_k(s), \quad s_{k,2} = \text{second best}$
定义 margin：
$M_k = B_k(s_{k,1}) - B_k(s_{k,2})$
再定义 phase residual：
$r_k = \left|\mathrm{wrap}\left(\angle Z_k(s_{k,1}) - \theta_0^\star - \omega^\star k\right)\right|$
如果 $M_k$ 大、$r_k$ 小，这个符号就是 high-confidence symbol。
如果 $M_k$ 小、$r_k$ 大，这个符号就不要强行相信，后面交给 Hamming / soft decoding / CRC 筛选。
这点和 Sym-FEC 的思路可以区分开：Sym-FEC 是靠 Hamming block 内的 XOR 关系构造 spectrum copies，并评估多个 guiding set 的 block-wide decoding result；你这里不是枚举 guiding set，而是先在 PHY 侧给每个符号产生 phase-profile likelihood。

10. 如果 framesync 不准，把 offset 也放进同一个评分函数
你刚才担心低 SNR framesync 不准，这个问题非常关键。
不要把 payload start 当成完全已知。可以枚举一个小范围时间偏移：
$\Delta \in \{-4, -3, \ldots, 4\}\ \text{samples}$
对每个 $\Delta$ 重新切 symbol、重新 FFT，得到 $Z_{k,\Delta}(s)$。
然后目标函数变成：
$J(\Delta, \theta_0, \omega) = \sum_k \max_{s \in S_{k,\Delta}} B_{k,\Delta}(s; \theta_0, \omega)$
最后选：
$(\Delta^\star, \theta_0^\star, \omega^\star) = \arg\max_{\Delta, \theta_0, \omega} J(\Delta, \theta_0, \omega)$
这个设计很重要，因为它让相位主趋势反过来服务于 framesync refine：哪个 payload 切分方式能让整包候选 bin 的相位最像同一条线，哪个切分方式就更可信。
这比先死切 payload 再强行选 bin 更稳。
这个说实话目前没必要考虑，我们做的场景是 解码比同步率先失效

11. 完整流程
可以按这个顺序实现：
1. 对每个候选 payload offset $\Delta$，切出 payload symbols
2. 每个 symbol 做 dechirp+FFT，保存复数 $Z_{k,\Delta}(s)$
3. 每个 $k$ 保留 Top-L 候选 bin，$L$ 先取 16 或 32
4. 在 $(\theta_0, \omega)$ 网格上搜索 packet-level phase line（这里我是真觉得可以先用前导码做为anchor算一根）
5. 对每条 line，计算每个符号 Top-L 里的最佳候选分数
6. 用 trimmed sum 选出最优 $(\Delta^\star, \theta_0^\star, \omega^\star)$
7. 固定最优 line，每个符号选最高分 bin
8. 用选出来的 bin 做 robust line refit
9. 重复 2–3 轮
10. 输出 hard bin + per-symbol confidence + Top-L likelihood

12. 这个算法最关键的实验验证
你现在马上应该跑三个指标。
第一，phase-line selector accuracy：
$\mathrm{acc} = \frac{1}{K} \sum_k \mathbf{1}[\hat{s}_k = s_k^{\mathrm{GT}}]$
和传统 argmax acc=0.06 对比。
第二，Top-L recall：
$\mathrm{Recall}@L = \frac{1}{K} \sum_k \mathbf{1}[s_k^{\mathrm{GT}} \in S_k]$
如果 Recall@16 只有 0.2，那算法没法救；如果 Recall@32 到 0.7/0.8，这个方向就很有希望。
第三，phase residual separation：
比较最终相位线下：
$r_k(s_k^{\mathrm{GT}}) \quad \text{和} \quad \min_{s \in S_k, s \neq s_k^{\mathrm{GT}}} r_k(s)$
如果 GT-bin residual 明显更小，那 phase trend 真的有判别力。

一句话总结：
你要搜的不是 bin 序列，而是相位主趋势；bin 是相位主趋势确定后，在每个符号 Top-L 候选里自然选出来的。第一版就做"Top-L 候选 + wrapped phase-line grid search + trimmed score + robust refit"，不要一上来做复杂 Viterbi。