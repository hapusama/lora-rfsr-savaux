# Phase-MAP 残差解码方法说明

这份说明把 `data/phase_guided/` 下面可复现实验当前使用的方法固定下来。这里不是要声称“原始相位在所有情况下都稳定”，更准确的说法是：

```text
LoRa dechirp 之后会得到一个包内局部相位坐标。
当 LoRa 编码结构和同一 session 的数据结构把 FFT bin 搜索空间缩小之后，
相位似然可以在幅度 argmax 失效后，把正确的残差轨迹区分出来。
```

## 解码器改动范围

baseline 的做法是：每个 symbol 都按 FFT 幅度最大值，也就是 amplitude argmax，选一个 FFT bin。

Phase-MAP 保留普通同步器和 explicit header 路径，只改弱 payload 的判决层：

1. 解出 explicit header；如果 header argmax 失败，就尝试 rescue。
2. 从强包里构建当前 session 的 payload 模板。
3. 枚举模板里仍未知的 residual byte 值。
4. 把每一个完整 byte 候选重新走一遍 gr-lora_sdr 的硬编码路径，重新编码成符号序列。这里也包括 explicit header block 后半部分继续流入 payload/CRC 的 tail nibbles。
5. 用幅度、phase-line 一致性、可选的 preamble in-chirp profile 证据、可选的 byte-model 先验给每个重编码候选打分。
6. 跨多个 packet 累积候选似然，并用 session trajectory 模型做联合判决。目前使用的是模 256 的 affine byte 规则。

关键设计点：相位不是直接拿来做一个全局的“phase -> symbol”规则，而是在 codec projection 之后使用。也就是说，先用编码结构把候选空间压小，再让相位负责区分候选。

## 候选似然

对候选 `c`，令 `K(c)` 表示重编码后已知 coded symbol 的 payload symbol 位置集合。令 `Z[k,b]` 表示 payload symbol `k` 在候选 bin `b` 上的复数 FFT 值，`theta_k` 表示包内 phase line 预测出的相位。

MAP 的相位项是：

```text
L_phase(c) = kappa / |K(c)| * sum_k cos(angle(Z[k, b_c,k]) - theta_k)
```

其中相位差会 wrap 到 `[-pi, pi]`。当前稳定设置是 `kappa = 2.0`。`kappa` 太大容易过度相信带噪相位；太小又会拿掉深弱 SNR 下的区分能力。

候选总分还包括两个部分：一是所选候选相位能否拟合出平滑 phase line，二是有界的 log-amplitude 项。在当前论文 artifact 里，导出 candidate 时关闭了 dynamic-byte prior：

```text
candidate-search-prior-weight = 0.0
```

这样做是为了证明 session trajectory 的贡献来自 joint 层，而不是每个 packet 单独用了“答案先验”。

## Preamble In-Chirp Profile Hook

代码里有一个可选的 preamble in-chirp phase-profile 打分项。做法是：把重复的 preamble upchirp 对齐到同一个 dechirp 坐标系，去掉它们共同的相位，再把剩下的 chip-wise 单位相量平均成一个 profile。如果某个候选 bin 经过这个 profile 校正后相干能量变强，就可以给它加分。

这个 hook 在当前 artifact 里只是辅助项。它不是直接照搬某篇 in-chirp unwrapping 论文的思路：这里的 profile 只是 codec-projected candidate scoring 里面的一个弱似然项，并且有质量门控。主结果不依赖这个 hook。

## 联合 Session 轨迹

对当前数据集，剩下的动态 byte 满足：

```text
byte_value(packet_index) = slope * packet_index + intercept mod 256
```

joint decoder 会搜索 `(slope, intercept)`，并最大化所有 packet 的候选分数之和。所以即使单个 packet 的 residual candidate 很模糊，app-level 结果仍然可能被跨包轨迹稳定地恢复出来。

当前主报告结果使用的是 hard affine 路径。代码里也有 outlier-tolerant affine decoding 作为鲁棒性扩展，但它必须做 confidence gate；如果惩罚太低，它会接收带噪的独立 top-1 candidate，反而破坏整条轨迹。

## 当前证据

紧凑 threshold 表由下面命令生成：

```text
scripts/reproduce_phase_map_paper_artifacts.py --skip-sweeps
```

当前 session 摘要：

| SNR dB | Argmax SER | MAP residual SER | Joint app |
|---:|---:|---:|---:|
| -10 | 0.0000 | 0.0000 | 5/5 |
| -15 | 0.0114 | 0.0000 | 5/5 |
| -20 | 0.4000 | 0.0400 | 5/5 |
| -23 | 0.7657 | 0.1086 | 5/5 |
| -25 | 0.9486 | 0.1657 | 5/5 |
| -27 | 0.9886 | 0.2000 | 5/5 |

用稀疏 SNR 点在 `SER = 0.1` 处做线性插值，可以得到当前 session 的门限估计：

```text
argmax crossing      -16.14 dB
Phase-MAP crossing   -22.62 dB
estimated gain         6.48 dB
```

这个结果应该保守表述为“当前 session 上的证据”，不要写成通用 LoRa 极限。

## 额外采集的 sanity check

另外两个 USRP_IQ capture 也用同一套 residual MAP scoring 路径做了测试。每个 SNR 点只取前 5 个 packet：

```text
data/phase_guided/paper_tables/generalization_capture_validation.csv
data/phase_guided/paper_tables/generalization_capture_validation.md
```

当前验证摘要：

| Capture | SNR dB | Argmax SER | MAP residual SER | Joint app |
|---|---:|---:|---:|---:|
| 0_0_0_10_14_8 | -20 | 0.2000 | 0.0286 | 5/5 |
| 0_0_0_10_14_8 | -23 | 0.6914 | 0.0800 | 5/5 |
| 0_0_0_10_14_32 | -20 | 0.4686 | 0.0914 | 5/5 |
| 0_0_0_10_14_32 | -23 | 0.8000 | 0.1943 | 3/5 |

这能防止过度拟合主 preamble-16 capture。它也暴露了一个边界情况：在 preamble-32、-23 dB 下，symbol SER 仍然显著下降，但 app-level affine trajectory 没有完全恢复。

## Phase Ablation

移除相位项，也就是设置 `kappa = 0`，在中等门限附近不一定会让恢复彻底失败，因为 codec projection 和 session accumulation 本身已经很强。但移除相位会降低模型 margin。在更深的弱 SNR 点，相位就变得必要：

```text
-25 dB: kappa=0 只能恢复 1/5，kappa>=0.5 可以恢复 5/5
-27 dB: kappa=0 和 0.5 只能恢复 1/5，kappa>=1.0 可以恢复 5/5
```

因此论文/报告里比较稳妥的表述应该是：

```text
Codec/session structure provides the search space.
Phase likelihood provides separation once amplitude evidence is insufficient.
```

翻成中文就是：

```text
编码结构和 session 结构提供搜索空间。
当幅度证据不足时，相位似然负责把正确轨迹分离出来。
```

## 复现命令

只从已有 sweep 结果重新生成论文表格：

```powershell
D:\mysoft2\miniconda3\envs\gr-lora\python.exe -B `
  "weakPacket_decoding copy\scripts\reproduce_phase_map_paper_artifacts.py" `
  --skip-sweeps
```

重新跑完整的当前 artifact pipeline：

```powershell
D:\mysoft2\miniconda3\envs\gr-lora\python.exe -B `
  "weakPacket_decoding copy\scripts\reproduce_phase_map_paper_artifacts.py"
```
