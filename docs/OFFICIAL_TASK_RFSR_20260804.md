# 官方 OTA 两阶段 task-aware RF-SR 实验（2026-08-04）

## 最终结论

这轮已经把“有没有预训练”补齐，并尽量保留原 RF-SR 论文/代码真正有价值的部分：

1. 合法随机 PHY payload 的合成预训练；
2. 插值基线加学习残差的 polyphase 先验；
3. synthetic pretrain -> official OTA fine-tune 两阶段流程；
4. 低率输入、高率 clean/reference 监督的基本思想。

其余部分按当前 Savaux 解调任务重建：先在 1 MS/s 施加多径、CFO、STO、SFO、
gain/phase 和 AWGN，再严格抽取到 250 kS/s；loss 使用可微的 Savaux Eq. (36)-(37)
八支路相干频谱，而不是原先四支路功率相加。

训练本身成功，网络没有塌缩：hard 模型在 synthetic validation 上把缺失相位 waveform
loss 从 sinc 的 0.715 降到 0.239；官方 validation 上从 1.155 降到 0.867，Savaux
concentration 从 3.213 降到 2.918。可微 Torch Savaux 与项目 NumPy reference 的随机
波形逐频点最大误差为 `3.1e-6`，clean chirp 最大误差为 `6.4e-5`。

但最终 held-out test 的 hard decision 没有超过强插值：20 个 test reference、每个取最接近
-15 dB 的一个 OTA capture，共 800 个 symbols，插值、hard-RF-SR、soft-RF-SR 在 FFT、
Savaux、Savaux+GLS 下均为 `121/800`。网络改善了部分 peak margin，但 cluster bootstrap
和 paired sign-flip 没有 SER improvement。因此当前正确结论是：两阶段新方案可作为后续研究
checkpoint，但还不能替换主链的 `resample_poly` baseline。

## 对“论文是否这么做、现在是否有预训练”的明确回答

原 RF-SR 上游 OTA loader 不是“同一个 OTA capture 的 received-to-received”任务。其代码是：

```text
x = OTA received IQ 从 2 MS/s 抽到 250 kS/s
y = 配对 signalout reference 从 2 MS/s 抽到 1 MS/s
```

也就是 OTA -> clean/reference 监督。此前第一轮新网络为了满足硬样点不变量，改成了：

```text
x = 同一 OTA capture 的 250 kS/s 抽取
y = 同一 OTA capture 的 1 MS/s 抽取
```

这是我们的实验设计，不应说成论文原方法。第一轮 scratch checkpoint 也确实没有预训练。

现在的新方案有正式预训练，而且是架构匹配、checkpoint 可直接续训的预训练：

```text
随机合法 payload -> 1 MS/s 物理信道/噪声 -> 250 kS/s 输入
                    -> task-aware TCN -> clean 1 MS/s + Savaux task loss
                    -> official OTA received-target fine-tune
```

## 数据与防泄漏划分

官方数据位于：

```text
/root/autodl-tmp/.autodl/rfsr_ota_official
```

已解压并校验 2,768 个 `.cfile`：2,668 OTA、100 reference，解压后约 91.45 GiB，
压缩 SHA-256 和解压 MD5 失败均为 0。

使用 metadata reference/payload ID 作为 group，seed 42 做 60/20/20 划分；同一 payload
在不同地点、实验和 receiver gain 下的全部 capture 只能进入一个 split。过滤 `[-35, 15] dB`
后，train/validation/test 为 1,261/422/413 captures。训练 checkpoint 绑定完整 split manifest。

## 新网络和训练数据

主要实现：

```text
third_party/rfsr/rfsr/nn/task_model.py
third_party/rfsr/rfsr/nn/task_loss.py
third_party/rfsr/rfsr/nn/task_pretraining_dataset.py
third_party/rfsr/rfsr/nn/official_ota_dataset.py
tools/pretrain_task_rfsr.py
tools/train_official_task_rfsr.py
tools/evaluate_task_rfsr_symbol_metrics.py
tools/evaluate_official_task_rfsr.py
```

`TaskAwarePolyphaseTCN` 使用 32 channels 和 dilation
`1,2,4,8,16,32,64,128`，在 250 kS/s 网格运行。hard 版本 58,790 参数，只预测三个
缺失相位并逐点复制 phase 0；soft 版本 58,856 参数，对四个相位都能输出去噪残差。
输入只用自身 complex RMS 归一化，推理后恢复同一尺度。

插值先验从线性升级成 radius-16 Kaiser-windowed sinc。带限复正弦测试中，其 interior RMSE
比线性插值低约 47--6,500 倍，且 hard 模式仍严格满足
`output[..., ::4] == input`。

合成 dataset 的每个 item 先随机生成 20-byte payload，经 gr-lora-compatible PHY 编码取得
真实 header/payload symbol ID。每个 1 MS/s symbol 加入：

- 1--3 tap multipath；
- CFO `[-12, 12] kHz`；
- fractional STO `[-6, 6]` output samples；
- SFO `[-25, 25] ppm`；
- 随机 gain、carrier phase；
- SNR `[-24, 8] dB` 的 high-rate AWGN。

最后才执行 `x = noisy_high[::4]`。hard 模型以 clean channel target 监督三个缺失相位，
观测 phase 0 在 waveform loss 中 mask；soft 模型监督全部四相位。这样既保留论文 clean-target
预训练的价值，又消除旧方案里 x/y 不在同一物理坐标的问题。

## Savaux 相干 task loss

旧 scratch loss 把一个 32,768 点 symbol 拆成四个 8,192 点 branch，优化
`sum_q |FFT_q[k]|^2`。它不包含 branch 相对相位，和下游 Savaux/GLS 不一致。

新 loss 直接实现：

1. 1 MS/s symbol 按 OSR=8 拆成 8 个 4,096 点 branch；
2. 对 q>0 使用 Savaux Eq. (36) wrap-tail chirp-z correction；
3. 用 `exp(-j 2 pi q k / (N R))` 做 Eq. (37) coherent sum；
4. 在 combined complex spectrum 上计算 correct-region concentration 和 peak margin。

总 loss 为 complex Charbonnier waveform 加 concentration/margin。所有 FFT 分支保持可微，
并有 NumPy parity 和 gradient 测试。

## 两阶段训练结果

### Hard-observed 主实验

合成预训练：

```text
/root/autodl-tmp/rfsr-run/task_aware_pretrain_physical_savaux_seed42/best.pt
SHA256 eddc3d50a15a919157b1e6f7b764937f025e2774d3395a9edabaeeee2b8c7371
```

1,000 train items、200 fixed validation items、每 item 4 symbols、8 epochs，用时 170.6 s。
validation total/waveform 为 `0.3395/0.2392`，sinc waveform baseline 为 `0.7150`，硬样点
最大误差始终为 0。

官方 OTA 微调：

```text
/root/autodl-tmp/rfsr-run/task_aware_official_pretrained_savaux_seed42/best.pt
SHA256 a6f5398b90d9fd457a2f6047d9adc6b1686e78813c8a588a85b45a00f89aff25
```

8 epochs 用时 217.6 s。160-capture validation 上：

| 方法 | waveform | Savaux concentration | margin loss | exact-bin acc. | within-1 acc. |
| --- | ---: | ---: | ---: | ---: | ---: |
| linear | 1.0402 | 3.1898 | 0.5455 | 65.17% | 89.66% |
| sinc | 1.1554 | 3.2129 | 0.5527 | 65.52% | 89.66% |
| 旧 scratch | 0.8937 | 2.9778 | **0.4819** | **73.10%** | 90.00% |
| pretrain only | 0.8714 | 3.0678 | 0.5632 | 63.79% | 89.14% |
| pretrain + OTA | **0.8670** | **2.9183** | 0.5094 | 66.38% | **90.00%** |

窗口级 loss 改善不等于整包解码改善，所以后续只把它作为 gate。

### Soft-observed 消融

soft pretrain 和 OTA checkpoint：

```text
/root/autodl-tmp/rfsr-run/task_aware_pretrain_soft_savaux_seed42/best.pt
SHA256 4fde0d526140561b5c998fa7a7d58c190632e691a93dacc86b13b5a4239fd989

/root/autodl-tmp/rfsr-run/task_aware_official_soft_savaux_seed42/best.pt
SHA256 b84e20fddf54579eae95cc37d4a461cb17a14121064833e4477fb055d26a8ae9
```

soft synthetic/OTA 各 8 epochs，用时 174.6/197.7 s。OTA validation total 为 1.0304，
略优于 hard 的 1.0384，但 strength=1 会破坏整包同步。validation 预先选择 strength=0.5：
4/4 同步、三 decoder 0/160，并提高 FFT/Savaux margin；之后才进入 test。

## 端到端对比实验

### 旧 scratch 8-reference 筛查

| Decoder | 插值 | 旧 scratch |
| --- | ---: | ---: |
| FFT | 23/320 | 24/320 |
| Savaux | 23/320 | 23/320 |
| Savaux+GLS | 23/320 | 37/320 |

### hard 两阶段模型，strength=0.5，全部 test reference

选择每个 test reference 最接近 -15 dB 的一个 capture；reference 99 只有 -8.41 dB 可用。
共 20 clusters、20 captures、800 symbols。参数先在 validation 锁定，未按 test 调整。

| Decoder | 插值错误 | hard RF-SR 错误 | 插值 margin | RF-SR margin |
| --- | ---: | ---: | ---: | ---: |
| FFT | 121/800 | 121/800 | 4.899 dB | **5.448 dB** |
| Savaux | 121/800 | 121/800 | **7.414 dB** | 7.345 dB |
| Savaux+GLS | 121/800 | 121/800 | **7.447 dB** | 7.325 dB |

两者均同步 18/20。三个 decoder 的 candidate-baseline SER difference 都为 0，cluster
bootstrap 95% CI 均 `[0,0]`，不存在显著 improvement。

完整结果：

```text
/root/autodl-tmp/rfsr-run/task_aware_official_pretrained_savaux_seed42/eval_test20_alpha050.json
```

### soft 两阶段模型，strength=0.5

| Decoder | 插值错误 | soft RF-SR 错误 | 插值 margin | RF-SR margin |
| --- | ---: | ---: | ---: | ---: |
| FFT | 121/800 | 121/800 | 4.899 dB | **5.289 dB** |
| Savaux | 121/800 | 121/800 | 7.414 dB | **7.439 dB** |
| Savaux+GLS | 121/800 | 121/800 | **7.447 dB** | 7.155 dB |

同样同步 18/20，hard decisions 完全持平。完整结果：

```text
/root/autodl-tmp/rfsr-run/task_aware_official_soft_savaux_seed42/eval_test20_alpha050.json
```

## 被验证后否决的方案

1. **无预训练 scratch**：Savaux 不改善，GLS 从 23 错误恶化到 37。
2. **exact-bin 强监督**：窗口 exact-bin 66.4% -> 77.4%，但整包同步降到 7/8，Savaux
   变成 63/320；说明局部 CFO/bin 标签过拟合会破坏 packet synchronization。
3. **soft strength=1**：validation 只同步 3/4，不能进入 test。
4. **sync strength=0.5 / decode strength=1**：虽保持 4/4 同步，但 validation FFT 产生
   12/160 错误且 margin 下降；不能用强去噪只替换 data 解调。

## 如何复现最佳 hard 流程

```bash
cd /root/lora-rfsr-savaux

python tools/pretrain_task_rfsr.py \
  --run-dir /root/autodl-tmp/rfsr-run/task_aware_pretrain_physical_savaux_seed42 \
  --train-items 1000 --validation-items 200 --symbols-per-item 4 \
  --epochs 8 --batch-size 2 --workers 4 --seed 42

python tools/train_official_task_rfsr.py \
  --official-root /root/autodl-tmp/.autodl/rfsr_ota_official \
  --run-dir /root/autodl-tmp/rfsr-run/task_aware_official_pretrained_savaux_seed42 \
  --pretrained /root/autodl-tmp/rfsr-run/task_aware_pretrain_physical_savaux_seed42/best.pt \
  --epochs 8 --waveform-warmup-epochs 1 \
  --symbols-per-capture 4 --batch-size 2 --workers 4 \
  --validation-capture-limit 160 --seed 42 --split-seed 42

python tools/evaluate_official_task_rfsr.py \
  --official-root /root/autodl-tmp/.autodl/rfsr_ota_official \
  --checkpoint /root/autodl-tmp/rfsr-run/task_aware_official_pretrained_savaux_seed42/best.pt \
  --output /root/autodl-tmp/rfsr-run/task_aware_official_pretrained_savaux_seed42/eval_test20_alpha050.json \
  --split test --captures-per-reference 1 \
  --selection target_snr --selection-snr-db -15 \
  --residual-strength 0.5 --sync-mode common_native
```

## 下一步研究判断

250 kS/s 对 125 kHz complex-baseband LoRa 已不低于其带宽采样需求，强 polyphase 插值本身
保留了大部分可用信息；网络无法凭空恢复独立噪声 realization。当前 hard/soft 两条路线都显示，
网络更容易改变 peak margin，而不改变最终 symbol decision。若继续投入，应把问题从“生成更漂亮的
1 MS/s IQ”改成 decoder-native uncertainty/LLR 学习，或训练可证明改善 packet synchronization
与 CRC/PER 的联合目标；仅扩大 TCN、epoch 或窗口级 loss 权重，不再是有证据支持的下一步。
