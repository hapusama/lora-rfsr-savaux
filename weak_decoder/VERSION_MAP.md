# weak_decoder 主线边界

## Active mainline

```text
synchronization/preamble_detector.py
synchronization/frame_locator.py
synchronization/grlora_frame_sync.py
decoding/header_first_demod.py
os_lora/
```

目标是先把真实 IQ 的前导码检测和 frame sync 做稳，再以同步后的过采样
branch 观测进入 OS-LoRa/GLS。`run_iq_frontend.py` 是标准 `.bin` 入口。

## Shared LoRa PHY

```text
chirp.py
decoding/payload_codec.py
branch4_profile.py
```

这些模块提供 chirp/FFT 基础、标准 LoRa PHY codec 和当前硬件实验参数。

## Preserved baselines

```text
baselines/
decoding/adaptive_path_demod.py
decoding/structured_path_demod.py
decoding/timing_path_demod.py
```

这些实现只用于对照、消融或诊断，不应静默成为 GLS 主算法的一部分。

## Retired research branches

下列方向已经退出活动源码：

```text
phase-guided payload demod
phase-consistency candidate reranking
symbol-phase two-stage selector
codec/CRC beam search
blind payload search
```

需要追溯时查看 `doc/history/`、`notes/` 或 Git 历史，不再从当前模块导入。
