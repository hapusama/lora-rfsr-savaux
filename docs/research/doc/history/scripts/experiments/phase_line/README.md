# Phase-Line Experiments

这里存放“相位线构建/可靠性验证”相关实验脚本。

这些脚本暂时不做 FFT bin 重选，也不改主解码链。当前目标是先回答：

```text
只用 preamble / known anchors 能不能构造 packet-level phase line？
这根 phase line 外推到 payload GT bin 后 residual 是否足够稳定？
```

入口脚本：

```text
run_preamble_phase_line_experiment.py
```

