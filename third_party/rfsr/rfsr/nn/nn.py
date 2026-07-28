"""RF-SR 模型定义与训练入口。

不带 ``--ota`` 时，脚本会调用 ``SyntheticLoRaDataset`` 在线合成 LoRa
packet，并完成论文中的第一阶段合成预训练；该路径不读取离线 IQ 文件。
带 ``--ota`` 时才会进入真实采集数据微调路径，但公开仓库中的 OTA loader
仍依赖未随代码发布的文件筛选、数据切片和可复现划分辅助函数。

注意：合成数据集只在启动时生成一次，之后每个 epoch 重复使用同一批样本。
另外，只要目标 checkpoint 文件名已经存在，脚本就会自动加载并继续训练，
因此使用仓库自带模型的同名参数并不代表“从零开始预训练”。
"""

import os
import re
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rfsr.nn import (
    OTALoRaDataset,
    ReferencePhyPretrainingDataset,
    SyntheticLoRaDataset,
)
from rfsr.interp import resample_poly_torch_batch2

os.environ[
    "PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # This helps to prevent GPU memory fragementation (relevant for model0v2 and model3


# --- Complex 1D Convolutional Layer (Re+Im as 2 channels) ---
class ComplexConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels * 2,  # I and Q components
            out_channels * 2,
            *args,
            **kwargs
        )

    def forward(self, x):
        # x: [B, 2*in_channels, L]
        return self.conv(x)


class SimpleComplexCNN0(nn.Module): # run model on interpolated input
    def __init__(self, oversampling=1, model="model0v0"):
        super().__init__()
        # self.upsample = # nn.Upsample(scale_factor=oversampling, mode='linear', align_corners=False) # we use our custom interpolation
        self.oversampling = oversampling
        self.model = model
        self.gated = False
        self.residual_vec = None  # only to extract it for visualization purpose
        self.x_interp_vec = None  # only to extract it for visualization purpose

        # quantization
        self.quant = torch.ao.quantization.QuantStub()  # The door to Int8
        self.dequant = torch.ao.quantization.DeQuantStub()  # The exit to Float

        if model.__contains__("v0"):
            self.residual_net = nn.Sequential(
                ComplexConv1D(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                ComplexConv1D(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                ComplexConv1D(32, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                ComplexConv1D(16, 1, kernel_size=3, padding=1)
            )
        elif model.__contains__("v4"):  # add dropout
            self.residual_net = nn.Sequential(
                ComplexConv1D(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(p=0.3),  # <-- Add Dropout after activation
                ComplexConv1D(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(p=0.3),  # <-- Add Dropout after activation
                ComplexConv1D(32, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(p=0.3),  # <-- Add Dropout after activation
                ComplexConv1D(16, 1, kernel_size=3, padding=1)
            )
        elif model.__contains__("v0gated"):
            self.residual_net = nn.Sequential(
                ComplexConv1D(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                ComplexConv1D(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                ComplexConv1D(32, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                ComplexConv1D(16, 1, kernel_size=3, padding=1)
            )
            # 2. The SNR-based gating network
            #    This is a simple MLP that takes 1 scalar (SNR)
            #    and outputs 1 scalar (gate)
            self.gating_net = nn.Sequential(
                nn.Linear(1, 16),  # Takes 1 SNR value in
                nn.ReLU(),
                nn.Linear(16, 1),  # Outputs 1 gate value
                nn.Sigmoid()  # Squashes output between 0 and 1
            )
            print(f"Running gated model")
            self.gated = True
        elif model.__contains__("v1"):
            self.residual_net = nn.Sequential(
                ComplexConv1D(1, 16, kernel_size=7, padding=3),  # larger kernel
                nn.ReLU(),
                ComplexConv1D(16, 32, kernel_size=3, padding=2, dilation=2),  # dilated
                nn.ReLU(),
                ComplexConv1D(32, 16, kernel_size=3, padding=4, dilation=4),  # dilated
                nn.ReLU(),
                ComplexConv1D(16, 1, kernel_size=3, padding=1)
            )

        elif model.__contains__("v2"):
            self.residual_net = nn.Sequential(
                # First layer: larger kernel for more context
                ComplexConv1D(1, 32, kernel_size=15, padding=7),
                nn.ReLU(),

                # Dilated stack to expand receptive field quickly
                ComplexConv1D(32, 64, kernel_size=3, padding=2, dilation=2),
                nn.ReLU(),
                ComplexConv1D(64, 64, kernel_size=3, padding=4, dilation=4),
                nn.ReLU(),
                ComplexConv1D(64, 64, kernel_size=3, padding=8, dilation=8),
                nn.ReLU(),

                # Bottleneck-style compression/expansion
                ComplexConv1D(64, 128, kernel_size=1),  # expand channels
                nn.ReLU(),
                ComplexConv1D(128, 64, kernel_size=1),  # compress
                nn.ReLU(),

                # Final mapping back to 1 channel
                ComplexConv1D(64, 1, kernel_size=3, padding=1)
            )
        else:
            raise RuntimeError("Not implemented")

    def forward(self, x, snr_value):
        # x: [B, 2, L] expected (real & imag as separate channels)
        if x.dim() == 4:
            # If input is [B, 1, 2, L], squeeze and permute to [B, 2, L]
            x = x.squeeze(1)  # [B, 2, L]

        x_interp = resample_poly_torch_batch2(x, self.oversampling, 1)  # x_interp = self.upsample(x)  # [B, 2, OSF*L]

        # start quantization
        x_interp_quant = self.quant(x_interp)
        # Run in Int8

        # if self.model.__contains__("v2"): # this saves GPU memory, but is slower..
        #     # Add this import at the top of your .py file
        #     from torch.utils.checkpoint import checkpoint
        #     # --- MODIFIED LINE ---
        #     # Instead of: residual = self.residual_net(x_interp)
        #     # We use checkpoint() to avoid storing intermediate activations
        #     # use_reentrant=False is the modern, more efficient implementation
        #     residual = checkpoint(self.residual_net, x_interp, use_reentrant=False)  # [B, 2, OSF*L]
        #     # --- END MODIFICATION ---
        # else:
        #     residual = self.residual_net(x_interp)
        out = self.residual_net(x_interp_quant)  # [B, 2, OSF*L]

        # end quantization
        residual = self.dequant(out)

        self.residual_vec = residual
        self.x_interp_vec = x_interp
        if self.gated:
            # 2. Predict the gate from the SNR
            #    gate shape: (B, 1)
            raise RuntimeError("Re-activate gating-net")
            # gate = self.gating_net(snr_value)
            # return x_interp + (gate.unsqueeze(-1) * residual)

        return x_interp + residual


class SimpleComplexCNN1(nn.Module): # run NN on non-interpolated input
    def __init__(self, oversampling=1, model="model1v0"):
        super().__init__()
        # self.upsample = # nn.Upsample(scale_factor=oversampling, mode='linear', align_corners=False)
        self.oversampling = oversampling

        if model.__contains__("v0"):
            self.residual_net = nn.Sequential(
                ComplexConv1D(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                ComplexConv1D(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                ComplexConv1D(32, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                # instead of outputting 1 complex channel, output OSF channels
                ComplexConv1D(16, oversampling, kernel_size=3, padding=1)
            )
        elif model.__contains__("v1"):
            self.residual_net = nn.Sequential(
                ComplexConv1D(1, 16, kernel_size=7, padding=3),  # larger kernel
                nn.ReLU(),
                ComplexConv1D(16, 32, kernel_size=3, padding=2, dilation=2),  # dilated
                nn.ReLU(),
                ComplexConv1D(32, 16, kernel_size=3, padding=4, dilation=4),  # dilated
                nn.ReLU(),
                # instead of outputting 1 complex channel, output OSF channels

                ComplexConv1D(16, oversampling, kernel_size=3, padding=1)
            )

        elif model.__contains__("v2"):
            self.residual_net = nn.Sequential(
                # First layer: larger kernel for more context
                ComplexConv1D(1, 32, kernel_size=15, padding=7),
                nn.ReLU(),

                # Dilated stack to expand receptive field quickly
                ComplexConv1D(32, 64, kernel_size=3, padding=2, dilation=2),
                nn.ReLU(),
                ComplexConv1D(64, 64, kernel_size=3, padding=4, dilation=4),
                nn.ReLU(),
                ComplexConv1D(64, 64, kernel_size=3, padding=8, dilation=8),
                nn.ReLU(),

                # Bottleneck-style compression/expansion
                ComplexConv1D(64, 128, kernel_size=1),  # expand channels
                nn.ReLU(),
                ComplexConv1D(128, 64, kernel_size=1),  # compress
                nn.ReLU(),

                # Final mapping back to 1 channel
                # instead of outputting 1 complex channel, output OSF channels
                ComplexConv1D(64, oversampling, kernel_size=3, padding=1)
            )

    def forward(self, x):
        # x: [B, 2, L] expected (real & imag as separate channels)
        if x.dim() == 4:
            # If input is [B, 1, 2, L], squeeze and permute to [B, 2, L]
            x = x.squeeze(1)  # [B, 2, L]

        x_interp = resample_poly_torch_batch2(x, self.oversampling, 1)  # x_interp = self.upsample(x)  # [B, 2, OSF*L]

        residual = self.residual_net(x)  # [B, 2, OSF*L]
        # Reshape: group the OSF into the length dimension
        B, _, L = residual.shape
        residual = residual.view(B, 2, self.oversampling * L)  # [B, 2, L*OSF]

        return x_interp + residual


class ComplexResidualBlock(nn.Module):
    def __init__(self, in_c, hidden_c, skip_c, kernel_size=3, dilation=1, causal=False):
        """
        in_c/hidden_c/skip_c are *complex* channel counts (ComplexConv1D convention).
        """
        super().__init__()
        self.causal = causal
        pad = (kernel_size - 1) * dilation if causal else (kernel_size - 1) // 2 * dilation

        self.conv1 = ComplexConv1D(in_c, hidden_c, kernel_size=kernel_size,
                                   padding=pad, dilation=dilation)
        self.act1 = nn.ReLU()
        self.conv2 = ComplexConv1D(hidden_c, in_c, kernel_size=1)  # channel mixer back to in_c

        # Skip path projects hidden features to skip channels
        self.skip = ComplexConv1D(hidden_c, skip_c, kernel_size=1)

        # Optional 1x1 to match residual dims if needed (here in_c -> in_c, so identity)
        self.proj_res = nn.Identity()

        # Second conv for skip path should “see” conv1’s hidden
        self.post_hidden = ComplexConv1D(in_c, hidden_c, kernel_size=1)

    def forward(self, x):
        """
        x: [B, 2*in_c, L] internally (but you pass as [B, 2, L] when in_c=1).
        Returns:
            y: residual output (same complex channels as input)
            s: skip features (skip_c complex channels)
        """
        # First conv (dilated)
        h = self.conv1(x)
        h = self.act1(h)

        # Skip features from hidden
        s = self.skip(h)

        # Residual branch
        y = self.conv2(h)  # back to in_c complex channels
        y = y + self.proj_res(x)
        return y, s


def complex_channel_to_time(x, osf):
    """
    x: [B, 2*OSF, L]  (i.e., OSF complex channels)
    -> [B, 2, L*OSF]
    """
    B, C, L = x.shape
    assert C % 2 == 0, "Channel count must be even (real/imag pairs)."
    assert (C // 2) % osf == 0 or (C == 2 * osf), "Expecting exactly OSF complex channels."
    # x is arranged as [real0, imag0, real1, imag1, ..., real(OSF-1), imag(OSF-1)]
    x = x.view(B, 2, osf, L)  # [B, 2, OSF, L]
    x = x.permute(0, 1, 3, 2)  # [B, 2, L, OSF]
    x = x.reshape(B, 2, L * osf)  # [B, 2, L*OSF]
    return x


class ResidualTCN(nn.Module):
    def __init__(
            self,
            in_c=1,  # complex channels in (1 = IQ)
            base_c=64,  # hidden complex channels
            skip_c=64,  # skip complex channels
            kernel_size=3,
            num_stacks=3,
            layers_per_stack=5,  # dilations per stack: 1,2,4,8,16
            osf=4,
            causal=False,
            model="model2"
    ):
        super().__init__()
        self.osf = osf
        self.causal = causal

        # Initial “stem” to lift features
        self.stem = ComplexConv1D(in_c, base_c, kernel_size=7 if not causal else 3,
                                  padding=(7 - 1) // 2 if not causal else 0)

        # Build residual stacks with exponentially increasing dilation
        blocks = []
        for s in range(num_stacks):
            for l in range(layers_per_stack):
                d = 2 ** l
                blocks.append(
                    ComplexResidualBlock(
                        in_c=base_c,
                        hidden_c=base_c,
                        skip_c=skip_c,
                        kernel_size=kernel_size,
                        dilation=d,
                        causal=causal
                    )
                )
        self.blocks = nn.ModuleList(blocks)

        # Skip aggregator -> OSF complex channels
        self.skip_act = nn.ReLU()
        self.head = nn.Sequential(
            ComplexConv1D(skip_c, base_c, kernel_size=1),
            nn.ReLU(),
            ComplexConv1D(base_c, osf, kernel_size=1)  # OSF complex channels
        )

    @staticmethod
    def _sum_skips(acc, s):
        return s if acc is None else acc + s

    def forward(self, x):
        """
        x: [B, 2, L]  (1 complex channel represented as 2 real planes)
        returns: [B, 2, L*OSF]
        """
        h = self.stem(x)  # [B, 2*base_c, L]
        skip_acc = None
        for blk in self.blocks:
            h, s = blk(h)  # h: [B, 2*base_c, L], s: [B, 2*skip_c, L]
            skip_acc = self._sum_skips(skip_acc, s)

        z = self.skip_act(skip_acc)
        z = self.head(z)  # [B, 2*OSF, L]
        y = complex_channel_to_time(z, self.osf)  # [B, 2, L*OSF]
        # return y

        x_interp = resample_poly_torch_batch2(x, self.osf, 1)
        return x_interp + y


class LoRaResidualTCN(nn.Module):
    def __init__(self, osf=4, base_c=96, skip_c=96, kernel_size=3, causal=False, model="model3v0"):
        """
        LoRa SF12, BW=125k, fs=1e6 -> N_sym ~ 32768.
        Use a single long stack with 15 layers, dilations 1..2^14.
        """
        super().__init__()
        self.osf = osf

        if model.__contains__("v0"):
            self.num_layers = 15  # default
        elif model.__contains__("v1"):
            self.num_layers = 10  # shallower
        elif model.__contains__("v2"):
            self.num_layers = 20  # deeper
        else:
            raise RuntimeError("Invalid version")

        self.stem = ComplexConv1D(1, base_c, kernel_size=15, padding=7 if not causal else 0)
        blocks = []
        for l in range(self.num_layers):  # 15 layers -> RF > 32768 (k=3, doubling dilation)
            d = 2 ** l
            blocks.append(
                ComplexResidualBlock(
                    in_c=base_c,
                    hidden_c=base_c,
                    skip_c=skip_c,
                    kernel_size=kernel_size,
                    dilation=d,
                    causal=causal
                )
            )
        self.blocks = nn.ModuleList(blocks)

        self.skip_act = nn.ReLU()
        self.head = nn.Sequential(
            ComplexConv1D(skip_c, base_c, kernel_size=1),
            nn.ReLU(),
            ComplexConv1D(base_c, osf, kernel_size=1)  # OSF complex channels
        )

    def _compute_tcn_output(self, h):
        """Helper function containing the TCN logic to be checkpointed."""
        skip_acc = None
        for blk in self.blocks:
            h, s = blk(h)
            skip_acc = s if skip_acc is None else skip_acc + s

        z = self.skip_act(skip_acc)
        z = self.head(z)
        y = complex_channel_to_time(z, self.osf)
        return y

    def forward(self, x):
        """
        x: [B, 2, L]
        -> [B, 2, L*OSF]
        """
        # Part 1: Initial stem
        h = self.stem(x)

        # Part 2: TCN computation (Checkpointing applied here)
        # Use a lambda to pass the necessary arguments (like self) to the helper function
        # This avoids storing intermediate activations of the TCN layers.
        # use_reentrant=False is the modern, memory-efficient implementation.
        from torch.utils.checkpoint import checkpoint
        y = checkpoint(
            self._compute_tcn_output,
            h,
            use_reentrant=False
        )

        # Part 3: Residual connection
        x_interp = resample_poly_torch_batch2(x, self.osf, 1)

        return x_interp + y


def parse_args():
    """解析训练参数；未传 ``--ota`` 就使用合成数据集。"""
    parser = argparse.ArgumentParser(description="Train a model on IQ data")
    # dataset_size 是本次启动时生成的唯一 packet 数量，不是“每个 epoch
    # 重新生成”的数量。README 推荐 250 个 packet、重复训练 100 个 epoch。
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--dataset_size', type=int, default=250, help='Total number of samples in the dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    # 论文配置：2 MSPS / DSF=8 -> 250 kSPS 输入，再按 OSF=4 -> 1 MSPS 输出。
    parser.add_argument('--osf', type=int, default=4, help='Oversampling factor')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')  # typically: 1e-5 to 1e-3
    parser.add_argument('--model', type=str, default='v0', help='Model architecture to use')
    parser.add_argument('--optimizer', type=str, default='adam', choices=["adam", "adamw", "sgd"], help='Optimizer')
    # --ota 只负责把数据源切换为 OTALoRaDataset，不会自动执行连续 IQ
    # 切片，也不会自动加载合成预训练权重。
    parser.add_argument("--ota", action="store_true", help="Enable OTA (set to True if provided)")
    parser.add_argument("--dsf", type=int, default=8,
                        help="Down sampling factor (used for OTA, Fs=2e6, we go to /8 -> 0.25e6 by default")
    parser.add_argument(
        "--synthetic-source",
        choices=("reference_phy", "upstream"),
        default="reference_phy",
        help=(
            "合成预训练波形来源。reference_phy 使用 metadata 中的 PHY "
            "配置在线生成 raw 33-byte payload；upstream 保留官方私有头路径。"
        ),
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path(__file__).resolve().parents[4] / "data" / "reference_phy",
        help=(
            "generate_reference_phy.py 的 output root；其中必须包含 "
            "reference/ 和 metadata/。"
        ),
    )
    parser.add_argument(
        "--synthetic-seed",
        type=int,
        default=42,
        help="reference_phy 随机 payload 和 AWGN 的随机种子。",
    )
    parser.add_argument(
        "--fixed-reference-payloads",
        action="store_true",
        help=(
            "关闭启动时随机 payload，改为读取 reference/ 中已有的固定 cfile；"
            "仅用于对照。"
        ),
    )
    parser.add_argument(
        "--ota-root",
        type=Path,
        default=(
            Path(__file__).resolve().parents[4]
            / "data"
            / "reference_phy"
            / "rfsr_db"
        ),
        help=(
            "Manifest OTA dataset root generated by "
            "tools/build_rfsr_ota_dataset.py."
        ),
    )
    parser.add_argument(
        "--ota-target",
        choices=("received", "reference"),
        default="received",
        help=(
            "OTA 高采样标签。received 使用同一接收 OTA 波形（严格 RF-SR，"
            "默认）；reference 仅保留给旧的 received-to-ideal 对照实验。"
        ),
    )
    parser.add_argument(
        "--ota-max-groups",
        type=int,
        default=None,
        help=(
            "微调时最多使用多少个物理 OTA 包；按 seed 先抽取后固定 6:2:2 "
            "划分。省略时使用全部物理包。"
        ),
    )
    parser.add_argument(
        "--ota-split-seed",
        type=int,
        default=42,
        help="OTA 物理包 6:2:2 划分与可选抽样的随机种子。",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=10,
        help="连续多少个 epoch 验证 loss 未改善后停止，默认 10。",
    )
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=None,
        help=(
            "首次训练当前 checkpoint 时加载的预训练权重；若当前 checkpoint "
            "已存在，则优先续训当前 checkpoint。"
        ),
    )
    parser.add_argument(
        "--snr-min-db",
        type=float,
        default=-22.0,
        help="合成预训练随机 SNR 下限，默认 -22 dB。",
    )
    parser.add_argument(
        "--snr-max-db",
        type=float,
        default=10.0,
        help="合成预训练随机 SNR 上限，默认 10 dB。",
    )
    parser.add_argument(
        "--snr-db",
        "--snr",
        type=float,
        default=None,
        help="使用固定 SNR；传入后覆盖 --snr-min-db/--snr-max-db。",
    )
    parser.add_argument(
        "--cfo-min-hz",
        type=float,
        default=-35_000.0,
        help="reference_phy 输入 x 的随机 CFO 下限，默认 -35000 Hz。",
    )
    parser.add_argument(
        "--cfo-max-hz",
        type=float,
        default=35_000.0,
        help="reference_phy 输入 x 的随机 CFO 上限，默认 35000 Hz。",
    )
    parser.add_argument(
        "--cfo-hz",
        type=float,
        default=None,
        help="为输入 x 使用固定 CFO；标签 y 保持零 CFO。",
    )
    parser.add_argument(
        "--sto-initial-min-chips",
        type=float,
        default=-0.5,
        help="输入 x 的初始 STO 下限，单位 chip，默认 -0.5。",
    )
    parser.add_argument(
        "--sto-initial-max-chips",
        type=float,
        default=0.5,
        help="输入 x 的初始 STO 上限，单位 chip，默认 0.5。",
    )
    parser.add_argument(
        "--sto-slope-min-chips-per-symbol",
        type=float,
        default=-0.05,
        help="STO 逐符号变化斜率下限，默认 -0.05 chip/symbol。",
    )
    parser.add_argument(
        "--sto-slope-max-chips-per-symbol",
        type=float,
        default=0.05,
        help="STO 逐符号变化斜率上限，默认 0.05 chip/symbol。",
    )
    parser.add_argument(
        "--no-sto",
        action="store_true",
        help="关闭输入 x 的 STO 和逐符号漂移，用于消融。",
    )
    args = parser.parse_args()
    if args.snr_db is None and args.snr_min_db > args.snr_max_db:
        parser.error("--snr-min-db must be <= --snr-max-db")
    if args.cfo_hz is None and args.cfo_min_hz > args.cfo_max_hz:
        parser.error("--cfo-min-hz must be <= --cfo-max-hz")
    if args.sto_initial_min_chips > args.sto_initial_max_chips:
        parser.error(
            "--sto-initial-min-chips must be <= --sto-initial-max-chips"
        )
    if (
        args.sto_slope_min_chips_per_symbol
        > args.sto_slope_max_chips_per_symbol
    ):
        parser.error(
            "--sto-slope-min-chips-per-symbol must be <= "
            "--sto-slope-max-chips-per-symbol"
        )
    if args.ota_max_groups is not None and args.ota_max_groups < 3:
        parser.error("--ota-max-groups must be at least 3")
    if args.early_stop_patience < 1:
        parser.error("--early-stop-patience must be positive")
    return args


def build_synthetic_dataset(
    *,
    source,
    reference_root,
    oversampling,
    size,
    downsampling,
    sf,
    bandwidth,
    seed,
    snr_range,
    random_payload,
    cfo_range_hz,
    sto_enabled,
    sto_initial_range_chips,
    sto_slope_range_chips_per_symbol,
):
    """根据命令行选择官方或 raw-frame 合成预训练后端。"""

    if source == "upstream":
        return SyntheticLoRaDataset(
            oversampling=oversampling,
            size=size,
            downsampling=downsampling,
            SF=sf,
            BW=bandwidth,
            snr_range=snr_range,
        )

    label_rate_hz = 2_000_000 * int(oversampling) / int(downsampling)
    if not float(label_rate_hz).is_integer():
        raise ValueError(
            "2 MSPS * OSF / DSF must produce an integer label sample rate; "
            f"got OSF={oversampling}, DSF={downsampling}."
        )
    return ReferencePhyPretrainingDataset(
        reference_root=reference_root,
        oversampling=int(oversampling),
        size=int(size),
        expected_sample_rate_hz=int(label_rate_hz),
        expected_sf=int(sf),
        expected_bandwidth_hz=int(bandwidth),
        seed=int(seed),
        snr_range=snr_range,
        random_payload=bool(random_payload),
        cfo_range_hz=cfo_range_hz,
        sto_enabled=bool(sto_enabled),
        sto_initial_range_chips=sto_initial_range_chips,
        sto_slope_range_chips_per_symbol=(
            sto_slope_range_chips_per_symbol
        ),
    )


def load_eval_model(model_name):
    # extract params, parse with regex
    pattern = r"model_(?P<model>\w+)_bs(?P<batch_size>\d+)_osf(?P<osf>\d+)_ds(?P<dataset_size>\d+)_lr(?P<lr>[0-9.]+)_wd(?P<wd>[0-9.e-]+)"
    match = re.match(pattern, model_name)
    if match:
        params = match.groupdict()
        # convert to correct types
        BATCH_SIZE = int(params["batch_size"])
        OSF = int(params["osf"])
        # NUM_EPOCHS = int(params["num_epochs"])
        DATASET_SIZE = int(params["dataset_size"])
        LR = float(params["lr"])
        WEIGHT_DECAY = float(params["wd"])
        MODEL = params["model"]
        print(params)

    # 1. Recreate the model with the same architecture
    # --- Train Loop (minimal) ---
    if MODEL.__contains__("model0"):
        model = SimpleComplexCNN0(oversampling=OSF, model=MODEL)
    elif MODEL.__contains__("model1"):
        model = SimpleComplexCNN1(oversampling=OSF, model=MODEL)
    elif MODEL.__contains__("model2"):  # seems to end with NAN -> try lowering LR, i.e. try lr=1e-4
        model = ResidualTCN(osf=OSF, model=MODEL)
    elif MODEL.__contains__("model3"):
        model = LoRaResidualTCN(osf=OSF, model=MODEL)
    else:
        raise RuntimeError("No valid model selected")

    # 2. Load the saved weights
    model.load_state_dict(torch.load(f"checkpoints/{model_name}.pth", map_location="cpu"))

    # 3. Set to eval (if inference)
    model.eval()

    return model


def load_existing_state(model_name, model):
    """加载同名 loss 历史和权重，实现作者原有的自动续训行为。"""
    # check for the loss history
    # 2. Check if the file exists and load its data
    loss_filepath = f"checkpoints/{model_name}_loss_history.json"
    if os.path.exists(loss_filepath):
        print(f"Loss history exists, load history..")
        try:
            with open(loss_filepath, "r") as f:
                loss_history = json.load(f)
                # We expect the file to contain a list.
                if not isinstance(loss_history, list):
                    loss_history = []
        except json.JSONDecodeError:
            # This handles cases where the file is empty or has invalid JSON
            print(f"Warning: {loss_filepath} was empty or corrupt. Starting fresh.")
            loss_history = []
    else:
        loss_history = []

    # 2. Load the saved weights
    if os.path.exists(f"checkpoints/{model_name}.pth"):
        print(f"Model exists, load state..")
        model.load_state_dict(torch.load(f"checkpoints/{model_name}.pth", map_location="cpu"))

    return loss_history, model


class HybridDenoiserLoss(nn.Module):
    """
    A hybrid loss function combining Time-Domain L1 and Frequency-Domain Magnitude L1.
    Recommended for denoisers with potential clock skew/non-Gaussian noise.

    Hybrid Denoiser Loss Function
    This approach uses two components:
        - Time-Domain ℓ1 Loss (MAE): To ensure phase coherence and robustness to impulsive OTA noise (less sensitive to outliers than MSE).
        - Frequency-Domain Magnitude Loss: To provide a time-invariant signal that guides training despite the clock skew (Δt).


    LTotal =λ_Time * L_ℓ1_Time + λ_Freq * L_MAE_Mag

    You should set the time weight (λTime) higher than the frequency weight (λFreq), for example, λTime=1.0 and λFreq=0.1.

    OTA 微调时，时域 L1 抵抗脉冲/非高斯噪声，FFT 幅度 L1 则降低小量
    STO、全局相位和 CFO 造成的配对误差。它只能提高对轻微未对齐的容忍度，
    不能替代逐包同步和长度对齐。论文正文报告 λFreq=0.5，而公开代码默认
    为 0.1；复现实验时需要明确记录采用哪一个值。
    """

    def __init__(self, lambda_time=1.0, lambda_freq=0.1):
        super(HybridDenoiserLoss, self).__init__()
        # Weights for the two components
        self.lambda_time = lambda_time
        self.lambda_freq = lambda_freq

        # L1 Loss is used for both components for robustness
        self.l1_loss = nn.L1Loss()

    def forward(self, prediction, target):
        # Tensors are expected to be (B, 2, L) where 2 is (I, Q)

        # 1. TIME DOMAIN L1 LOSS (for phase coherence and robustness)
        # L1 is calculated directly on the real/imag components (B, 2, L)
        loss_time = self.l1_loss(prediction, target)

        # 2. FREQUENCY DOMAIN MAGNITUDE LOSS (for time-invariance/skew compensation)

        # a. Convert (B, 2, L) to (B, L) complex tensor
        prediction_complex = torch.complex(prediction[:, 0, :], prediction[:, 1, :])
        target_complex = torch.complex(target[:, 0, :], target[:, 1, :])

        # b. Apply FFT (along the time dimension L)
        pred_fft = torch.fft.fft(prediction_complex)
        target_fft = torch.fft.fft(target_complex)

        # c. Calculate Magnitude (Absolute Value)
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        # d. Calculate L1 loss on the Magnitude
        loss_freq_mag = self.l1_loss(pred_mag, target_mag)

        # 3. COMBINE LOSSES
        total_loss = (self.lambda_time * loss_time) + (self.lambda_freq * loss_freq_mag)

        return total_loss


class SpectrogramL1Loss(nn.Module):
    def __init__(self, n_fft=256, hop_length=64, window_fn=torch.hann_window):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.l1_loss = nn.L1Loss()

        # --- FIX IS HERE ---
        # 1. Create the window function
        #    We register it as a buffer so it gets moved to the .to(device)
        #    along with the model.
        self.register_buffer('window', window_fn(self.n_fft))
        # -------------------

    def forward(self, prediction, target):
        # 1. Convert (B, 2, L) to (B, L) complex
        pred_complex = torch.complex(prediction[:, 0, :], prediction[:, 1, :])
        targ_complex = torch.complex(target[:, 0, :], target[:, 1, :])

        # 2. Get STFT
        # --- AND FIX IS HERE ---
        #    Pass the window to the stft call
        pred_stft = torch.stft(
            pred_complex,
            self.n_fft,
            self.hop_length,
            window=self.window,  # <--- Added
            return_complex=True
        )
        targ_stft = torch.stft(
            targ_complex,
            self.n_fft,
            self.hop_length,
            window=self.window,  # <--- Added
            return_complex=True
        )
        # -----------------------

        # 3. Calculate L1 loss on the magnitude
        pred_mag = torch.abs(pred_stft)
        targ_mag = torch.abs(targ_stft)

        loss = self.l1_loss(pred_mag, targ_mag)
        return loss


class SpectrogramHybridLoss(nn.Module):
    def __init__(self, n_fft=256, hop_length=64, window_fn=torch.hann_window, mag_weight=1.0, complex_weight=0.5):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.mag_weight = mag_weight
        self.complex_weight = complex_weight

        self.l1_loss = nn.L1Loss()
        self.register_buffer('window', window_fn(self.n_fft))

    def forward(self, prediction, target):
        # --- FIX 1: Check for NaNs in input from model/data ---
        if torch.isnan(prediction).any():
            raise ValueError("NaN detected in 'prediction' (from model) input to loss function")
        if torch.isnan(target).any():
            raise ValueError("NaN detected in 'target' (from data) input to loss function")
        # ----------------------------------------------------

        # 1. Convert (B, 2, L) to (B, L) complex
        pred_complex = torch.complex(prediction[:, 0, :], prediction[:, 1, :])
        targ_complex = torch.complex(target[:, 0, :], target[:, 1, :])

        # --- FIX 2: Check and Pad (To prevent STFT out-of-bounds) ---
        sequence_length = pred_complex.shape[-1]  # Get L

        if sequence_length < self.n_fft:
            # If sequence is too short, pad it with zeros
            padding_needed = self.n_fft - sequence_length
            # F.pad format is (pad_left, pad_right) for the last dimension
            pred_complex = F.pad(pred_complex, (0, padding_needed))
            targ_complex = F.pad(targ_complex, (0, padding_needed))
        # -----------------------------------------------------------

        # 2. Get STFT
        pred_stft = torch.stft(pred_complex, self.n_fft, self.hop_length, window=self.window, return_complex=True)
        targ_stft = torch.stft(targ_complex, self.n_fft, self.hop_length, window=self.window, return_complex=True)

        # --- FIX 3: Check for NaNs after STFT ---
        # This can happen if, for example, the input was all zeros
        if torch.isnan(pred_stft).any():
            raise ValueError("NaN detected in prediction STFT output")
        if torch.isnan(targ_stft).any():
            raise ValueError("NaN detected in target STFT output")
        # ----------------------------------------

        # 3. Magnitude Loss (Time-invariant)
        pred_mag = torch.abs(pred_stft)
        targ_mag = torch.abs(targ_stft)
        loss_mag = self.l1_loss(pred_mag, targ_mag)

        # 4. Complex Loss (Phase-aware)
        loss_complex = self.l1_loss(pred_stft, targ_stft)

        # 5. Combine
        total_loss = (self.mag_weight * loss_mag) + (self.complex_weight * loss_complex)

        # --- FIX 4: Final NaN check ---
        if torch.isnan(total_loss):
            # This would be highly unusual if the above checks passed
            raise ValueError("NaN detected in final loss computation")
        # ------------------------------

        return total_loss


if __name__ == "__main__":

    # 只固定 NumPy 随机源：payload、SNR 和 AWGN 因此可复现；
    # 这里没有固定 PyTorch/CUDA 的随机源。
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    # 命令行参数决定模型结构、采样率关系、数据来源和优化器。
    args = parse_args()
    NUM_EPOCHS = args.num_epochs
    DATASET_SIZE = args.dataset_size
    BATCH_SIZE = args.batch_size  # adjust to available memory -> use 1 as most fit in only once due to the huge input sequences
    OSF = args.osf
    MODEL = args.model
    LR = args.learning_rate
    WEIGHT_DECAY = args.weight_decay
    OPTIMIZER = args.optimizer
    OTA = args.ota
    DSF = args.dsf
    SYNTHETIC_SOURCE = args.synthetic_source
    REFERENCE_ROOT = args.reference_root
    OTA_ROOT = args.ota_root
    PRETRAINED = args.pretrained
    OTA_TARGET = args.ota_target
    OTA_MAX_GROUPS = args.ota_max_groups
    OTA_SPLIT_SEED = args.ota_split_seed
    EARLY_STOP_PATIENCE = args.early_stop_patience
    SYNTHETIC_SEED = args.synthetic_seed
    RANDOM_SYNTHETIC_PAYLOAD = not bool(args.fixed_reference_payloads)
    if args.snr_db is None:
        SYNTHETIC_SNR_RANGE = (
            float(args.snr_min_db),
            float(args.snr_max_db),
        )
        SNR_NAME = f"{args.snr_min_db:g}to{args.snr_max_db:g}"
    else:
        SYNTHETIC_SNR_RANGE = (float(args.snr_db), float(args.snr_db))
        SNR_NAME = f"{args.snr_db:g}"
    if args.cfo_hz is None:
        SYNTHETIC_CFO_RANGE = (
            float(args.cfo_min_hz),
            float(args.cfo_max_hz),
        )
        CFO_NAME = f"{args.cfo_min_hz:g}to{args.cfo_max_hz:g}"
    else:
        SYNTHETIC_CFO_RANGE = (float(args.cfo_hz), float(args.cfo_hz))
        CFO_NAME = f"{args.cfo_hz:g}"
    STO_ENABLED = not bool(args.no_sto)
    STO_INITIAL_RANGE = (
        float(args.sto_initial_min_chips),
        float(args.sto_initial_max_chips),
    )
    STO_SLOPE_RANGE = (
        float(args.sto_slope_min_chips_per_symbol),
        float(args.sto_slope_max_chips_per_symbol),
    )
    if STO_ENABLED:
        STO_NAME = (
            f"i{STO_INITIAL_RANGE[0]:g}to{STO_INITIAL_RANGE[1]:g}"
            f"_d{STO_SLOPE_RANGE[0]:g}to{STO_SLOPE_RANGE[1]:g}"
        )
    else:
        STO_NAME = "none"

    # checkpoint 名字由全部关键参数拼接得到。load_existing_state 会自动
    # 续训同名文件，训练结束时也会覆盖这个名字对应的权重文件。
    # 注意：--ota 会给文件名增加 "_ota_dsf..."，所以合成 checkpoint 与
    # OTA checkpoint 默认不是同名；使用 --pretrained 接上第一阶段权重。
    if OTA:
        group_name = "all" if OTA_MAX_GROUPS is None else str(OTA_MAX_GROUPS)
        model_name = (
            f"model_{MODEL}_bs{BATCH_SIZE}_osf{OSF}_ds{DATASET_SIZE}"
            f"_lr{LR}_wd{WEIGHT_DECAY}_ota_{OTA_TARGET}_g{group_name}"
            f"_dsf{DSF}"
        )
    else:
        model_name = f"model_{MODEL}_bs{BATCH_SIZE}_osf{OSF}_ds{DATASET_SIZE}_lr{LR}_wd{WEIGHT_DECAY}"
        if SYNTHETIC_SOURCE == "reference_phy":
            payload_mode = (
                "random" if RANDOM_SYNTHETIC_PAYLOAD else "fixed"
            )
            model_name += (
                f"_synthref_{payload_mode}_snr{SNR_NAME}_cfo{CFO_NAME}"
                f"_sto{STO_NAME}"
            )
        elif SYNTHETIC_SNR_RANGE != (-22.0, 10.0):
            model_name += f"_snr{SNR_NAME}"

    if MODEL.__contains__("SF7"):
        SF = 7
    elif MODEL.__contains__("SF8"):
        SF = 8
    elif MODEL.__contains__("SF9"):
        SF = 9
    elif MODEL.__contains__("SF10"):
        SF = 10
    elif MODEL.__contains__("SF11"):
        SF = 11
    else:
        SF = 12  # default

    if MODEL.__contains__("BW250"):
        BW = 250e3
    else:
        BW = 125e3

    # 建立网络。论文推荐的 model0v0 是“polyphase 插值 + 四层残差 CNN”。
    if MODEL.__contains__("model0"):
        model = SimpleComplexCNN0(oversampling=OSF, model=MODEL).to(device)
    elif MODEL.__contains__("model1"):
        model = SimpleComplexCNN1(oversampling=OSF, model=MODEL).to(device)
    elif MODEL.__contains__("model2"):  # seems to end with NAN -> try lowering LR, i.e. try lr=1e-4
        model = ResidualTCN(osf=OSF, model=MODEL).to(device)
    elif MODEL.__contains__("model3"):
        model = LoRaResidualTCN(osf=OSF, model=MODEL).to(device)
    else:
        raise RuntimeError("No valid model selected")

    if OPTIMIZER == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=LR,
                               weight_decay=WEIGHT_DECAY)  # optimize optimizer and lr weight_decay, typically: 1e-5 to 1e-3
    elif OPTIMIZER == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    if MODEL.__contains__("l1"):
        print("Using L1Loss")
        loss_fn = nn.L1Loss()
    elif MODEL.__contains__("hl"):
        # 模型名包含 "hl" 时启用论文为 OTA 数据设计的混合损失。
        print("Using HybridDenoiserLoss")
        loss_fn = HybridDenoiserLoss()
    elif MODEL.__contains__("spechybl"):
        print("Using SpectrogramHybridLoss")
        loss_fn = SpectrogramHybridLoss().to(device)
    elif MODEL.__contains__("specl"):
        print("Using SpectrogramL1Loss")
        loss_fn = SpectrogramL1Loss().to(device)
    else:
        print("Using MSELoss")
        loss_fn = nn.MSELoss()

    # 选择数据来源：
    #   不带 --ota：运行时合成 packet，降采样后加 AWGN，无需离线采集。
    #   带 --ota：读取已经逐包切片并与 reference 配对的真实 IQ。
    if OTA:
        # 公开 OTA 数据只覆盖 SF12，因此作者在入口中直接拒绝其他 SF。
        if SF != 12:
            raise RuntimeError(f"No SF{SF} dataset. exit.")

        if MODEL.__contains__("model3"):
            raise RuntimeError(
                "model3's legacy asymmetric trim is not compatible with the "
                "aligned manifest dataset; use model0 for this OTA stage."
            )
        # 一个物理包的所有 ADC/polyphase 视图只能在同一个 split。默认
        # received 标签是同一接收波形的 1 MS/s 版本，不做 CFO/SFO/增益校正。
        dataset = OTALoRaDataset(
            oversampling=OSF,
            downsampling=DSF,
            return_snr=True,
            dataset_root=OTA_ROOT,
            split="train",
            split_seed=OTA_SPLIT_SEED,
            max_groups=OTA_MAX_GROUPS,
            target_source=OTA_TARGET,
        )
        validation_dataset = OTALoRaDataset(
            oversampling=OSF,
            downsampling=DSF,
            return_snr=True,
            dataset_root=OTA_ROOT,
            split="validation",
            split_seed=OTA_SPLIT_SEED,
            max_groups=OTA_MAX_GROUPS,
            target_source=OTA_TARGET,
        )
        test_dataset = OTALoRaDataset(
            oversampling=OSF,
            downsampling=DSF,
            return_snr=True,
            dataset_root=OTA_ROOT,
            split="test",
            split_seed=OTA_SPLIT_SEED,
            max_groups=OTA_MAX_GROUPS,
            target_source=OTA_TARGET,
        )
        print(
            f"LoRa train/validation/test={dataset.size}/"
            f"{validation_dataset.size}/{test_dataset.size}, "
            f"ota_root={OTA_ROOT}, target={OTA_TARGET}"
        )
    else:
        validation_dataset = None
        test_dataset = None
        dataset = build_synthetic_dataset(
            source=SYNTHETIC_SOURCE,
            reference_root=REFERENCE_ROOT,
            oversampling=OSF,
            size=DATASET_SIZE,
            downsampling=DSF,
            sf=SF,
            bandwidth=BW,
            seed=SYNTHETIC_SEED,
            snr_range=SYNTHETIC_SNR_RANGE,
            random_payload=RANDOM_SYNTHETIC_PAYLOAD,
            cfo_range_hz=SYNTHETIC_CFO_RANGE,
            sto_enabled=STO_ENABLED,
            sto_initial_range_chips=STO_INITIAL_RANGE,
            sto_slope_range_chips_per_symbol=STO_SLOPE_RANGE,
        )
        print(
            f"Synthetic source={SYNTHETIC_SOURCE}, size={len(dataset)}, "
            f"random_payload={RANDOM_SYNTHETIC_PAYLOAD}, "
            f"snr_range={SYNTHETIC_SNR_RANGE}, "
            f"cfo_range_hz={SYNTHETIC_CFO_RANGE}, "
            f"sto_enabled={STO_ENABLED}, "
            f"sto_initial_chips={STO_INITIAL_RANGE}, "
            f"sto_slope_chips_per_symbol={STO_SLOPE_RANGE}, "
            f"reference_root={REFERENCE_ROOT}"
        )

    # 特殊实验模型可以把 2000 个合成 packet 与 OTA 数据串接起来。
    if MODEL.__contains__("synthotacomb"):
        print("Combining dataset")
        from torch.utils.data import ConcatDataset

        # synthdata = SyntheticLoRaDataset(oversampling=OSF, size=len(dataset), downsampling=DSF) # TOO large, does not fit into memory
        synthdata = build_synthetic_dataset(
            source=SYNTHETIC_SOURCE,
            reference_root=REFERENCE_ROOT,
            oversampling=OSF,
            size=2000,
            downsampling=DSF,
            sf=SF,
            bandwidth=BW,
            seed=SYNTHETIC_SEED,
            snr_range=SYNTHETIC_SNR_RANGE,
            random_payload=RANDOM_SYNTHETIC_PAYLOAD,
            cfo_range_hz=SYNTHETIC_CFO_RANGE,
            sto_enabled=STO_ENABLED,
            sto_initial_range_chips=STO_INITIAL_RANGE,
            sto_slope_range_chips_per_symbol=STO_SLOPE_RANGE,
        )
        # Combine the datasets sequentially
        dataset = ConcatDataset([synthdata, dataset])

    # 所有合成数据集均在初始化时生成或加载；之后每个 epoch 只改变读取顺序。
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = (
        DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
        if validation_dataset is not None
        else None
    )

    # 同名 checkpoint 优先用于续训；仅在它不存在时才加载预训练权重，
    # 从而支持 synthetic -> OTA 两阶段训练且不会覆盖已经开始的 OTA 进度。
    current_checkpoint = Path("checkpoints") / f"{model_name}.pth"
    resume_current_checkpoint = current_checkpoint.is_file()
    loss_history, model = load_existing_state(model_name, model)
    if not resume_current_checkpoint and PRETRAINED is not None:
        pretrained_path = PRETRAINED.expanduser().resolve()
        if not pretrained_path.is_file():
            raise FileNotFoundError(
                f"missing pretrained checkpoint: {pretrained_path}"
            )
        model.load_state_dict(
            torch.load(pretrained_path, map_location="cpu", weights_only=True)
        )
        print(f"Loaded pretrained checkpoint: {pretrained_path}")

    # 合成预训练沿用训练 loss；OTA 微调固定监控从未反传的 validation
    # loss。两种路径均恢复本次运行最佳权重，patience 默认 10。
    early_stop_patience = EARLY_STOP_PATIENCE
    epochs_without_improvement = 0
    best_loss = float("inf")
    best_epoch = 0
    best_model_state = None
    train_loss_history: list[float] = []
    validation_loss_history: list[float] = []

    def predict(xb, snr_b):
        if hasattr(model, "gated"):
            return model(xb, snr_b)
        return model(xb)

    # 标准监督训练：低速率 OTA IQ -> RF-SR -> 同一接收波形的高采样 OTA IQ。
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        running_loss = 0.0
        model.train()

        # 1. Assign the tqdm object to a variable (e.g., 'train_bar')
        train_bar = tqdm(dataloader, desc="Training", unit="batch")

        # Wrap the dataloader with tqdm
        for xb, yb, snr_b in train_bar:
            # OTA model0 batch 的预期 shape：
            # xb=[B, 2, L]，yb=[B, 2, OSF*L]，snr_b=[B] 或 [B, 1]。
            xb, yb, snr_b = xb.to(device), yb.to(device), snr_b.to(device)

            pred = predict(xb, snr_b)

            loss = loss_fn(pred, yb)

            # 3. Calculate the L1 penalty  & add to the main loss
            l1l = torch.tensor(0.0)
            if MODEL.__contains__("lopenalty"):
                # 模型名包含 "lopenalty" 时，再给全部 CNN 参数施加 L1/Lasso
                # 稀疏正则；公开 OTA checkpoint 使用了这一分支。
                l1_lambda = 1.0
                l1_penalty_sum = 0.0
                for param in model.parameters():
                    l1_penalty_sum += torch.abs(param).sum()
                l1l = l1_lambda * l1_penalty_sum
                # print(f"L1 Loss: {l1l}")
                loss += l1l

            opt.zero_grad()
            loss.backward()

            # This caps the gradient norm at 1.0, preventing explosions.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            opt.step()

            # --- 3. THIS IS THE NEW PART ---
            # Update the progress bar's postfix
            # Use .item() to get the Python number from the tensor
            train_bar.set_postfix(
                total_loss=loss.item(),
                l1_loss=l1l.item()
            )

            running_loss += loss.item()

        avg_train_loss = running_loss / len(dataloader)
        train_loss_history.append(avg_train_loss)

        if validation_loader is None:
            monitored_loss = avg_train_loss
            tqdm.write(f"Epoch {epoch + 1}: loss = {avg_train_loss:.7f}")
        else:
            model.eval()
            validation_total = 0.0
            with torch.inference_mode():
                for xb, yb, snr_b in tqdm(
                    validation_loader,
                    desc="Validation",
                    unit="batch",
                    leave=False,
                ):
                    xb = xb.to(device)
                    yb = yb.to(device)
                    snr_b = snr_b.to(device)
                    validation_total += loss_fn(
                        predict(xb, snr_b), yb
                    ).item()
            monitored_loss = validation_total / len(validation_loader)
            validation_loss_history.append(monitored_loss)
            tqdm.write(
                f"Epoch {epoch + 1}: train_loss = {avg_train_loss:.7f}, "
                f"validation_loss = {monitored_loss:.7f}"
            )
        loss_history.append(monitored_loss)

        if monitored_loss < best_loss:
            best_loss = monitored_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_model_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1
            tqdm.write(
                "Early stopping: no loss improvement for "
                f"{epochs_without_improvement}/{early_stop_patience} epochs."
            )
            if epochs_without_improvement >= early_stop_patience:
                tqdm.write(
                    f"Early stopping at epoch {epoch + 1}; best loss "
                    f"{best_loss:.7f} was at epoch {best_epoch}."
                )
                break

    # 保存最终权重与逐 epoch loss；同名文件会被覆盖。
    os.makedirs("checkpoints/", exist_ok=True)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), f"checkpoints/{model_name}.pth")
    with open(f"checkpoints/{model_name}_loss_history.json", "w") as f:
        json.dump(loss_history, f)
    with open(f"checkpoints/{model_name}_train_loss_history.json", "w") as f:
        json.dump(train_loss_history, f)
    if validation_loader is not None:
        with open(
            f"checkpoints/{model_name}_validation_loss_history.json", "w"
        ) as f:
            json.dump(validation_loss_history, f)
