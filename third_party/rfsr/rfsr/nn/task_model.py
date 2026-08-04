"""Task-aware polyphase RF super-resolution network."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F


def linear_polyphase_baseline(x: torch.Tensor, factor: int = 4) -> torch.Tensor:
    """Complex linear interpolation with exact observed-sample placement."""

    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError("x must have shape [batch, 2, time]")
    if int(factor) != 4:
        raise ValueError("the current RF-SR task is fixed at factor=4")
    following = torch.cat((x[..., 1:], x[..., -1:]), dim=-1)
    phases = torch.stack(
        (
            x,
            0.75 * x + 0.25 * following,
            0.50 * x + 0.50 * following,
            0.25 * x + 0.75 * following,
        ),
        dim=-1,
    )
    return phases.reshape(x.shape[0], 2, x.shape[-1] * 4)


def _fractional_sinc_kernel(
    fraction: float,
    *,
    radius: int,
    beta: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Finite Kaiser-windowed sinc for one fractional low-rate delay."""

    offsets = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    argument = float(fraction) - offsets
    sinc = torch.sinc(argument)
    normalized = offsets / float(radius + 1)
    window_argument = torch.clamp(1.0 - normalized.square(), min=0.0)
    beta_tensor = torch.tensor(float(beta), dtype=dtype, device=device)
    window = torch.i0(beta_tensor * torch.sqrt(window_argument)) / torch.i0(
        beta_tensor
    )
    kernel = sinc * window
    return kernel / kernel.sum().clamp_min(torch.finfo(dtype).eps)


def sinc_polyphase_baseline(
    x: torch.Tensor,
    factor: int = 4,
    *,
    radius: int = 16,
    beta: float = 8.6,
) -> torch.Tensor:
    """Band-limited polyphase interpolation with exact phase-zero samples.

    This retains the useful interpolation-plus-residual inductive bias of the
    original RF-SR model, but computes the FIR on the low-rate grid and then
    hard-replaces the observed branch.  The learned network therefore spends
    capacity on departures from a strong band-limited prior instead of
    relearning interpolation from the linear baseline.
    """

    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError("x must have shape [batch, 2, time]")
    if int(factor) != 4:
        raise ValueError("the current RF-SR task is fixed at factor=4")
    if int(radius) < 2:
        raise ValueError("radius must be at least 2")
    kernel_size = 2 * int(radius) + 1
    padded = F.pad(x, (int(radius), int(radius)), mode="replicate")
    phases: list[torch.Tensor] = [x]
    for phase in range(1, 4):
        kernel = _fractional_sinc_kernel(
            phase / 4.0,
            radius=int(radius),
            beta=float(beta),
            dtype=x.dtype,
            device=x.device,
        )
        # conv1d performs cross-correlation.  Kernel offsets are ordered from
        # -radius to +radius, exactly matching each padded input window.
        weight = kernel.reshape(1, 1, kernel_size).repeat(2, 1, 1)
        phases.append(F.conv1d(padded, weight, groups=2))
    stacked = torch.stack(phases, dim=-1)
    return stacked.reshape(x.shape[0], 2, x.shape[-1] * 4)


class _GatedDilatedBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.filter_gate = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=3,
            padding=int(dilation),
            dilation=int(dilation),
        )
        self.project = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        candidate, gate = self.filter_gate(x).chunk(2, dim=1)
        update = torch.tanh(candidate) * torch.sigmoid(gate)
        update = self.dropout(self.project(update))
        return (x + update) / math.sqrt(2.0)


class TaskAwarePolyphaseTCN(nn.Module):
    """Predict only the three unobserved 1 MS/s polyphase branches.

    The network operates on the 250 kS/s grid, so its dilated receptive field
    is useful without allocating high-rate feature maps.  Phase zero is copied
    from the input after all learned operations, making sample consistency an
    invariant rather than a soft penalty.
    """

    def __init__(
        self,
        *,
        channels: int = 32,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128),
        dropout: float = 0.0,
        baseline: str = "linear",
        sinc_radius: int = 16,
        hard_observed: bool = True,
    ):
        super().__init__()
        if int(channels) < 4:
            raise ValueError("channels must be at least 4")
        if not dilations or any(int(value) <= 0 for value in dilations):
            raise ValueError("dilations must contain positive integers")
        self.channels = int(channels)
        self.dilations = tuple(int(value) for value in dilations)
        self.dropout = float(dropout)
        self.baseline = str(baseline)
        self.sinc_radius = int(sinc_radius)
        self.hard_observed = bool(hard_observed)
        if self.baseline not in {"linear", "sinc"}:
            raise ValueError("baseline must be 'linear' or 'sinc'")
        self.stem = nn.Conv1d(2, self.channels, kernel_size=7, padding=3)
        self.blocks = nn.ModuleList(
            _GatedDilatedBlock(self.channels, value, self.dropout)
            for value in self.dilations
        )
        # Hard mode predicts three missing phases; soft mode also learns a
        # denoising correction for the observed branch. A zero head always
        # starts at the documented interpolation prior.
        self.head = nn.Conv1d(
            self.channels, 6 if self.hard_observed else 8, kernel_size=1
        )
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @property
    def receptive_field_input_samples(self) -> int:
        return int(7 + 2 * sum(self.dilations))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 2:
            raise ValueError("x must have shape [batch, 2, time]")
        features = torch.tanh(self.stem(x))
        for block in self.blocks:
            features = block(features)
        residual_phases = 3 if self.hard_observed else 4
        residual = self.head(features).reshape(
            x.shape[0], 2, residual_phases, x.shape[-1]
        )
        baseline_values = self.interpolation_baseline(x)
        baseline = baseline_values.reshape(
            x.shape[0], 2, x.shape[-1], 4
        )
        if self.hard_observed:
            missing = baseline[..., 1:] + residual.permute(0, 1, 3, 2)
            phases = torch.cat((x.unsqueeze(-1), missing), dim=-1)
        else:
            phases = baseline + residual.permute(0, 1, 3, 2)
        output = phases.reshape(x.shape[0], 2, x.shape[-1] * 4)
        # This assertion catches future layout regressions immediately.
        if output.shape[-1] != 4 * x.shape[-1]:
            raise RuntimeError("polyphase output length invariant failed")
        return output

    def interpolation_baseline(self, x: torch.Tensor) -> torch.Tensor:
        """Return the exact baseline used underneath the learned residual."""

        return (
            sinc_polyphase_baseline(x, radius=self.sinc_radius)
            if self.baseline == "sinc"
            else linear_polyphase_baseline(x)
        )


def input_rms_scale(x: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Return an inference-available per-window complex RMS scale."""

    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError("x must have shape [batch, 2, time]")
    power = x.square().sum(dim=1).mean(dim=-1, keepdim=True)
    return power.clamp_min(float(epsilon) ** 2).sqrt().unsqueeze(1)


def load_task_aware_checkpoint(
    checkpoint: str | Path, device: str | torch.device = "cpu"
) -> tuple[TaskAwarePolyphaseTCN, dict[str, object]]:
    """Load the self-describing checkpoint written by the official trainer."""

    path = Path(checkpoint).expanduser().resolve()
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or payload.get("schema") != "task-aware-rfsr-v1":
        raise ValueError(f"unsupported task-aware RF-SR checkpoint: {path}")
    model_config = dict(payload["model_config"])
    if "dilations" in model_config:
        model_config["dilations"] = tuple(model_config["dilations"])
    model = TaskAwarePolyphaseTCN(**model_config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


class TaskAwareRFSRFrontend:
    """NumPy inference wrapper with overlap-safe long-packet chunking."""

    def __init__(
        self,
        model: TaskAwarePolyphaseTCN,
        *,
        device: str | torch.device,
        chunk_input_samples: int = 131_072,
        residual_strength: float = 1.0,
    ):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.chunk_input_samples = int(chunk_input_samples)
        self.residual_strength = float(residual_strength)
        if self.chunk_input_samples <= 0:
            raise ValueError("chunk_input_samples must be positive")
        if not 0.0 <= self.residual_strength <= 1.0:
            raise ValueError("residual_strength must be in [0, 1]")
        self.overlap_input_samples = (
            int(model.receptive_field_input_samples) // 2 + 2
        )

    @torch.inference_mode()
    def enhance(self, samples: np.ndarray) -> np.ndarray:
        values = np.asarray(samples, dtype=np.complex64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("samples must be a non-empty complex vector")
        rms = float(np.sqrt(np.mean(np.abs(values).astype(np.float64) ** 2)))
        scale = max(rms, 1e-8)
        normalized = values / scale
        pieces: list[np.ndarray] = []
        for core_start in range(0, values.size, self.chunk_input_samples):
            core_stop = min(values.size, core_start + self.chunk_input_samples)
            context_start = max(0, core_start - self.overlap_input_samples)
            context_stop = min(
                values.size, core_stop + self.overlap_input_samples
            )
            context = normalized[context_start:context_stop]
            tensor = torch.from_numpy(
                np.stack((context.real, context.imag), axis=0)[None].copy()
            ).to(self.device)
            full = self.model(tensor)
            if self.residual_strength < 1.0:
                baseline = self.model.interpolation_baseline(tensor)
                full = baseline + self.residual_strength * (full - baseline)
            prediction = full[0].detach().float().cpu().numpy()
            crop_start = 4 * (core_start - context_start)
            crop_stop = crop_start + 4 * (core_stop - core_start)
            complex_prediction = prediction[0] + 1j * prediction[1]
            pieces.append(
                np.asarray(
                    complex_prediction[crop_start:crop_stop] * scale,
                    dtype=np.complex64,
                )
            )
        output = np.concatenate(pieces)
        if output.size != 4 * values.size:
            raise RuntimeError("chunked RF-SR output length mismatch")
        # Hard checkpoints retain exact observation identity at wrapper level.
        if self.model.hard_observed:
            output[::4] = values
        return output


__all__ = [
    "TaskAwareRFSRFrontend",
    "TaskAwarePolyphaseTCN",
    "input_rms_scale",
    "linear_polyphase_baseline",
    "load_task_aware_checkpoint",
    "sinc_polyphase_baseline",
]
