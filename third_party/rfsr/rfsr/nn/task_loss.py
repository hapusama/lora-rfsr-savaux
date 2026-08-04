"""Waveform and coherent-demodulation objectives for task-aware RF-SR."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rfsr.PHY import lora_chirp


class TaskAwareRFSRLoss(nn.Module):
    """Complex Charbonnier plus concentration and peak-margin terms.

    ``spectral_mode="savaux"`` implements the exact differentiable form of
    Savaux Eq. (36)-(37): all eight 1 MS/s branches receive the wrap-tail DFT
    correction and are then combined coherently.  ``four_branch`` is retained
    only so checkpoints and measurements from the first experiment remain
    reproducible.
    """

    def __init__(
        self,
        *,
        sf: int = 12,
        output_os_factor: int = 8,
        polyphase_branches: int = 4,
        charbonnier_epsilon: float = 1e-3,
        spectral_epsilon: float = 1e-12,
        correct_radius: int = 1,
        guard_radius: int = 2,
        margin_db: float = 6.0,
        spectral_mode: str = "four_branch",
    ):
        super().__init__()
        self.sf = int(sf)
        self.n_bins = 1 << self.sf
        self.output_os_factor = int(output_os_factor)
        self.polyphase_branches = int(polyphase_branches)
        self.symbol_samples = self.n_bins * self.output_os_factor
        if self.symbol_samples % self.polyphase_branches:
            raise ValueError("symbol length must be divisible by branch count")
        self.branch_bins = self.symbol_samples // self.polyphase_branches
        self.spectral_mode = str(spectral_mode)
        if self.spectral_mode not in {"four_branch", "savaux"}:
            raise ValueError("spectral_mode must be 'four_branch' or 'savaux'")
        self.charbonnier_epsilon = float(charbonnier_epsilon)
        self.spectral_epsilon = float(spectral_epsilon)
        self.correct_radius = int(correct_radius)
        self.guard_radius = int(guard_radius)
        if self.correct_radius < 0 or self.guard_radius < self.correct_radius:
            raise ValueError("guard radius must cover the correct-bin region")
        self.margin_log_power = float(
            math.log(10.0 ** (float(margin_db) / 10.0))
        )
        upchirp, _ = lora_chirp(
            +1,
            0,
            125_000,
            self.n_bins,
            self.output_os_factor,
            0,
            0,
        )
        downchirp = np.conjugate(upchirp).astype(np.complex64)
        self.register_buffer(
            "downchirp_real", torch.from_numpy(downchirp.real.copy())
        )
        self.register_buffer(
            "downchirp_imag", torch.from_numpy(downchirp.imag.copy())
        )

        # Constants for Savaux Eq. (36)'s chirp-z convolution and Eq. (37)'s
        # coherent branch alignment.  They are buffers so device transfers and
        # checkpoint serialization remain automatic.
        indexes = np.arange(self.n_bins, dtype=np.float64)
        forward_chirp = np.exp(
            1j * np.pi * indexes * indexes / float(self.n_bins)
        )
        reverse_chirp = np.exp(
            -1j * np.pi * indexes * indexes / float(self.n_bins)
        )
        wrap_fft_length = 1 << int((2 * self.n_bins - 1).bit_length())
        reverse_chirp_fft = np.fft.fft(reverse_chirp, wrap_fft_length)
        savaux_branches = np.arange(
            self.output_os_factor, dtype=np.float64
        )[:, None]
        savaux_bins = indexes[None, :]
        branch_weights = np.exp(
            -2j
            * np.pi
            * savaux_branches
            * savaux_bins
            / float(self.n_bins * self.output_os_factor)
        )
        wrap_phases = np.exp(
            2j
            * np.pi
            * np.arange(self.output_os_factor, dtype=np.float64)
            / float(self.output_os_factor)
        )
        tail_mask = np.ones(self.n_bins, dtype=np.float32)
        tail_mask[0] = 0.0
        self.wrap_fft_length = int(wrap_fft_length)
        self.register_buffer(
            "savaux_forward_chirp",
            torch.from_numpy(forward_chirp.astype(np.complex64)),
        )
        self.register_buffer(
            "savaux_reverse_chirp_fft",
            torch.from_numpy(reverse_chirp_fft.astype(np.complex64)),
        )
        self.register_buffer(
            "savaux_branch_weights",
            torch.from_numpy(branch_weights.astype(np.complex64)),
        )
        self.register_buffer(
            "savaux_wrap_phases",
            torch.from_numpy(wrap_phases.astype(np.complex64)),
        )
        self.register_buffer("savaux_tail_mask", torch.from_numpy(tail_mask))

    def waveform_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if prediction.shape != target.shape or prediction.ndim != 3:
            raise ValueError("prediction and target must have equal [B,2,L] shape")
        squared_magnitude = (prediction - target).square().sum(dim=1)
        values = torch.sqrt(
            squared_magnitude + self.charbonnier_epsilon**2
        )
        if valid_mask is None:
            return values.mean()
        mask = valid_mask.to(device=values.device, dtype=values.dtype)
        if mask.shape != values.shape:
            raise ValueError("valid_mask must have shape [B,L]")
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    def four_branch_power(self, prediction: torch.Tensor) -> torch.Tensor:
        if (
            prediction.ndim != 3
            or prediction.shape[1] != 2
            or prediction.shape[-1] != self.symbol_samples
        ):
            raise ValueError(
                f"prediction must have shape [B,2,{self.symbol_samples}]"
            )
        real, imag = prediction[:, 0], prediction[:, 1]
        dechirped_real = real * self.downchirp_real - imag * self.downchirp_imag
        dechirped_imag = real * self.downchirp_imag + imag * self.downchirp_real
        dechirped = torch.complex(dechirped_real, dechirped_imag)
        branches = dechirped.reshape(
            prediction.shape[0], self.branch_bins, self.polyphase_branches
        )
        spectrum = torch.fft.fft(branches, dim=1, norm="ortho")
        return spectrum.abs().square().sum(dim=2)

    def savaux_coherent_spectrum(
        self, prediction: torch.Tensor
    ) -> torch.Tensor:
        """Return the exact complex Savaux Eq. (37) spectrum."""

        if (
            prediction.ndim != 3
            or prediction.shape[1] != 2
            or prediction.shape[-1] != self.symbol_samples
        ):
            raise ValueError(
                f"prediction must have shape [B,2,{self.symbol_samples}]"
            )
        real, imag = prediction[:, 0], prediction[:, 1]
        dechirped = torch.complex(
            real * self.downchirp_real - imag * self.downchirp_imag,
            real * self.downchirp_imag + imag * self.downchirp_real,
        )
        branches = dechirped.reshape(
            prediction.shape[0], self.n_bins, self.output_os_factor
        )
        normalization = math.sqrt(float(self.n_bins))
        spectra = torch.fft.fft(branches, dim=1) / normalization

        if self.output_os_factor > 1:
            values = branches[:, :, 1:]
            lhs = torch.zeros_like(values)
            lhs[:, 1:, :] = torch.flip(values[:, 1:, :], dims=(1,)) * (
                self.savaux_forward_chirp[None, 1:, None]
            )
            convolution = torch.fft.ifft(
                torch.fft.fft(lhs, n=self.wrap_fft_length, dim=1)
                * self.savaux_reverse_chirp_fft[None, :, None],
                dim=1,
            )[:, : self.n_bins, :]
            tails = (
                self.savaux_forward_chirp[None, :, None]
                * convolution
                * self.savaux_tail_mask[None, :, None]
            )
            corrected = spectra[:, :, 1:] + (
                self.savaux_wrap_phases[None, None, 1:] - 1.0
            ) * tails / normalization
            spectra = torch.cat((spectra[:, :, :1], corrected), dim=2)

        # Buffer layout is [branch, bin], while spectra is [batch, bin,
        # branch].  The sum is complex/coherent, not a sum of branch powers.
        return (
            spectra
            * self.savaux_branch_weights.transpose(0, 1)[None, :, :]
        ).sum(dim=2)

    def savaux_coherent_power(self, prediction: torch.Tensor) -> torch.Tensor:
        return self.savaux_coherent_spectrum(prediction).abs().square()

    def spectral_power(self, prediction: torch.Tensor) -> torch.Tensor:
        if self.spectral_mode == "savaux":
            return self.savaux_coherent_power(prediction)
        return self.four_branch_power(prediction)

    def task_losses(
        self,
        power: torch.Tensor,
        correct_bins: torch.Tensor,
        task_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if power.ndim != 2 or power.shape[1] <= 0:
            raise ValueError("power has an unexpected shape")
        spectrum_bins = int(power.shape[1])
        centers = correct_bins.to(device=power.device, dtype=torch.long).reshape(-1)
        if centers.numel() != power.shape[0]:
            raise ValueError("one correct bin is required per symbol")
        centers = centers.remainder(spectrum_bins)
        batch = torch.arange(power.shape[0], device=power.device)[:, None]
        correct_offsets = torch.arange(
            -self.correct_radius,
            self.correct_radius + 1,
            device=power.device,
        )[None, :]
        positive_indices = (centers[:, None] + correct_offsets).remainder(
            spectrum_bins
        )
        positive = power[batch, positive_indices].sum(dim=1)
        total = power.sum(dim=1)
        concentration = -torch.log(
            (positive + self.spectral_epsilon)
            / (total + self.spectral_epsilon)
        )

        bins = torch.arange(spectrum_bins, device=power.device)[None, :]
        circular_distance = torch.minimum(
            (bins - centers[:, None]).remainder(spectrum_bins),
            (centers[:, None] - bins).remainder(spectrum_bins),
        )
        wrong = power.masked_fill(
            circular_distance <= self.guard_radius,
            torch.finfo(power.dtype).min,
        ).amax(dim=1)
        margin = F.softplus(
            self.margin_log_power
            + torch.log(wrong.clamp_min(0.0) + self.spectral_epsilon)
            - torch.log(positive + self.spectral_epsilon)
        )
        if task_mask is None:
            return concentration.mean(), margin.mean()
        weights = task_mask.to(device=power.device, dtype=power.dtype).reshape(-1)
        if weights.numel() != power.shape[0]:
            raise ValueError("task_mask must have one value per symbol")
        denominator = weights.sum()
        if bool((denominator <= 0).item()):
            connected_zero = power.sum() * 0.0
            return connected_zero, connected_zero
        return (
            (concentration * weights).sum() / denominator,
            (margin * weights).sum() / denominator,
        )

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        correct_bins: torch.Tensor,
        *,
        task_mask: torch.Tensor | None = None,
        concentration_weight: float = 0.05,
        margin_weight: float = 0.05,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        waveform = self.waveform_loss(prediction, target, valid_mask)
        power = self.spectral_power(prediction)
        concentration, margin = self.task_losses(
            power, correct_bins, task_mask
        )
        total = (
            waveform
            + float(concentration_weight) * concentration
            + float(margin_weight) * margin
        )
        return {
            "total": total,
            "waveform": waveform,
            "concentration": concentration,
            "margin": margin,
        }


__all__ = ["TaskAwareRFSRLoss"]
