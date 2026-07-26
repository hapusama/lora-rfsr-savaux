"""Adapter for the authors' RF-SR interpolation and pretrained CNN.

PyTorch and the bundled or explicitly selected ``RFSuperResolution`` source
tree are loaded only when the frontend is instantiated. Importing weak-decoder
therefore does not add a mandatory ML dependency to the existing Savaux
receiver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib
from pathlib import Path
import re
import sys
from types import ModuleType
from typing import Literal

import numpy as np


DEFAULT_SYNTHETIC_CHECKPOINT = (
    "model_model0v0_bs5_osf4_ds250_lr0.001_wd1e-05.pth"
)
DEFAULT_OTA_CHECKPOINT = (
    "model_model0v0lopenaltyhl_bs1_osf4_ds250_lr0.0001_wd1e-05_ota_dsf8.pth"
)

FrontendMode = Literal["interpolation", "rfsr"]


def default_rfsr_repo_root() -> Path:
    """Return the RFSR source tree vendored in this consolidated repository."""

    return Path(__file__).resolve().parents[2] / "third_party" / "rfsr"


@dataclass(frozen=True)
class RFSRFrontendConfig:
    """RFSR source tree, checkpoint, and bounded-memory inference settings."""

    repo_root: Path = field(default_factory=default_rfsr_repo_root)
    checkpoint: Path | None = None
    checkpoint_name: str = DEFAULT_OTA_CHECKPOINT
    model_variant: str | None = None
    upsample_factor: int = 4
    device: str = "cpu"
    chunk_input_samples: int = 65_536
    overlap_input_samples: int = 68

    def resolved_checkpoint(self) -> Path:
        if self.checkpoint is not None:
            return Path(self.checkpoint).expanduser().resolve()
        return (
            Path(self.repo_root).expanduser().resolve()
            / "checkpoints"
            / str(self.checkpoint_name)
        )


@dataclass(frozen=True)
class RFSRProvenance:
    repo_root: str
    repo_commit: str
    checkpoint: str
    checkpoint_sha256: str
    model_variant: str
    upsample_factor: int
    device: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(repo_root: Path) -> str:
    head = repo_root / ".git" / "HEAD"
    if not head.is_file():
        vendored_commit = repo_root / "UPSTREAM_COMMIT"
        if vendored_commit.is_file():
            return vendored_commit.read_text(
                encoding="ascii", errors="replace"
            ).strip()
        return "unknown"
    value = head.read_text(encoding="ascii", errors="replace").strip()
    if value.startswith("ref: "):
        ref = repo_root / ".git" / value[5:]
        if ref.is_file():
            return ref.read_text(encoding="ascii", errors="replace").strip()
        packed = repo_root / ".git" / "packed-refs"
        if packed.is_file():
            suffix = value[5:]
            for row in packed.read_text(encoding="ascii", errors="replace").splitlines():
                if row and not row.startswith("#") and row.endswith(f" {suffix}"):
                    return row.split(" ", maxsplit=1)[0]
        return "unknown"
    return value


def _checkpoint_model_variant(checkpoint: Path) -> str:
    match = re.search(r"model_(model0[^_]*)_bs", checkpoint.name)
    if match is None:
        raise ValueError(
            "cannot infer RF-SR model variant from checkpoint name; "
            "set RFSRFrontendConfig.model_variant"
        )
    return str(match.group(1))


def _import_from_checkout(repo_root: Path, module_name: str) -> ModuleType:
    root_text = str(repo_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module = importlib.import_module(module_name)
    source = Path(str(module.__file__)).resolve()
    try:
        source.relative_to(repo_root)
    except ValueError as exc:
        raise RuntimeError(
            f"{module_name} was imported from {source}, not requested checkout {repo_root}"
        ) from exc
    return module


class RFSuperResolutionFrontend:
    """Run the exact author interpolation baseline or pretrained RF-SR model."""

    # The author interpolation has a 511-configured (512 actual) FIR and the
    # four kernel-3 CNN layers add four high-rate samples of context.
    minimum_overlap_input_samples = 65

    def __init__(self, config: RFSRFrontendConfig):
        self.config = config
        self.repo_root = Path(config.repo_root).expanduser().resolve()
        self.checkpoint = config.resolved_checkpoint()
        self._validate_paths()
        if int(config.upsample_factor) != 4:
            raise ValueError("the published checkpoints require upsample_factor=4")
        if int(config.chunk_input_samples) <= 0:
            raise ValueError("chunk_input_samples must be positive")
        if int(config.overlap_input_samples) < self.minimum_overlap_input_samples:
            raise ValueError(
                "overlap_input_samples must be at least "
                f"{self.minimum_overlap_input_samples} for artifact-free chunking"
            )

        try:
            self._torch = importlib.import_module("torch")
        except ImportError as exc:
            raise RuntimeError(
                "RF-SR inference requires PyTorch; use the isolated RF-SR environment"
            ) from exc

        interp_module = _import_from_checkout(self.repo_root, "rfsr.interp")
        nn_module = _import_from_checkout(self.repo_root, "rfsr.nn.nn")
        self._interpolate_tensor = interp_module.resample_poly_torch_batch2
        variant = config.model_variant or _checkpoint_model_variant(self.checkpoint)
        self.model_variant = str(variant)
        self.device = self._torch.device(str(config.device))
        self.model = nn_module.SimpleComplexCNN0(
            oversampling=int(config.upsample_factor),
            model=self.model_variant,
        ).to(self.device)
        self.model.load_state_dict(self._load_state_dict())
        self.model.eval()

        self.provenance = RFSRProvenance(
            repo_root=str(self.repo_root),
            repo_commit=_git_commit(self.repo_root),
            checkpoint=str(self.checkpoint),
            checkpoint_sha256=_sha256(self.checkpoint),
            model_variant=self.model_variant,
            upsample_factor=int(config.upsample_factor),
            device=str(self.device),
        )

    def _validate_paths(self) -> None:
        if not (self.repo_root / "rfsr" / "interp.py").is_file():
            raise FileNotFoundError(
                f"not an RF-SR checkout (missing rfsr/interp.py): {self.repo_root}"
            )
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"RF-SR checkpoint not found: {self.checkpoint}")

    def _load_state_dict(self):
        try:
            return self._torch.load(
                self.checkpoint,
                map_location=self.device,
                weights_only=True,
            )
        except TypeError:  # PyTorch before weights_only was introduced.
            return self._torch.load(self.checkpoint, map_location=self.device)

    def _input_tensor(self, samples: np.ndarray):
        values = np.asarray(samples, dtype=np.complex64)
        if values.ndim != 1:
            raise ValueError(f"IQ input must be one-dimensional, got {values.shape}")
        channels = np.stack((values.real, values.imag), axis=0).astype(
            np.float32, copy=False
        )
        return self._torch.from_numpy(channels).unsqueeze(0).to(self.device)

    def _run_tensor(self, samples: np.ndarray, mode: FrontendMode, snr_db: float):
        tensor = self._input_tensor(samples)
        with self._torch.inference_mode():
            if mode == "interpolation":
                output = self._interpolate_tensor(
                    tensor, int(self.config.upsample_factor), 1
                )
            elif mode == "rfsr":
                output = self.model(tensor, float(snr_db))
            else:
                raise ValueError(f"unknown RF-SR frontend mode: {mode}")
        channels = output[0].detach().to("cpu").numpy()
        result = np.asarray(channels[0] + 1j * channels[1], dtype=np.complex64)
        expected = int(np.asarray(samples).size) * int(self.config.upsample_factor)
        if result.size != expected:
            raise RuntimeError(f"RF-SR returned {result.size} samples, expected {expected}")
        if not np.all(np.isfinite(result)):
            raise RuntimeError("RF-SR returned non-finite IQ samples")
        return result

    def transform(
        self,
        samples: np.ndarray,
        mode: FrontendMode,
        snr_db: float = 0.0,
    ) -> np.ndarray:
        """Transform complex IQ with overlap-cropped bounded-memory inference."""

        values = np.asarray(samples, dtype=np.complex64)
        if values.ndim != 1:
            raise ValueError(f"IQ input must be one-dimensional, got {values.shape}")
        if values.size == 0:
            return np.empty(0, dtype=np.complex64)
        chunk = int(self.config.chunk_input_samples)
        if values.size <= chunk:
            return self._run_tensor(values, mode, float(snr_db))

        up = int(self.config.upsample_factor)
        overlap = int(self.config.overlap_input_samples)
        output = np.empty(values.size * up, dtype=np.complex64)
        for core_start in range(0, int(values.size), chunk):
            core_stop = min(int(values.size), core_start + chunk)
            extended_start = max(0, core_start - overlap)
            extended_stop = min(int(values.size), core_stop + overlap)
            extended = self._run_tensor(
                values[extended_start:extended_stop], mode, float(snr_db)
            )
            crop_start = (core_start - extended_start) * up
            crop_stop = crop_start + (core_stop - core_start) * up
            output[core_start * up : core_stop * up] = extended[crop_start:crop_stop]
        return output

    def interpolate(self, samples: np.ndarray) -> np.ndarray:
        """Return the exact polyphase baseline used inside the author model."""

        return self.transform(samples, "interpolation")

    def enhance(self, samples: np.ndarray, snr_db: float = 0.0) -> np.ndarray:
        """Return polyphase interpolation plus the pretrained CNN residual."""

        return self.transform(samples, "rfsr", snr_db=float(snr_db))
