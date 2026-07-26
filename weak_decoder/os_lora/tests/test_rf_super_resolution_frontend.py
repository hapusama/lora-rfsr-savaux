"""Optional integration checks for the external RF-SR checkout."""

from __future__ import annotations

import os
from pathlib import Path
import unittest

import numpy as np

from weak_decoder.rf_super_resolution import (
    DEFAULT_SYNTHETIC_CHECKPOINT,
    RFSRFrontendConfig,
    RFSuperResolutionFrontend,
    default_rfsr_repo_root,
)


RFSR_REPO = Path(
    os.environ.get("RFSR_REPO", "").strip() or default_rfsr_repo_root()
).resolve()
RFSR_CHECKPOINT = RFSR_REPO / "checkpoints" / DEFAULT_SYNTHETIC_CHECKPOINT


@unittest.skipUnless(
    RFSR_CHECKPOINT.is_file(),
    f"RF-SR author checkpoint not found: {RFSR_CHECKPOINT}",
)
class RFSuperResolutionFrontendTests(unittest.TestCase):
    def test_shapes_and_chunk_boundaries(self) -> None:
        repo = RFSR_REPO
        rng = np.random.default_rng(1701)
        values = (
            rng.normal(size=300) + 1j * rng.normal(size=300)
        ).astype(np.complex64)
        common = {
            "repo_root": repo,
            "checkpoint_name": DEFAULT_SYNTHETIC_CHECKPOINT,
            "device": "cpu",
        }
        whole = RFSuperResolutionFrontend(
            RFSRFrontendConfig(**common, chunk_input_samples=1_000)
        )
        chunked = RFSuperResolutionFrontend(
            RFSRFrontendConfig(
                **common,
                chunk_input_samples=97,
                overlap_input_samples=68,
            )
        )
        for mode in ("interpolation", "rfsr"):
            expected = whole.transform(values, mode, snr_db=-20.0)
            actual = chunked.transform(values, mode, snr_db=-20.0)
            self.assertEqual((1_200,), actual.shape)
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=5e-7)


if __name__ == "__main__":
    unittest.main()
