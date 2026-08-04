from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch


RFSR_ROOT = Path(__file__).resolve().parents[1]
if str(RFSR_ROOT) not in sys.path:
    sys.path.insert(0, str(RFSR_ROOT))

from rfsr.PHY import lora_chirp  # noqa: E402
from rfsr.nn.official_ota_dataset import (  # noqa: E402
    BRANCH_FFT_BINS,
    HIGH_SAMPLES_PER_SYMBOL,
    OFFICIAL_DATA_SYMBOL_OFFSET,
    OFFICIAL_GUARD_HIGH,
    OFFICIAL_RISE_HIGH,
    OfficialOTASymbolDataset,
    deterministic_reference_splits,
    scan_official_ota_records,
)
from rfsr.nn.task_loss import TaskAwareRFSRLoss  # noqa: E402
from rfsr.nn.task_model import (  # noqa: E402
    TaskAwareRFSRFrontend,
    TaskAwarePolyphaseTCN,
    input_rms_scale,
    linear_polyphase_baseline,
    sinc_polyphase_baseline,
)
from rfsr.nn.task_pretraining_dataset import (  # noqa: E402
    TaskAwareSyntheticSymbolDataset,
)
from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (  # noqa: E402
    paper_oversampled_spectrum,
)


class TaskAwareModelTest(unittest.TestCase):
    def test_output_shape_gradient_and_hard_preservation(self):
        torch.manual_seed(3)
        model = TaskAwarePolyphaseTCN(channels=8, dilations=(1, 2, 4))
        x = torch.randn(2, 2, 97, requires_grad=True)
        output = model(x)
        self.assertEqual(tuple(output.shape), (2, 2, 388))
        torch.testing.assert_close(output[..., ::4], x, rtol=0.0, atol=0.0)
        output[..., 1::4].square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        # The residual head is deliberately zero-initialized, so the first
        # backward pass trains that head while keeping the initial mapping an
        # exact interpolation baseline.
        self.assertGreater(float(model.head.weight.grad.abs().sum()), 0.0)

    def test_soft_observed_model_can_denoise_all_four_phases(self):
        torch.manual_seed(4)
        model = TaskAwarePolyphaseTCN(
            channels=8,
            dilations=(1, 2, 4),
            hard_observed=False,
        )
        self.assertEqual(model.head.out_channels, 8)
        x = torch.randn(2, 2, 41)
        initial = model(x)
        torch.testing.assert_close(
            initial, model.interpolation_baseline(x), rtol=0.0, atol=0.0
        )
        with torch.no_grad():
            model.head.bias[0] = 0.25
        corrected = model(x)
        self.assertGreater(float((corrected[..., ::4] - x).abs().max()), 0.0)

    def test_input_scale_uses_only_x_and_normalizes_complex_rms(self):
        x = torch.tensor(
            [[[3.0, 0.0], [4.0, 5.0]], [[0.0, 2.0], [0.0, 0.0]]]
        )
        scale = input_rms_scale(x)
        normalized_power = (x / scale).square().sum(dim=1).mean(dim=-1)
        torch.testing.assert_close(normalized_power, torch.ones_like(normalized_power))

    def test_chunked_numpy_frontend_preserves_length_and_observations(self):
        model = TaskAwarePolyphaseTCN(channels=8, dilations=(1, 2, 4))
        frontend = TaskAwareRFSRFrontend(
            model, device="cpu", chunk_input_samples=31
        )
        rng = np.random.default_rng(17)
        samples = (
            rng.standard_normal(103) + 1j * rng.standard_normal(103)
        ).astype(np.complex64)
        output = frontend.enhance(samples)
        self.assertEqual(output.shape, (412,))
        np.testing.assert_array_equal(output[::4], samples)

    def test_sinc_baseline_improves_bandlimited_tone_and_is_hard_consistent(self):
        n = torch.arange(256, dtype=torch.float32)
        frequency = 0.08
        x = torch.stack(
            (
                torch.cos(2 * torch.pi * frequency * n),
                torch.sin(2 * torch.pi * frequency * n),
            )
        )[None]
        high_n = torch.arange(1024, dtype=torch.float32) / 4.0
        truth = torch.stack(
            (
                torch.cos(2 * torch.pi * frequency * high_n),
                torch.sin(2 * torch.pi * frequency * high_n),
            )
        )[None]
        linear = linear_polyphase_baseline(x)
        sinc = sinc_polyphase_baseline(x)
        interior = slice(80, -80)
        linear_error = (linear[..., interior] - truth[..., interior]).square().mean()
        sinc_error = (sinc[..., interior] - truth[..., interior]).square().mean()
        self.assertLess(float(sinc_error), float(linear_error) / 100.0)
        torch.testing.assert_close(sinc[..., ::4], x, rtol=0.0, atol=0.0)


class TaskAwareLossTest(unittest.TestCase):
    def test_correct_clean_chirp_has_better_task_loss_than_wrong_bin(self):
        objective = TaskAwareRFSRLoss()
        chirp, _ = lora_chirp(+1, 351, 125_000, 4096, 8, 0, 0)
        prediction = torch.from_numpy(
            np.stack((chirp.real, chirp.imag), axis=0)[None].copy()
        )
        power = objective.four_branch_power(prediction)
        peak = power.argmax(dim=1)
        good = objective.task_losses(power, peak)
        bad = objective.task_losses(power, (peak + 100).remainder(BRANCH_FFT_BINS))
        self.assertLess(float(good[0]), float(bad[0]))
        self.assertLess(float(good[1]), float(bad[1]))

    def test_charbonnier_is_complex_magnitude_not_separate_iq_mae(self):
        objective = TaskAwareRFSRLoss(charbonnier_epsilon=0.0)
        prediction = torch.tensor([[[3.0], [4.0]]])
        target = torch.zeros_like(prediction)
        self.assertEqual(float(objective.waveform_loss(prediction, target)), 5.0)

    def test_differentiable_savaux_matches_reference_and_backpropagates(self):
        rng = np.random.default_rng(91)
        samples = (
            rng.standard_normal(32768) + 1j * rng.standard_normal(32768)
        ).astype(np.complex64)
        prediction = torch.from_numpy(
            np.stack((samples.real, samples.imag), axis=0)[None].copy()
        ).requires_grad_(True)
        objective = TaskAwareRFSRLoss(spectral_mode="savaux")
        actual = objective.savaux_coherent_spectrum(prediction)
        expected, _, _ = paper_oversampled_spectrum(samples, 0, 12, 8)
        np.testing.assert_allclose(
            actual.detach().numpy()[0], expected, rtol=2e-5, atol=5e-6
        )
        actual.abs().square().mean().backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(float(prediction.grad.abs().sum()), 0.0)


class SyntheticPretrainingDatasetTest(unittest.TestCase):
    def test_random_payload_channel_shapes_mask_and_bin_truth(self):
        dataset = TaskAwareSyntheticSymbolDataset(
            item_count=2, symbols_per_item=3, seed=8
        )
        item = dataset[0]
        self.assertEqual(tuple(item["x"].shape), (3, 2, 8192))
        self.assertEqual(tuple(item["y"].shape), (3, 2, 32768))
        self.assertEqual(tuple(item["valid_mask"].shape), (3, 32768))
        self.assertEqual(float(item["valid_mask"][..., ::4].max()), 0.0)
        objective = TaskAwareRFSRLoss(spectral_mode="savaux")
        peaks = objective.savaux_coherent_power(item["y"]).argmax(dim=1)
        distance = torch.minimum(
            (peaks - item["correct_bins"]).remainder(4096),
            (item["correct_bins"] - peaks).remainder(4096),
        )
        self.assertLessEqual(int(distance.max()), 1)


def _sparse_complex_file(path: Path, sample_count: int) -> None:
    with path.open("wb") as handle:
        handle.truncate(sample_count * np.dtype("<c8").itemsize)


class OfficialArchiveDatasetTest(unittest.TestCase):
    def _build_archive(self, root: Path, references: int = 5) -> None:
        for name in ("metadata", "ota", "reference"):
            (root / name).mkdir(parents=True, exist_ok=True)
        sample_count = 3_434_456 + 2 * OFFICIAL_GUARD_HIGH
        for reference_id in range(references):
            reference = (
                root
                / "reference"
                / f"signalout_{reference_id:06d}_fulltrim.cfile"
            )
            ota = root / "ota" / f"exp0_{reference_id:06d}_rxg30_0_fulltrim.cfile"
            _sparse_complex_file(reference, sample_count)
            _sparse_complex_file(ota, sample_count)
            metadata = {
                "payload": list(range(16)),
                "center_freq": 923_000_000.0,
                "sf": 12,
                "bw": 125_000.0,
                "sample_rate": 2_000_000.0,
                "src": 0,
                "dst": 1,
                "seqn": 7,
                "cr": 4,
                "enable_crc": 1,
                "implicit_header": 0,
                "preamble_bits": 8,
                "num_samples": 3_434_456,
                "files": [[f"ota/{ota.name}", 5.0]],
            }
            (root / "metadata" / f"{reference_id:06d}.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )

    def test_reference_group_split_is_disjoint_and_covers_all_ids(self):
        splits = deterministic_reference_splits(range(10), seed=9)
        values = {name: set(group) for name, group in splits.items()}
        self.assertFalse(values["train"] & values["validation"])
        self.assertFalse(values["train"] & values["test"])
        self.assertFalse(values["validation"] & values["test"])
        self.assertEqual(set().union(*values.values()), set(range(10)))

    def test_direct_scan_and_received_target_polyphase_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._build_archive(root)
            split_records = {}
            split_ids = {}
            for split in ("train", "validation", "test"):
                records, splits = scan_official_ota_records(
                    root, split=split, split_seed=11
                )
                split_records[split] = records
                split_ids[split] = set(splits[split])
            self.assertFalse(split_ids["train"] & split_ids["test"])
            self.assertEqual(sum(map(len, split_records.values())), 5)

            dataset = OfficialOTASymbolDataset(
                root,
                split="train",
                split_seed=11,
                symbols_per_capture=1,
            )
            record = dataset.records[0]
            source = np.memmap(record.ota_path, dtype="<c8", mode="r+")
            # Put a valid preamble into the sparse fixture so CFO estimation is
            # deterministic, and nonzero data into every possible phase view.
            preamble, _ = lora_chirp(+1, 0, 125_000, 4096, 16, 0, 0)
            preamble_start = OFFICIAL_GUARD_HIGH + OFFICIAL_RISE_HIGH
            for index in range(8):
                start = preamble_start + index * HIGH_SAMPLES_PER_SYMBOL
                source[start : start + HIGH_SAMPLES_PER_SYMBOL] = preamble
            data_start = int(
                OFFICIAL_GUARD_HIGH
                + OFFICIAL_RISE_HIGH
                + OFFICIAL_DATA_SYMBOL_OFFSET * HIGH_SAMPLES_PER_SYMBOL
            )
            generator = np.random.default_rng(5)
            source[
                data_start : data_start
                + len(record.raw_symbol_k) * HIGH_SAMPLES_PER_SYMBOL
                + 8
            ] = (
                generator.standard_normal(
                    len(record.raw_symbol_k) * HIGH_SAMPLES_PER_SYMBOL + 8
                )
                + 1j
                * generator.standard_normal(
                    len(record.raw_symbol_k) * HIGH_SAMPLES_PER_SYMBOL + 8
                )
            ).astype(np.complex64)
            source.flush()

            item = dataset[0]
            self.assertEqual(tuple(item["x"].shape), (1, 2, 8192))
            self.assertEqual(tuple(item["y"].shape), (1, 2, 32768))
            torch.testing.assert_close(
                item["x"], item["y"][..., ::4], rtol=0.0, atol=0.0
            )


if __name__ == "__main__":
    unittest.main()
