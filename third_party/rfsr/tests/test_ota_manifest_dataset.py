from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


RFSR_ROOT = Path(__file__).resolve().parents[1]
if str(RFSR_ROOT) not in sys.path:
    sys.path.insert(0, str(RFSR_ROOT))

from rfsr.nn.ota_dataset import OTALoRaDataset  # noqa: E402


class OTAManifestDatasetTest(unittest.TestCase):
    def test_polyphase_views_and_group_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifests").mkdir()
            (root / "ota").mkdir()
            (root / "reference").mkdir()
            (root / "metadata").mkdir()

            reference = (
                np.arange(16, dtype=np.float32)
                + 1j * np.arange(16, dtype=np.float32)
            ).astype(np.complex64)
            reference.tofile(root / "reference" / "ref.cfile")

            rows = []
            for packet in range(2):
                ota_name = f"exp0_{packet:06d}_rxg20_0_fulltrim.cfile"
                (reference + packet).tofile(root / "ota" / ota_name)
                (root / "metadata" / f"{Path(ota_name).stem}.json").write_text(
                    json.dumps(
                        {
                            "schema": "lora-rfsr-ota-view",
                            "alignment": {"grlora_snr_db": -5.0 + packet},
                        }
                    ),
                    encoding="utf-8",
                )
                for phase in range(4):
                    rows.append(
                        {
                            "view_id": f"packet{packet}:q{phase}",
                            "split_group": f"packet{packet}",
                            "ota_path": f"ota/{ota_name}",
                            "reference_path": "reference/ref.cfile",
                            "lowrate_phase_1m": phase,
                            "target_samples": 16,
                            "input_samples": 4,
                        }
                    )

            manifest = root / "manifests" / "views.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)

            training = OTALoRaDataset(
                dataset_root=root,
                training=True,
                test_split=0.5,
                return_snr=True,
            )
            testing = OTALoRaDataset(
                dataset_root=root,
                training=False,
                test_split=0.5,
                return_snr=True,
            )
            self.assertEqual(len(training), 4)
            self.assertEqual(len(testing), 4)
            train_groups = {
                str(record["split_group"]) for record in training.records
            }
            test_groups = {
                str(record["split_group"]) for record in testing.records
            }
            self.assertFalse(train_groups & test_groups)

            x, y, snr = training[0]
            self.assertEqual(tuple(x.shape), (2, 4))
            self.assertEqual(tuple(y.shape), (2, 16))
            self.assertEqual(tuple(snr.shape), (1,))

    def test_rejects_manifest_path_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifests").mkdir()
            manifest = root / "manifests" / "views.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "view_id",
                        "split_group",
                        "ota_path",
                        "reference_path",
                        "lowrate_phase_1m",
                        "target_samples",
                        "input_samples",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "view_id": "bad",
                        "split_group": "bad",
                        "ota_path": "../outside.cfile",
                        "reference_path": "reference/ref.cfile",
                        "lowrate_phase_1m": 0,
                        "target_samples": 16,
                        "input_samples": 4,
                    }
                )
            with self.assertRaisesRegex(ValueError, "escapes"):
                OTALoRaDataset(
                    dataset_root=root,
                    training=True,
                    test_split=0.0,
                )


if __name__ == "__main__":
    unittest.main()
