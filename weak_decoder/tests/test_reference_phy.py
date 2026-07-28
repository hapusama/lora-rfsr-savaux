from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

import numpy as np

from tools.plot_reference_phy_stft import build_symbol_spans
from weak_decoder.rf_super_resolution.reference_phy import (
    ReferencePhyConfig,
    encode_reference_phy,
    parse_uart_reference_log,
    phy_config_from_uart,
    write_reference_packet,
)


FRAME0 = bytes.fromhex(
    "40 44 33 22 11 00 01 00 58 "
    "00 00 B9 88 3A 2D 11 8A A8 70 66 A9 9F 80 48 26 E7 52 29 AF "
    "78 56 34 12"
)
APP0 = bytes.fromhex(
    "00 00 B9 88 3A 2D 11 8A A8 70 66 A9 9F 80 48 26 E7 52 29 AF"
)


def _uart_text() -> str:
    app_hex = APP0.hex(" ").upper()
    frame_hex = FRAME0.hex(" ").upper()
    return (
        f"[TX Payload] seq=0 round=0 id=0 app_len=20 data={app_hex}\n"
        f"[TX Frame] seq=0 id=0 frame_len=33 data={frame_hex}\n"
        "[TX Frame Fields] seq=0 id=0 mhdr=0x40 devaddr=0x11223344 "
        "fctrl=0x00 fcnt=1 fport=88 app_len=20 mic=0x12345678 "
        "radio_send_len=33\n"
        "[TX PHY] modem=LORA configured_freq_hz=487700000 "
        "actual_freq_hz=487699951 sf=12 bw_hz=125000 cr=4/8 "
        "preamble_symbols=16 syncword=0x12 public_network=0 "
        "header=explicit phy_payload_len=33 phy_crc=1 ldro=1 "
        "iq_inverted=0 freq_hop=0 hop_period=0 tx_power_dbm=2 "
        "toa_ms=2761 tx_period_ms=6000\n"
        "[TX Reference Config] generator=xorshift32 seed=0x52465352 "
        "variants=1 id_encoding=uint16_le app_payload_len=20 "
        "radio_send_len=33 fixed_fcnt=1 unencrypted=1 mic_valid=0\n"
        "[TX PHY Reg 2] ModemConfig1=0x78 ModemConfig2=0xC4 "
        "ModemConfig3=0x0C Preamble=0x0010 PayloadLength=0x21 "
        "SyncWord=0x12\n"
    )


class ReferencePhyTest(unittest.TestCase):
    def test_current_raw_frame_has_rfsr_leading_silence(self) -> None:
        config = ReferencePhyConfig()
        encoded = encode_reference_phy(FRAME0, config)

        expected_symbols = 16 + 2 + 2.25 + 8 + 56
        on_air_samples = int(expected_symbols * config.samples_per_symbol)
        expected_samples = config.leading_silence_samples + on_air_samples
        self.assertEqual(on_air_samples, 2_760_704)
        self.assertEqual(expected_samples, 2_770_704)
        self.assertEqual(encoded.samples.size, expected_samples)
        self.assertEqual(encoded.samples.dtype, np.dtype("<c8"))
        self.assertEqual(len(encoded.header_symbol_ids), 8)
        self.assertEqual(len(encoded.payload_symbol_ids), 56)
        self.assertTrue(
            np.all(encoded.samples[: config.leading_silence_samples] == 0)
        )
        self.assertAlmostEqual(
            float(abs(encoded.samples[config.leading_silence_samples])),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(float(abs(encoded.samples[-1])), 1.0, places=5)

    def test_leading_silence_can_be_disabled_explicitly(self) -> None:
        config = ReferencePhyConfig(leading_silence_samples=0)
        encoded = encode_reference_phy(FRAME0, config)

        self.assertEqual(encoded.samples.size, 2_760_704)
        self.assertAlmostEqual(float(abs(encoded.samples[0])), 1.0, places=6)

    def test_uart_parser_and_writer_keep_raw_33_byte_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            uart_path = root / "packet_reference.txt"
            uart_path.write_text(_uart_text(), encoding="utf-8")
            uart_log = parse_uart_reference_log(uart_path)
            packet = uart_log.first_reference_cycle()[0]
            config = phy_config_from_uart(uart_log)

            self.assertEqual(packet.frame, FRAME0)
            self.assertEqual(packet.app_payload, APP0)
            self.assertEqual(config.cr, 4)
            self.assertEqual(config.preamble_symbols, 16)
            self.assertEqual(config.crc_mode, "grlora")

            iq_path, metadata_path = write_reference_packet(
                uart_log,
                packet,
                config,
                output_root=root / "rfsr_db",
            )
            iq = np.fromfile(iq_path, dtype=np.dtype("<c8"))
            self.assertEqual(iq.size, 2_770_704)
            self.assertTrue(np.all(iq[:10_000] == 0))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["schema_version"], 2)
            self.assertEqual(metadata["packet"]["frame_bytes"], 33)
            self.assertEqual(metadata["iq"]["leading_silence_samples"], 10_000)
            self.assertEqual(metadata["iq"]["trailing_silence_samples"], 0)
            self.assertEqual(metadata["iq"]["artificial_cfo_hz"], 0.0)
            self.assertEqual(metadata["phy"]["crc_mode"], "grlora")
            self.assertEqual(
                metadata["generator"]["phy_backend"],
                "rfsr.PHY.encode_raw_phy",
            )

            spans = build_symbol_spans(metadata, sample_count=int(iq.size))
            self.assertEqual(len(spans), 16 + 2 + 3 + 8 + 56)
            self.assertEqual(sum(span.section == "preamble" for span in spans), 16)
            self.assertEqual(sum(span.section == "sync" for span in spans), 2)
            self.assertEqual(sum(span.section == "sfd" for span in spans), 3)
            self.assertEqual(sum(span.section == "header" for span in spans), 8)
            self.assertEqual(sum(span.section == "payload" for span in spans), 56)
            self.assertEqual(spans[0].start_sample, 10_000)
            self.assertEqual(
                next(span.start_sample for span in spans if span.section == "header"),
                673_552,
            )
            self.assertEqual(spans[-1].stop_sample, iq.size)

            with self.assertRaises(FileExistsError):
                write_reference_packet(
                    uart_log,
                    packet,
                    config,
                    output_root=root / "rfsr_db",
                )


if __name__ == "__main__":
    unittest.main()
