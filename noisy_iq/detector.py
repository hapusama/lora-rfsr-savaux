"""GNU Radio based packet detection and payload measurement.

中文说明：这一层负责把 gr-lora_sdr 接收链跑一遍，收集 frame_sync、
header_decoder、crc_verif 发布的消息，并整理成 packet 级 metadata。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import errno
import gc
import hashlib
import math
import os
from pathlib import Path
import threading
from typing import Any

import numpy as np

from .constants import FILE_SOURCE_STAGING_DIR
from .iq_file import load_complex64_memmap
from .utils import payload_bytes_to_text


def prepare_file_source_path(input_file: Path) -> Path:
    """Create an ASCII-only hardlink for GNU Radio file_source on Windows."""
    source_path = Path(input_file).resolve()
    FILE_SOURCE_STAGING_DIR.mkdir(parents=True, exist_ok=True)
    # 用源文件绝对路径算 hash，避免不同目录下同名 IQ 文件互相覆盖 staging 链接。
    digest = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()
    staged_path = FILE_SOURCE_STAGING_DIR / f"{digest}{source_path.suffix.lower()}"

    if staged_path.exists():
        try:
            if staged_path.samefile(source_path):
                return staged_path
        except OSError:
            pass
        staged_path.unlink()

    try:
        os.link(source_path, staged_path)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise RuntimeError(
                "failed to create ASCII hardlink for GNU Radio file_source: "
                f"{source_path}"
            ) from exc
        # The experiment data commonly live on /root/autodl-tmp while the
        # ASCII staging directory is on the repository filesystem. A symlink
        # preserves the source file and works across those mount points.
        try:
            os.symlink(source_path, staged_path)
        except OSError as symlink_exc:
            raise RuntimeError(
                "failed to create ASCII file_source link for GNU Radio: "
                f"{source_path}"
            ) from symlink_exc
    return staged_path


def cleanup_file_source_path(path: Path) -> None:
    """Remove the temporary GNU Radio file_source hardlink."""
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def resolve_ldro(sf: int, bw: float, ldro_mode: int) -> int:
    """Resolve low-data-rate optimization: 0 off, 1 on, 2 auto."""
    if int(ldro_mode) == 0:
        return 0
    if int(ldro_mode) == 1:
        return 1
    return 1 if ((1 << int(sf)) / float(bw)) >= 0.016 else 0


def lora_payload_symbol_count(
    sf: int,
    bw: float,
    cr: int,
    payload_len: int,
    has_crc: bool,
    impl_head: bool,
    ldro_mode: int,
) -> int:
    """Estimate payload symbols using the LoRa airtime formula."""
    sf = int(sf)
    cr = int(cr)
    payload_len = max(0, int(payload_len))
    crc = 1 if has_crc else 0
    ih = 1 if impl_head else 0
    de = resolve_ldro(sf, bw, ldro_mode)
    denominator = 4 * max(1, sf - 2 * de)
    numerator = 8 * payload_len - 4 * sf + 28 + 16 * crc - 20 * ih
    # 这里复用 LoRa airtime 公式估计 header 后面的 payload symbol 数。
    coded_blocks = max(math.ceil(numerator / denominator), 0)
    return 8 + coded_blocks * (cr + 4)


def estimate_packet_range(iq_size: int, frame: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Extend frame_sync preamble ranges into complete packet sample ranges."""
    sf = int(frame.get("sf", args.sf))
    bw = float(frame.get("bw", args.bw))
    samples_per_symbol = int(frame.get("samples_per_symbol", (1 << sf) * int(round(args.samp_rate / args.bw))))

    pay_len = int(frame.get("pay_len", args.pay_len))
    if pay_len < 0:
        pay_len = int(args.pay_len)
    cr = int(frame.get("cr", args.cr))
    if cr < 1:
        cr = int(args.cr)
    has_crc = bool(int(frame.get("crc", int(args.has_crc))))
    ldro_mode = int(frame.get("ldro_mode", args.ldro_mode))

    payload_symbols = lora_payload_symbol_count(
        sf,
        bw,
        cr,
        pay_len,
        has_crc,
        args.impl_head,
        ldro_mode,
    )
    # frame_sync 发布的 end_sample 只覆盖前导码/同步/SFD；
    # 完整 packet 结束位置要用 header 解出的 pay_len、CR、CRC 再向后推。
    packet_symbols = float(frame.get("preamble_len", args.preamble_len)) + 4.25 + float(payload_symbols)
    packet_start = max(0, int(frame["start_sample"]))
    preamble_end = max(packet_start, int(frame["end_sample"]))
    packet_end = packet_start + int(math.ceil(packet_symbols * samples_per_symbol))
    packet_end = max(preamble_end, min(int(iq_size), packet_end))

    result = dict(frame)
    result.update(
        {
            "payload_symbols": int(payload_symbols),
            "packet_symbols": float(packet_symbols),
            "packet_start_sample": int(packet_start),
            "packet_end_sample": int(packet_end),
            "packet_samples": int(max(0, packet_end - packet_start)),
        }
    )
    return result


def merge_detector_frames(
    frames: list[dict[str, Any]],
    headers: list[dict[str, Any]],
    payloads: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Merge frame_sync, header_decoder, and crc_verif metadata by frame_count."""
    # header_decoder 的 frame_info 有可能比 preamble 消息晚到，所以这里统一按 frame_count 合并。
    headers_by_id = {
        int(header["frame_count"]): header
        for header in headers
        if int(header.get("frame_count", -1)) >= 0 and int(header.get("header_err", 1)) == 0
    }
    merged = []
    for frame in frames:
        item = dict(frame)
        header = headers_by_id.get(int(frame.get("frame_count", -1)))
        if header:
            item.update(header)
        merged.append(item)

    if payloads:
        payloads_by_id = {
            int(payload["frame_count"]): payload
            for payload in payloads
            if int(payload.get("frame_count", -1)) >= 0
        }
        used_payload_indexes: set[int] = set()
        for index, item in enumerate(merged):
            payload = payloads_by_id.get(int(item.get("frame_count", -1)))
            if payload is None and index < len(payloads):
                payload = payloads[index]
                used_payload_indexes.add(index)
            if payload is not None:
                item.update(payload)
        if len(payloads) > len(merged):
            for index, payload in enumerate(payloads):
                if index in used_payload_indexes:
                    continue
                if int(payload.get("frame_count", -1)) in {
                    int(item.get("frame_count", -2)) for item in merged
                }:
                    continue
                merged.append(dict(payload))
    return merged


@dataclass
class GrloraPacketDetector:
    """Object wrapper for the short gr-lora_sdr receive chain used by this tool.

    GNU Radio 相关代码集中在这个类里，其他模块在 dry-run/total 模式下不需要导入 gnuradio。
    """

    args: argparse.Namespace

    def detect(self, input_path: Path) -> list[dict[str, Any]]:
        """Run gr-lora_sdr and return complete packet ranges plus decoded metadata."""
        try:
            from gnuradio import blocks, gr
            import gnuradio.lora_sdr as lora_sdr
            import pmt
        except ImportError as exc:
            raise RuntimeError(
                "gr-lora_sdr SNR measurement requires GNU Radio and gr-lora_sdr. "
                "Run this script in the gr-lora conda environment, or use --dry-run if you only need planned outputs."
            ) from exc

        args = self.args

        class DictMessageSink(gr.basic_block):
            """Small PMT dict reader shared by the three metadata sinks."""

            def _dict_value(self, msg, key, default=None):
                value = pmt.dict_ref(msg, pmt.intern(key), pmt.PMT_NIL)
                if pmt.is_null(value):
                    return default
                try:
                    return pmt.to_python(value)
                except Exception:
                    return default

        class PreambleMetadataSink(DictMessageSink):
            """Collect aligned preamble/SFD sample ranges from frame_sync."""

            def __init__(self):
                gr.basic_block.__init__(self, name="noisy_iq_preamble_sink", in_sig=None, out_sig=None)
                self.frames: list[dict[str, Any]] = []
                self._lock = threading.Lock()
                self.message_port_register_in(pmt.intern("preamble"))
                self.set_msg_handler(pmt.intern("preamble"), self.handle_preamble)

            def handle_preamble(self, msg):
                if not pmt.is_dict(msg):
                    return
                start_sample = self._dict_value(msg, "start_sample", None)
                end_sample = self._dict_value(msg, "end_sample", None)
                if start_sample is None or end_sample is None:
                    return
                frame = {
                    "frame_count": int(self._dict_value(msg, "frame_count", 0)),
                    "sf": int(self._dict_value(msg, "sf", args.sf)),
                    "bw": float(self._dict_value(msg, "bw", args.bw)),
                    "sample_rate": float(self._dict_value(msg, "sample_rate", args.samp_rate)),
                    "samples_per_symbol": int(
                        self._dict_value(
                            msg,
                            "samples_per_symbol",
                            (1 << int(args.sf)) * int(round(args.samp_rate / args.bw)),
                        )
                    ),
                    "preamble_len": int(self._dict_value(msg, "preamble_len", args.preamble_len)),
                    "start_sample": int(start_sample),
                    "end_sample": int(end_sample),
                    "n_samples": int(self._dict_value(msg, "n_samples", int(end_sample) - int(start_sample))),
                    "n_symbols": float(self._dict_value(msg, "n_symbols", 0.0)),
                    "grlora_snr_db": float(self._dict_value(msg, "snr_db", float("nan"))),
                    "cfo": float(self._dict_value(msg, "cfo", float("nan"))),
                    "sto": float(self._dict_value(msg, "sto", float("nan"))),
                    "sfo": float(self._dict_value(msg, "sfo", float("nan"))),
                    "netid1": int(self._dict_value(msg, "netid1", -1)),
                    "netid2": int(self._dict_value(msg, "netid2", -1)),
                }
                for source_key, output_key in (
                    ("cr", "cr"),
                    ("pay_len", "pay_len"),
                    ("crc", "crc"),
                    ("ldro_mode", "ldro_mode"),
                    ("err", "header_err"),
                ):
                    value = self._dict_value(msg, source_key, None)
                    if value is not None:
                        frame[output_key] = int(value)
                with self._lock:
                    self.frames.append(frame)

        class HeaderMetadataSink(DictMessageSink):
            """Collect PHY header fields decoded by header_decoder."""

            def __init__(self):
                gr.basic_block.__init__(self, name="noisy_iq_header_sink", in_sig=None, out_sig=None)
                self.headers: list[dict[str, Any]] = []
                self._lock = threading.Lock()
                self.message_port_register_in(pmt.intern("frame_info"))
                self.set_msg_handler(pmt.intern("frame_info"), self.handle_frame_info)

            def handle_frame_info(self, msg):
                if not pmt.is_dict(msg):
                    return
                header = {
                    "frame_count": int(self._dict_value(msg, "frame_count", -1)),
                    "cr": int(self._dict_value(msg, "cr", -1)),
                    "pay_len": int(self._dict_value(msg, "pay_len", -1)),
                    "crc": int(self._dict_value(msg, "crc", int(args.has_crc))),
                    "ldro_mode": int(self._dict_value(msg, "ldro_mode", args.ldro_mode)),
                    "header_err": int(self._dict_value(msg, "err", 1)),
                }
                for key in ("start_sample", "end_sample"):
                    value = self._dict_value(msg, key, None)
                    if value is not None:
                        header[key] = int(value)
                with self._lock:
                    self.headers.append(header)

        class PayloadMetadataSink(DictMessageSink):
            """Collect decoded payload and CRC status from crc_verif."""

            def __init__(self):
                gr.basic_block.__init__(self, name="noisy_iq_payload_sink", in_sig=None, out_sig=None)
                self.payloads: list[dict[str, Any]] = []
                self._lock = threading.Lock()
                self.message_port_register_in(pmt.intern("payload_metadata"))
                self.set_msg_handler(pmt.intern("payload_metadata"), self.handle_payload_metadata)

            def _payload_to_bytes(self, payload: Any) -> bytes:
                if payload is None:
                    return b""
                if isinstance(payload, bytes):
                    return payload
                if isinstance(payload, bytearray):
                    return bytes(payload)
                if isinstance(payload, str):
                    return payload.encode("latin-1", errors="replace")
                if isinstance(payload, np.ndarray):
                    return payload.astype(np.uint8, copy=False).tobytes()
                if isinstance(payload, (list, tuple)):
                    return bytes(int(item) & 0xFF for item in payload)
                return str(payload).encode("utf-8", errors="replace")

            def _payload_pmt_to_bytes(self, payload_pmt) -> bytes:
                if pmt.is_null(payload_pmt):
                    return b""
                if hasattr(pmt, "is_u8vector") and pmt.is_u8vector(payload_pmt):
                    return bytes(pmt.u8vector_elements(payload_pmt))
                if hasattr(pmt, "is_blob") and pmt.is_blob(payload_pmt):
                    return bytes(pmt.blob_data(payload_pmt))
                return self._payload_to_bytes(pmt.to_python(payload_pmt))

            def handle_payload_metadata(self, msg):
                if not pmt.is_dict(msg):
                    return
                payload_bytes_pmt = pmt.dict_ref(msg, pmt.intern("payload_bytes"), pmt.PMT_NIL)
                payload_pmt = pmt.dict_ref(msg, pmt.intern("payload"), pmt.PMT_NIL)
                payload_decode_error = ""
                try:
                    if not pmt.is_null(payload_bytes_pmt):
                        payload_bytes = self._payload_pmt_to_bytes(payload_bytes_pmt)
                    else:
                        payload_bytes = self._payload_pmt_to_bytes(payload_pmt)
                except Exception as exc:
                    payload_bytes = b""
                    payload_decode_error = f"{type(exc).__name__}: {exc}"
                payload = {
                    "frame_count": int(self._dict_value(msg, "frame_count", -1)),
                    "decoded_payload_len": int(self._dict_value(msg, "decoded_payload_len", len(payload_bytes))),
                    "decoded_payload_available": True,
                    "decoded_payload_hex": payload_bytes.hex(),
                    "decoded_payload_text": payload_bytes_to_text(payload_bytes),
                    "crc_valid": bool(self._dict_value(msg, "crc_valid", False)),
                }
                if payload_decode_error:
                    payload["payload_decode_error"] = payload_decode_error
                for key in (
                    "cr",
                    "pay_len",
                    "crc",
                    "ldro_mode",
                    "err",
                    "start_sample",
                    "end_sample",
                    "sf",
                    "samples_per_symbol",
                ):
                    value = self._dict_value(msg, key, None)
                    if value is not None:
                        output_key = "header_err" if key == "err" else key
                        try:
                            payload[output_key] = int(value)
                        except (TypeError, ValueError):
                            payload[output_key] = value
                with self._lock:
                    self.payloads.append(payload)

        class PacketDetectorTopBlock(gr.top_block):
            """Minimal receive chain: file_source -> sync/demod/decode -> crc_verif."""

            def __init__(self, file_source_path: Path):
                gr.top_block.__init__(self, "LoRa Packet Range Detector", catch_exceptions=True)
                os_factor = int(round(float(args.samp_rate) / float(args.bw)))
                min_buf = int(np.ceil(os_factor * ((1 << int(args.sf)) + 2)))

                self.file_source = blocks.file_source(gr.sizeof_gr_complex, str(file_source_path), False, 0, 0)
                self.file_source.set_min_output_buffer(min_buf)
                self.frame_sync = lora_sdr.frame_sync(
                    int(args.center_freq),
                    int(args.bw),
                    int(args.sf),
                    bool(args.impl_head),
                    [int(args.sync_word)],
                    os_factor,
                    int(args.preamble_len),
                )
                self.fft_demod = lora_sdr.fft_demod(bool(args.soft_decoding), bool(args.print_grlora))
                self.gray_mapping = lora_sdr.gray_mapping(bool(args.soft_decoding))
                self.deinterleaver = lora_sdr.deinterleaver(bool(args.soft_decoding))
                self.hamming_dec = lora_sdr.hamming_dec(bool(args.soft_decoding))
                self.header_decoder = lora_sdr.header_decoder(
                    bool(args.impl_head),
                    int(args.cr),
                    int(args.pay_len),
                    bool(args.has_crc),
                    int(args.ldro_mode),
                    bool(args.print_header),
                )
                self.dewhitening = lora_sdr.dewhitening()
                # Older local builds expose the extended Crc_mode API and
                # publish payload_metadata messages. Current upstream 3.10
                # instead exposes crc_verif(print_rx_msg, output_crc_check)
                # and writes decoded bytes/CRC status to two stream outputs.
                self.current_upstream_crc_api = not hasattr(
                    lora_sdr, "Crc_mode"
                )
                if self.current_upstream_crc_api:
                    self.crc_verif = lora_sdr.crc_verif(0, True)
                    self.payload_stream_sink = blocks.vector_sink_b()
                    self.crc_stream_sink = blocks.vector_sink_b()
                else:
                    crc_mode = (
                        lora_sdr.Crc_mode.SX1276
                        if int(args.crc_mode) == 1
                        else lora_sdr.Crc_mode.GRLORA
                    )
                    self.crc_verif = lora_sdr.crc_verif(0, False, crc_mode)
                self.preamble_sink = PreambleMetadataSink()
                self.header_sink = HeaderMetadataSink()
                self.payload_sink = PayloadMetadataSink()

                # 流式样本链路负责真正解调；消息链路只用于拿到包边界和 header/payload metadata。
                self.connect((self.file_source, 0), (self.frame_sync, 0))
                self.connect((self.frame_sync, 0), (self.fft_demod, 0))
                self.connect((self.fft_demod, 0), (self.gray_mapping, 0))
                self.connect((self.gray_mapping, 0), (self.deinterleaver, 0))
                self.connect((self.deinterleaver, 0), (self.hamming_dec, 0))
                self.connect((self.hamming_dec, 0), (self.header_decoder, 0))
                self.connect((self.header_decoder, 0), (self.dewhitening, 0))
                self.connect((self.dewhitening, 0), (self.crc_verif, 0))
                self.msg_connect((self.header_decoder, "frame_info"), (self.frame_sync, "frame_info"))
                self.msg_connect((self.header_decoder, "frame_info"), (self.header_sink, "frame_info"))
                if self.current_upstream_crc_api:
                    self.connect(
                        (self.crc_verif, 0), (self.payload_stream_sink, 0)
                    )
                    self.connect(
                        (self.crc_verif, 1), (self.crc_stream_sink, 0)
                    )
                else:
                    self.msg_connect(
                        (self.frame_sync, "preamble"),
                        (self.preamble_sink, "preamble"),
                    )
                    self.msg_connect(
                        (self.crc_verif, "payload_metadata"),
                        (self.payload_sink, "payload_metadata"),
                    )

        file_source_path = prepare_file_source_path(input_path)
        try:
            tb = PacketDetectorTopBlock(file_source_path)
            tb.start()
            tb.wait()
            if tb.current_upstream_crc_api:
                payload = bytes(tb.payload_stream_sink.data())
                crc_values = list(tb.crc_stream_sink.data())
                # Upstream 3.10 does not publish packet timing metadata. This
                # fallback is intended for one-packet trimmed cfiles, where
                # the complete stream output is one decoded frame.
                frames = (
                    [
                        {
                            "frame_count": 0,
                            "start_sample": 0,
                            "end_sample": 0,
                        }
                    ]
                    if payload or crc_values
                    else []
                )
                headers = []
                payloads = (
                    [
                        {
                            "frame_count": 0,
                            "decoded_payload_len": len(payload),
                            "decoded_payload_available": True,
                            "decoded_payload_hex": payload.hex(),
                            "decoded_payload_text": payload_bytes_to_text(
                                payload
                            ),
                            "crc_valid": bool(crc_values)
                            and all(int(value) != 0 for value in crc_values),
                        }
                    ]
                    if payload or crc_values
                    else []
                )
            else:
                frames = list(tb.preamble_sink.frames)
                headers = list(tb.header_sink.headers)
                payloads = list(tb.payload_sink.payloads)
        finally:
            # On Windows, file_source keeps the staged hardlink open until the
            # top block is destroyed.  Release it before unlinking so a
            # successful detect does not leave a 12+ GB-looking temp entry.
            if "tb" in locals():
                del tb
                gc.collect()
            cleanup_file_source_path(file_source_path)

        merged = merge_detector_frames(frames, headers, payloads)
        iq_size = load_complex64_memmap(input_path).size
        return [estimate_packet_range(iq_size, frame, args) for frame in merged]


def run_grlora_packet_detector(input_path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Compatibility function for callers that prefer the old functional API."""
    return GrloraPacketDetector(args=args).detect(input_path)
