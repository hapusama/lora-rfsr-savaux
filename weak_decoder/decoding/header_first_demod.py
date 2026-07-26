"""gr-lora_sdr 风格的 header-first FFT 解调工具。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from ..chirp import build_downchirp, bin_to_grlora_symbol, dechirp_fft, positive_mod, signed_fft_bin


@dataclass(frozen=True)
class SymbolDemodResult:
    """单个 header/payload LoRa symbol 的 FFT peak 解调结果。"""

    stage: str
    frame_symbol_index: int
    stage_symbol_index: int
    start_sample: int
    raw_fft_bin: int
    signed_fft_bin: int
    symbol_value: int
    peak_real: float
    peak_imag: float
    peak_amp: float
    peak_power: float
    peak_phase: float
    peak_margin_db: float
    total_power: float
    cfo_correction_mode: str
    cfo_common_phase_rad: float
    sfo_cum_before: float
    sfo_sample_adjust_after: int


@dataclass(frozen=True)
class HeaderDecodeResult:
    """PHY header hard-decision 解码结果。"""

    header_valid: bool
    payload_len: int
    cr: int
    has_crc: bool
    ldro: bool
    payload_symbol_count: int
    total_symbol_count: int
    header_checksum_received: int
    header_checksum_computed: int
    header_error: int
    header_nibbles: tuple[int, ...]
    gray_symbols: tuple[int, ...]
    codewords: tuple[int, ...]
    decoded_nibbles: tuple[int, ...]


def resolve_ldro(sf: int, bw: float, ldro_mode: int) -> bool:
    """按 gr-lora_sdr header_decoder 的规则解析 LDRO。"""

    mode = int(ldro_mode)
    if mode == 0:
        return False
    if mode == 1:
        return True
    return bool((float(1 << int(sf)) * 1e3 / float(bw)) > 16.0)


def payload_symbol_count(
    sf: int,
    bw: float,
    cr: int,
    payload_len: int,
    has_crc: bool,
    impl_head: bool,
    ldro_mode: int,
) -> tuple[int, bool]:
    """返回 header 后面的 payload/CRC 编码 symbol 数，以及最终 LDRO 状态。"""

    ldro = resolve_ldro(sf, bw, ldro_mode)
    numerator = (
        2 * int(payload_len)
        - int(sf)
        + 2
        + (0 if bool(impl_head) else 5)
        + (4 if bool(has_crc) else 0)
    )
    denominator = max(1, int(sf) - 2 * int(ldro))
    coded_blocks = max(0, int(math.ceil(float(numerator) / float(denominator))))
    return int(coded_blocks * (4 + int(cr))), bool(ldro)


def int_to_bits_msb(value: int, n_bits: int) -> list[bool]:
    """复刻 gr-lora_sdr utilities::int2bool，输出 MSB-first bit 列表。"""

    value = int(value)
    return [bool((value >> bit) & 1) for bit in range(int(n_bits) - 1, -1, -1)]


def bits_to_int(bits: Iterable[bool]) -> int:
    """复刻 gr-lora_sdr utilities::bool2int。"""

    result = 0
    for bit in bits:
        result = (result << 1) + int(bool(bit))
    return int(result)


def gray_demapping(symbol_values: Iterable[int]) -> list[int]:
    """复刻 gray_mapping_impl 的 hard path。"""

    return [int(value) ^ (int(value) >> 1) for value in symbol_values]


def deinterleave_hard(symbol_values: Iterable[int], sf: int, is_header: bool, cr: int, ldro: bool) -> list[int]:
    """复刻 deinterleaver_impl 的 hard path，输出 codeword 十进制值。"""

    values = [int(value) for value in symbol_values]
    sf_app = int(sf) - 2 if (bool(is_header) or bool(ldro)) else int(sf)
    cw_len = 8 if bool(is_header) else int(cr) + 4
    if len(values) < cw_len:
        raise ValueError(f"deinterleaver needs {cw_len} symbols, got {len(values)}.")

    inter_bin = [int_to_bits_msb(values[i], sf_app) for i in range(cw_len)]
    deinter_bin = [[False for _ in range(cw_len)] for _ in range(sf_app)]
    for i in range(cw_len):
        for j in range(sf_app):
            deinter_bin[positive_mod(i - j - 1, sf_app)][i] = inter_bin[i][j]
    return [bits_to_int(row) for row in deinter_bin]


def hamming_decode_hard(codewords: Iterable[int], is_header: bool, cr: int) -> list[int]:
    """复刻 hamming_dec_impl 的 hard path，输出 nibble。"""

    cr_app = 4 if bool(is_header) else int(cr)
    cw_len = cr_app + 4
    decoded: list[int] = []
    for value in codewords:
        codeword = int_to_bits_msb(int(value), cw_len)
        data_nibble = [codeword[3], codeword[2], codeword[1], codeword[0]]

        if cr_app == 4:
            # CR 4/8 先用整体奇偶校验决定是否尝试纠错。
            if sum(1 for bit in codeword if bit) % 2 == 0:
                decoded.append(bits_to_int(data_nibble))
                continue
            cr_to_handle = 3
        else:
            cr_to_handle = cr_app

        if cr_to_handle == 3:
            s0 = codeword[0] ^ codeword[1] ^ codeword[2] ^ codeword[4]
            s1 = codeword[1] ^ codeword[2] ^ codeword[3] ^ codeword[5]
            s2 = codeword[0] ^ codeword[1] ^ codeword[3] ^ codeword[6]
            syndrome = int(s0) + (int(s1) << 1) + (int(s2) << 2)
            if syndrome == 5:
                data_nibble[3] = not data_nibble[3]
            elif syndrome == 7:
                data_nibble[2] = not data_nibble[2]
            elif syndrome == 3:
                data_nibble[1] = not data_nibble[1]
            elif syndrome == 6:
                data_nibble[0] = not data_nibble[0]
        elif cr_to_handle == 2:
            # gr-lora_sdr 对 CR 4/6 hard path 不做数据位纠错。
            pass
        elif cr_to_handle == 1:
            # gr-lora_sdr 对 CR 4/5 hard path 不做数据位纠错。
            pass

        decoded.append(bits_to_int(data_nibble))
    return decoded


def compute_header_checksum(n0: int, n1: int, n2: int) -> int:
    """复刻 header_decoder_impl 的 5-bit PHY header checksum。"""

    c4 = ((n0 & 0b1000) >> 3) ^ ((n0 & 0b0100) >> 2) ^ ((n0 & 0b0010) >> 1) ^ (n0 & 0b0001)
    c3 = ((n0 & 0b1000) >> 3) ^ ((n1 & 0b1000) >> 3) ^ ((n1 & 0b0100) >> 2) ^ ((n1 & 0b0010) >> 1) ^ (n2 & 0b0001)
    c2 = ((n0 & 0b0100) >> 2) ^ ((n1 & 0b1000) >> 3) ^ (n1 & 0b0001) ^ ((n2 & 0b1000) >> 3) ^ ((n2 & 0b0010) >> 1)
    c1 = ((n0 & 0b0010) >> 1) ^ ((n1 & 0b0100) >> 2) ^ (n1 & 0b0001) ^ ((n2 & 0b0100) >> 2) ^ ((n2 & 0b0010) >> 1) ^ (n2 & 0b0001)
    c0 = (n0 & 0b0001) ^ ((n1 & 0b0010) >> 1) ^ ((n2 & 0b1000) >> 3) ^ ((n2 & 0b0100) >> 2) ^ ((n2 & 0b0010) >> 1) ^ (n2 & 0b0001)
    return int((c4 << 4) + (c3 << 3) + (c2 << 2) + (c1 << 1) + c0)


def decode_explicit_header(
    header_symbol_values: Iterable[int],
    sf: int,
    bw: float,
    ldro_mode: int,
) -> HeaderDecodeResult:
    """从 8 个 header symbol value 解出显式 PHY header。"""

    symbol_values = [int(value) for value in header_symbol_values]
    if len(symbol_values) != 8:
        raise ValueError(f"explicit header needs exactly 8 symbols, got {len(symbol_values)}.")

    gray_symbols = gray_demapping(symbol_values)
    codewords = deinterleave_hard(gray_symbols, sf=sf, is_header=True, cr=4, ldro=False)
    decoded = hamming_decode_hard(codewords, is_header=True, cr=4)
    if len(decoded) < 5:
        raise ValueError("decoded header has fewer than 5 nibbles.")

    payload_len = int((decoded[0] << 4) + decoded[1])
    has_crc = bool(decoded[2] & 1)
    cr = int(decoded[2] >> 1)
    checksum_received = int(((decoded[3] & 1) << 4) + decoded[4])
    checksum_computed = compute_header_checksum(decoded[0], decoded[1], decoded[2])
    header_error = int(checksum_received - checksum_computed)
    header_valid = bool(header_error == 0 and payload_len != 0)
    payload_symbols, ldro = payload_symbol_count(
        sf=sf,
        bw=bw,
        cr=cr,
        payload_len=payload_len,
        has_crc=has_crc,
        impl_head=False,
        ldro_mode=ldro_mode,
    )
    return HeaderDecodeResult(
        header_valid=header_valid,
        payload_len=payload_len,
        cr=cr,
        has_crc=has_crc,
        ldro=ldro,
        payload_symbol_count=payload_symbols if header_valid else 0,
        total_symbol_count=8 + payload_symbols if header_valid else 8,
        header_checksum_received=checksum_received,
        header_checksum_computed=checksum_computed,
        header_error=header_error,
        header_nibbles=tuple(int(item) for item in decoded[:5]),
        gray_symbols=tuple(gray_symbols),
        codewords=tuple(codewords),
        decoded_nibbles=tuple(decoded),
    )


def _symbol_sample_indexes(start_sample: int, sf: int, os_factor: int) -> np.ndarray:
    n_bins = 1 << int(sf)
    os_value = int(os_factor)
    return int(start_sample) + int(os_value / 2) + os_value * np.arange(n_bins, dtype=np.int64)


def demod_one_symbol(
    samples: np.ndarray,
    start_sample: int,
    sf: int,
    os_factor: int,
    downchirp: np.ndarray,
    stage: str,
    frame_symbol_index: int,
    stage_symbol_index: int,
    ldro: bool,
    cfo_correction_mode: str,
    cfo_common_phase_rad: float,
    sfo_cum_before: float,
    sfo_sample_adjust_after: int,
) -> SymbolDemodResult:
    """按 gr-lora_sdr 的 chip-rate downchirp 做一个 symbol 的 FFT argmax。"""

    indexes = _symbol_sample_indexes(start_sample, sf, os_factor)
    if int(indexes[0]) < 0 or int(indexes[-1]) >= samples.size:
        raise ValueError(f"symbol {frame_symbol_index} exceeds input sample range.")

    symbol = np.asarray(samples[indexes], dtype=np.complex64)
    if str(cfo_correction_mode) == "continuous":
        # gr-lora_sdr 的 downchirp 只消掉 symbol 内的 CFO 斜率；
        # 这里额外补偿从帧起点累计到当前 symbol 的公共 CFO 相位。
        symbol = (symbol * np.exp(-1j * float(cfo_common_phase_rad))).astype(np.complex64)
    spectrum = dechirp_fft(symbol, downchirp)
    power = np.abs(spectrum) ** 2
    raw_bin = int(np.argmax(power))
    peak = complex(spectrum[raw_bin])
    peak_power = float(power[raw_bin])
    second_power = float(np.partition(power, -2)[-2]) if power.size > 1 else 0.0
    total_power = float(np.sum(power, dtype=np.float64))
    is_header = str(stage) == "header"
    symbol_value = bin_to_grlora_symbol(raw_bin, sf=sf, is_header=is_header, ldro=ldro)

    return SymbolDemodResult(
        stage=str(stage),
        frame_symbol_index=int(frame_symbol_index),
        stage_symbol_index=int(stage_symbol_index),
        start_sample=int(start_sample),
        raw_fft_bin=raw_bin,
        signed_fft_bin=signed_fft_bin(raw_bin, 1 << int(sf)),
        symbol_value=int(symbol_value),
        peak_real=float(peak.real),
        peak_imag=float(peak.imag),
        peak_amp=float(abs(peak)),
        peak_power=peak_power,
        peak_phase=float(math.atan2(peak.imag, peak.real)),
        peak_margin_db=float(10.0 * math.log10((peak_power + 1e-30) / (second_power + 1e-30))),
        total_power=total_power,
        cfo_correction_mode=str(cfo_correction_mode),
        cfo_common_phase_rad=float(cfo_common_phase_rad),
        sfo_cum_before=float(sfo_cum_before),
        sfo_sample_adjust_after=int(sfo_sample_adjust_after),
    )


def advance_symbol_cursor(
    start_sample: int,
    sf: int,
    os_factor: int,
    sfo_cum: float,
    sfo_hat: float,
) -> tuple[int, float, int]:
    """复刻 frame_sync SFO_COMPENSATION 的逐符号 consume 调整。"""

    step = (1 << int(sf)) * int(os_factor)
    sample_adjust = 0
    threshold = 1.0 / (2.0 * int(os_factor))
    if abs(float(sfo_cum)) > threshold:
        sign = -1 if math.copysign(1.0, float(sfo_cum)) < 0.0 else 1
        step -= sign
        sample_adjust = -sign
        sfo_cum -= sign * (1.0 / int(os_factor))
    sfo_cum += float(sfo_hat)
    return int(start_sample + step), float(sfo_cum), int(sample_adjust)


def demod_symbol_sequence(
    samples: np.ndarray,
    header_start_sample: int,
    sf: int,
    os_factor: int,
    cfo_int: int,
    cfo_frac: float,
    sfo_hat: float,
    sfo_cum_initial: float,
    header_count: int,
    payload_count: int,
    payload_ldro: bool,
    cfo_correction_mode: str = "continuous",
) -> list[SymbolDemodResult]:
    """从 header 起点连续解调 header + payload FFT peak。"""

    mode = str(cfo_correction_mode)
    if mode not in {"symbol", "continuous"}:
        raise ValueError(f"unknown CFO correction mode: {mode}")

    downchirp = build_downchirp(sf, cfo_int=cfo_int, cfo_frac=cfo_frac)
    results: list[SymbolDemodResult] = []
    cursor = int(header_start_sample)
    sfo_cum = float(sfo_cum_initial)
    total_count = int(header_count) + int(payload_count)
    n_bins = 1 << int(sf)
    cfo_total = float(cfo_int) + float(cfo_frac)
    for frame_symbol_index in range(total_count):
        stage = "header" if frame_symbol_index < int(header_count) else "payload"
        stage_symbol_index = frame_symbol_index if stage == "header" else frame_symbol_index - int(header_count)
        sfo_before = float(sfo_cum)
        next_cursor, next_sfo_cum, sample_adjust = advance_symbol_cursor(
            cursor,
            sf=sf,
            os_factor=os_factor,
            sfo_cum=sfo_cum,
            sfo_hat=sfo_hat,
        )
        if mode == "continuous":
            relative_chip_start = float(cursor - int(header_start_sample)) / float(os_factor)
            cfo_common_phase_rad = float(2.0 * math.pi * cfo_total * relative_chip_start / n_bins)
        else:
            cfo_common_phase_rad = 0.0
        result = demod_one_symbol(
            samples=samples,
            start_sample=cursor,
            sf=sf,
            os_factor=os_factor,
            downchirp=downchirp,
            stage=stage,
            frame_symbol_index=frame_symbol_index,
            stage_symbol_index=stage_symbol_index,
            ldro=bool(payload_ldro),
            cfo_correction_mode=mode,
            cfo_common_phase_rad=cfo_common_phase_rad,
            sfo_cum_before=sfo_before,
            sfo_sample_adjust_after=sample_adjust,
        )
        results.append(result)
        cursor = next_cursor
        sfo_cum = next_sfo_cum
    return results
