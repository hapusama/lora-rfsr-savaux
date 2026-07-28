#!/usr/bin/env python3
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Copyright (c) 2022-23 Fabio Busacca <fabio.busacca@unict.it>
# Copyright (c) 2025-26 Andreas Kuster <S220003@e.ntu.edu.sg>

import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import matplotlib
import torch

matplotlib.use('Agg')  # non-GUI backend for command-line / batch plotting

PLOTTING = False
_internal = {}  # [Andreas]
_filename = "default"  # [Andreas]
_windowing = False  # [Andreas]
_zeropadding = False  # [Andreas]
_loratrimmer = False  # [Andreas]
_nelora = False
_gloriphy = False
gloriphy_model = None
gloriphy_opts = None
########### [Andreas]
# from rfsr.GLoRiPHY.Baseline import load_model_nelora, predict_nelora
nelora_model = None
neloar_opts = None
############


#############CONSTANTS#####################
# rising and falling edges duration
Trise = 50e-6
# Carrier Frequency Offset as declared in the datasheet
Cfo_PPM = 17
# Receiver Noise Figure
NF_dB = 6
# Minimum duration for a LoRa packet
min_time_lora_packet = 20e-3  # 20 milliseconds


##########################################################


@dataclass(frozen=True)
class RawPhyEncoding:
    """raw PHY payload 的理想 IQ 与调制器 symbol ID。"""

    payload: bytes
    cfo_hz: float
    samples: np.ndarray
    header_symbol_ids: tuple[int, ...]
    payload_symbol_ids: tuple[int, ...]


def _raw_whitening_sequence(length):
    """生成 gr-lora_sdr 使用的 8-bit whitening 序列。"""

    state = np.ones(8, dtype=np.uint8)
    feedback_mask = np.array([0, 0, 0, 1, 1, 1, 0, 1], dtype=np.uint8)
    sequence = []
    for _ in range(int(length)):
        sequence.append(
            sum(int(state[bit]) << bit for bit in range(8))
        )
        feedback = int(np.sum(state * feedback_mask) % 2)
        state = np.concatenate(
            (np.array([feedback], dtype=np.uint8), state[:-1])
        )
    return sequence


def _raw_crc16(payload):
    """CRC-16(poly=0x1021, init=0)，与 gr-lora_sdr 对齐。"""

    crc = 0
    for value in payload:
        byte = int(value) & 0xFF
        for _ in range(8):
            if (((crc & 0x8000) >> 8) ^ (byte & 0x80)):
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
            byte = (byte << 1) & 0xFF
    return int(crc)


def _raw_int_to_bits_msb(value, n_bits):
    return [
        (int(value) >> bit) & 1
        for bit in range(int(n_bits) - 1, -1, -1)
    ]


def _raw_bits_to_int(bits):
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return int(value)


def _raw_header_nibbles(payload_len, cr, has_crc):
    n0 = (int(payload_len) >> 4) & 0x0F
    n1 = int(payload_len) & 0x0F
    n2 = ((int(cr) & 0x07) << 1) | int(bool(has_crc))
    c4 = (
        ((n0 & 0b1000) >> 3)
        ^ ((n0 & 0b0100) >> 2)
        ^ ((n0 & 0b0010) >> 1)
        ^ (n0 & 0b0001)
    )
    c3 = (
        ((n0 & 0b1000) >> 3)
        ^ ((n1 & 0b1000) >> 3)
        ^ ((n1 & 0b0100) >> 2)
        ^ ((n1 & 0b0010) >> 1)
        ^ (n2 & 0b0001)
    )
    c2 = (
        ((n0 & 0b0100) >> 2)
        ^ ((n1 & 0b1000) >> 3)
        ^ (n1 & 0b0001)
        ^ ((n2 & 0b1000) >> 3)
        ^ ((n2 & 0b0010) >> 1)
    )
    c1 = (
        ((n0 & 0b0010) >> 1)
        ^ ((n1 & 0b0100) >> 2)
        ^ (n1 & 0b0001)
        ^ ((n2 & 0b0100) >> 2)
        ^ ((n2 & 0b0010) >> 1)
        ^ (n2 & 0b0001)
    )
    c0 = (
        (n0 & 0b0001)
        ^ ((n1 & 0b0010) >> 1)
        ^ ((n2 & 0b1000) >> 3)
        ^ ((n2 & 0b0100) >> 2)
        ^ ((n2 & 0b0010) >> 1)
        ^ (n2 & 0b0001)
    )
    return [
        n0,
        n1,
        n2,
        int(c4),
        int((c3 << 3) | (c2 << 2) | (c1 << 1) | c0),
    ]


def _raw_payload_nibbles(payload):
    whitening = _raw_whitening_sequence(len(payload))
    nibbles = []
    for offset, value in enumerate(payload):
        whitened = (int(value) & 0xFF) ^ int(whitening[offset])
        nibbles.extend((whitened & 0x0F, (whitened >> 4) & 0x0F))
    return nibbles


def _raw_crc_nibbles(payload, crc_mode):
    payload_bytes = bytes(int(value) & 0xFF for value in payload)
    mode = str(crc_mode).lower()
    if mode == "sx1276":
        crc = _raw_crc16(payload_bytes)
    elif mode == "grlora":
        crc = _raw_crc16(payload_bytes[: max(0, len(payload_bytes) - 2)])
        if len(payload_bytes) >= 2:
            crc ^= int(payload_bytes[-1])
            crc ^= int(payload_bytes[-2]) << 8
    else:
        raise ValueError(
            "crc_mode must be 'grlora' or 'sx1276', "
            f"got {crc_mode!r}."
        )
    return [
        crc & 0x000F,
        (crc & 0x00F0) >> 4,
        (crc & 0x0F00) >> 8,
        (crc & 0xF000) >> 12,
    ]


def _raw_hamming_nibble(nibble, cr):
    data = _raw_int_to_bits_msb(int(nibble) & 0x0F, 4)
    if int(cr) == 1:
        parity = data[0] ^ data[1] ^ data[2] ^ data[3]
        return int(
            (data[3] << 4)
            | (data[2] << 3)
            | (data[1] << 2)
            | (data[0] << 1)
            | parity
        )

    p0 = data[3] ^ data[2] ^ data[1]
    p1 = data[2] ^ data[1] ^ data[0]
    p2 = data[3] ^ data[2] ^ data[0]
    p3 = data[3] ^ data[1] ^ data[0]
    full = (
        (data[3] << 7)
        | (data[2] << 6)
        | (data[1] << 5)
        | (data[0] << 4)
        | (p0 << 3)
        | (p1 << 2)
        | (p2 << 1)
        | p3
    )
    return int(full >> (4 - int(cr)))


def _raw_hamming_encode(nibbles, sf, cr):
    return [
        _raw_hamming_nibble(
            nibble,
            4 if index < int(sf) - 2 else int(cr),
        )
        for index, nibble in enumerate(nibbles)
    ]


def _raw_interleave(codewords, sf, cr, ldro, frame_len_nibbles):
    values = [int(value) for value in codewords]
    codeword_count = 0
    position = 0
    symbols = []
    while position < len(values):
        first_or_ldro = codeword_count < int(sf) - 2 or bool(ldro)
        codeword_len = 4 + (
            4 if codeword_count < int(sf) - 2 else int(cr)
        )
        sf_app = int(sf) - 2 if first_or_ldro else int(sf)
        remaining = len(values) - position
        item_count = min(remaining, sf_app)
        if (
            item_count < sf_app
            and codeword_count + item_count != int(frame_len_nibbles)
        ):
            break

        codeword_bits = []
        for index in range(sf_app):
            if index >= item_count:
                codeword_bits.append(
                    _raw_int_to_bits_msb(0, codeword_len)
                )
            else:
                codeword_bits.append(
                    _raw_int_to_bits_msb(
                        values[position + index],
                        codeword_len,
                    )
                )
            codeword_count += 1
        position += item_count

        for row_index in range(codeword_len):
            row = [0 for _ in range(int(sf))]
            for column_index in range(sf_app):
                row[column_index] = codeword_bits[
                    (row_index - column_index - 1) % sf_app
                ][row_index]
            if codeword_count == int(sf) - 2 or bool(ldro):
                row[sf_app] = sum(row[:sf_app]) % 2
            symbols.append(_raw_bits_to_int(row))
    return symbols


def _raw_gray_map(symbols, sf):
    mask = (1 << int(sf)) - 1
    output = []
    for value in symbols:
        binary = int(value) & mask
        gray = binary
        for shift in range(1, int(sf)):
            gray ^= binary >> shift
        output.append((int(gray) + 1) & mask)
    return output


def encode_raw_phy_symbols(
    payload: bytes | bytearray | Sequence[int],
    *,
    sf=12,
    cr=4,
    enable_crc=1,
    ldro=True,
    crc_mode="grlora",
):
    """把 raw PHY payload 编码为 explicit-header LoRa symbol ID。"""

    payload_bytes = bytes(int(value) & 0xFF for value in payload)
    if not 1 <= len(payload_bytes) <= 255:
        raise ValueError(
            f"raw PHY payload length must be 1..255, got {len(payload_bytes)}."
        )
    if not 7 <= int(sf) <= 12:
        raise ValueError(f"SF must be in [7, 12], got {sf}.")
    if int(cr) not in {1, 2, 3, 4}:
        raise ValueError(f"CR must be 1..4, got {cr}.")

    nibbles = _raw_header_nibbles(
        len(payload_bytes),
        int(cr),
        bool(enable_crc),
    )
    nibbles.extend(_raw_payload_nibbles(payload_bytes))
    if bool(enable_crc):
        nibbles.extend(_raw_crc_nibbles(payload_bytes, crc_mode))

    codewords = _raw_hamming_encode(nibbles, int(sf), int(cr))
    interleaved = _raw_interleave(
        codewords,
        int(sf),
        int(cr),
        bool(ldro),
        len(nibbles),
    )
    symbols = _raw_gray_map(interleaved, int(sf))
    return tuple(symbols[:8]), tuple(symbols[8:])


def _raw_upchirp(sf, symbol_id, os_factor):
    """生成与 gr-lora_sdr 调制器一致的一条 upchirp。"""

    sf = int(sf)
    os_factor = int(os_factor)
    bins = 1 << sf
    symbol_id = int(symbol_id) % bins
    sample_index = np.arange(bins * os_factor, dtype=np.float64)
    fold = bins * os_factor - symbol_id * os_factor
    slope = (
        sample_index * sample_index
        / (2.0 * bins)
        / (os_factor * os_factor)
    )
    linear_before = (
        (symbol_id / bins - 0.5) * sample_index / os_factor
    )
    linear_after = (
        (symbol_id / bins - 1.5) * sample_index / os_factor
    )
    phase = np.where(
        sample_index < fold,
        slope + linear_before,
        slope + linear_after,
    )
    return np.exp(2j * np.pi * phase).astype(np.complex64)


def encode_raw_phy(
    payload: bytes | bytearray | Sequence[int],
    fs,
    *,
    SF=12,
    BW=125_000,
    cr=4,
    enable_crc=1,
    implicit_header=0,
    preamble_bits=8,
    sync_word=0x12,
    ldro=True,
    crc_mode="grlora",
    leading_silence_samples=10_000,
    trailing_silence_samples=0,
    cfo_hz=0.0,
):
    """生成不添加 RF-SR 私有 4-byte 头的理想 raw PHY IQ。

    ``payload`` 已经是完整 PHY payload。函数只添加 LoRa explicit header、
    可选 PHY CRC、前导码、sync word 和 SFD。``cfo_hz`` 只作用于包内波形；
    AWGN、信道响应和接收增益仍由上层决定。旧 :func:`encode` 的私有头语义
    保持不变。
    """

    if bool(implicit_header):
        raise ValueError("encode_raw_phy currently supports explicit header only.")
    if int(fs) <= 0 or int(BW) <= 0 or int(fs) % int(BW):
        raise ValueError(
            f"fs must be a positive integer multiple of BW, got {fs}/{BW}."
        )
    if not 0 <= int(sync_word) <= 0xFF:
        raise ValueError(f"sync_word must fit uint8, got {sync_word}.")
    if int(preamble_bits) < 5:
        raise ValueError("preamble_bits must be at least 5.")
    if int(leading_silence_samples) < 0:
        raise ValueError("leading_silence_samples must be non-negative.")
    if int(trailing_silence_samples) < 0:
        raise ValueError("trailing_silence_samples must be non-negative.")

    payload_bytes = bytes(int(value) & 0xFF for value in payload)
    header_symbols, payload_symbols = encode_raw_phy_symbols(
        payload_bytes,
        sf=int(SF),
        cr=int(cr),
        enable_crc=bool(enable_crc),
        ldro=bool(ldro),
        crc_mode=str(crc_mode),
    )
    os_factor = int(fs) // int(BW)
    samples_per_symbol = (1 << int(SF)) * os_factor
    upchirp = _raw_upchirp(int(SF), 0, os_factor)
    downchirp = np.conjugate(upchirp).astype(np.complex64)
    sync1 = ((int(sync_word) >> 4) & 0x0F) << 3
    sync2 = (int(sync_word) & 0x0F) << 3

    parts = [
        np.zeros(int(leading_silence_samples), dtype=np.dtype("<c8")),
        np.tile(upchirp, int(preamble_bits)),
        _raw_upchirp(int(SF), sync1, os_factor),
        _raw_upchirp(int(SF), sync2, os_factor),
        downchirp,
        downchirp,
        downchirp[: samples_per_symbol // 4],
    ]
    parts.extend(
        _raw_upchirp(int(SF), symbol, os_factor)
        for symbol in header_symbols + payload_symbols
    )
    if int(trailing_silence_samples):
        parts.append(
            np.zeros(
                int(trailing_silence_samples),
                dtype=np.dtype("<c8"),
            )
        )
    samples = np.concatenate(parts).astype(np.dtype("<c8"), copy=False)
    cfo_hz = float(cfo_hz)
    if cfo_hz != 0.0:
        # 相位从 packet 起点而非前置静默开始累计，和上游 complex_lora_packet
        # 先调制 packet、再写入固定 10,000 点静默缓冲区的顺序一致。
        packet_time = (
            np.arange(samples.size, dtype=np.float64)
            - int(leading_silence_samples)
        ) / float(fs)
        cfo_rotation = np.exp(
            2j * np.pi * cfo_hz * packet_time
        ).astype(np.complex64)
        samples = np.asarray(samples * cfo_rotation, dtype=np.dtype("<c8"))
    return RawPhyEncoding(
        payload=payload_bytes,
        cfo_hz=cfo_hz,
        samples=samples,
        header_symbol_ids=tuple(int(value) for value in header_symbols),
        payload_symbol_ids=tuple(int(value) for value in payload_symbols),
    )


def apply_hi2lora_sto(
    samples,
    *,
    fs,
    BW,
    SF,
    output_decimation,
    preamble_bits,
    leading_silence_samples=0,
    trailing_silence_samples=0,
    initial_sto_chips=0.0,
    sto_slope_chips_per_symbol=0.0,
):
    """按 Hi²LoRa 的符号级模型给低采样率输入加入 STO。

    ``initial_sto_chips`` 和 ``sto_slope_chips_per_symbol`` 都以 chip 为
    单位。这里直接实现产生式 ``i -> i - tau_s``，其中
    ``tau_s = tau_0 + s * slope``。对应式 (2e) 时，
    ``zeta = -slope / (2**SF - 1)``。

    分数采样直接作用于完整 PHY 波形，因此式 (2b) 的符号相关相位、
    式 (2d) 的 ``f_s ~= -tau_s``，以及式 (2c)/(2f) 描述的 chirp
    回绕相位差都会自然出现，无需再次乘相位补丁。
    """

    source = np.asarray(samples, dtype=np.dtype("<c8"))
    sample_rate_hz = int(fs)
    bandwidth_hz = int(BW)
    sf = int(SF)
    decimation = int(output_decimation)
    preamble_symbols = int(preamble_bits)
    leading_samples = int(leading_silence_samples)
    trailing_samples = int(trailing_silence_samples)
    initial_sto = float(initial_sto_chips)
    sto_slope = float(sto_slope_chips_per_symbol)

    if source.ndim != 1:
        raise ValueError("samples must be a one-dimensional complex array.")
    if sample_rate_hz <= 0 or bandwidth_hz <= 0:
        raise ValueError("fs and BW must be positive.")
    if sample_rate_hz % bandwidth_hz:
        raise ValueError("fs must be an integer multiple of BW.")
    if decimation < 1:
        raise ValueError("output_decimation must be positive.")
    if leading_samples < 0 or trailing_samples < 0:
        raise ValueError("silence sample counts must be non-negative.")
    if leading_samples + trailing_samples > source.size:
        raise ValueError("silence sample counts exceed the waveform length.")

    high_samples_per_chip = sample_rate_hz // bandwidth_hz
    high_samples_per_symbol = (1 << sf) * high_samples_per_chip
    if high_samples_per_symbol % decimation:
        raise ValueError(
            "output_decimation must preserve an integer number of samples "
            "per LoRa symbol."
        )

    output_source_indices = np.arange(
        0,
        source.size,
        decimation,
        dtype=np.int64,
    )
    output = np.asarray(source[output_source_indices], dtype=np.complex64)
    if initial_sto == 0.0 and sto_slope == 0.0:
        return output

    packet_end = source.size - trailing_samples
    packet_mask = (
        (output_source_indices >= leading_samples)
        & (output_source_indices < packet_end)
    )
    packet_indices = output_source_indices[packet_mask]
    packet_offsets = packet_indices - leading_samples

    # Preamble 后依次是两个 sync upchirp、两个完整 downchirp 和
    # 1/4 downchirp。data symbol 因此从 (preamble + 4.25)T_s 开始。
    data_start = (
        (preamble_symbols + 4) * high_samples_per_symbol
        + high_samples_per_symbol // 4
    )
    symbol_positions = np.floor_divide(
        packet_offsets,
        high_samples_per_symbol,
    ).astype(np.float64)
    data_mask = packet_offsets >= data_start
    symbol_positions[data_mask] = (
        preamble_symbols
        + 4.25
        + np.floor_divide(
            packet_offsets[data_mask] - data_start,
            high_samples_per_symbol,
        )
    )

    tau_chips = initial_sto + sto_slope * symbol_positions
    query_indices = (
        packet_indices.astype(np.float64)
        - tau_chips * high_samples_per_chip
    )
    left_indices = np.floor(query_indices).astype(np.int64)
    fractions = query_indices - left_indices
    valid = (query_indices >= 0.0) & (query_indices <= source.size - 1)
    left_clipped = np.clip(left_indices, 0, source.size - 1)
    right_clipped = np.clip(left_indices + 1, 0, source.size - 1)
    interpolated = (
        source[left_clipped] * (1.0 - fractions)
        + source[right_clipped] * fractions
    )
    interpolated[~valid] = 0.0
    output[packet_mask] = np.asarray(interpolated, dtype=np.complex64)
    return np.asarray(output, dtype=np.dtype("<c8"))


def encode_random_raw_phy(
    payload_length,
    fs,
    *,
    seed=None,
    rng=None,
    **phy_kwargs,
):
    """随机生成指定长度的 raw PHY payload，并编码成理想 IQ。

    ``seed`` 适合独立、可复现地生成一条样本；dataset 应传入长期持有的
    ``numpy.random.Generator``，从而让每次 ``__getitem__`` 得到新 payload。
    """

    length = int(payload_length)
    if not 1 <= length <= 255:
        raise ValueError(
            f"payload_length must be in [1, 255], got {payload_length}."
        )
    if seed is not None and rng is not None:
        raise ValueError("pass either seed or rng, not both.")
    generator = np.random.default_rng(seed) if rng is None else rng
    if not hasattr(generator, "integers"):
        raise TypeError("rng must provide numpy Generator.integers().")
    payload = generator.integers(
        0,
        256,
        size=length,
        dtype=np.uint8,
    ).tobytes()
    return encode_raw_phy(payload, fs, **phy_kwargs)


def lora_packet(BW, OSF, SF, k1, k2, n_pr, IH, CR, MAC_CRC, SRC, DST, SEQNO, MESSAGE, Trise, t0_frac, phi0):
    # in our implementation, the payload includes a 4 byte pseudo-header: (SRC, DST, SEQNO, LENGTH)
    LENGTH = np.uint16(4 + (MESSAGE.size))

    # n_sym_hdr: number of chirps encoding the header (0 if IH: True)
    # n_bits_hdr: [bit DI PAYLOAD presenti nell'header CHIEDERE A STEFANO]
    (n_sym_hdr, n_bits_hdr) = lora_header_init(SF, IH)
    [PAYLOAD, n_sym_payload] = lora_payload_init(SF, LENGTH, MAC_CRC, CR, n_bits_hdr, DST, SRC, SEQNO, MESSAGE)

    # -------------------------------------------------BIT TO SYMBOL MAPPING
    payload_ofs = 0
    if IH:
        k_hdr = []
    else:
        [k_hdr, payload_ofs] = lora_header(SF, LENGTH, CR, MAC_CRC, PAYLOAD, payload_ofs)

    k_payload = lora_payload(SF, CR, n_sym_payload, PAYLOAD, payload_ofs)

    # --------------------------------- CSS MODULATION

    # number of samples per chirp
    K = np.power(2, SF)
    N = int(K * OSF)
    # number of samples in the rising/falling edge
    Nrise = int(np.ceil(Trise * BW * OSF))

    # preamble modulation
    (p, phi) = lora_preamble(n_pr, k1, k2, BW, K, OSF, t0_frac, phi0)

    # samples initialization
    s = np.concatenate((np.zeros(Nrise), p, np.zeros(N * (n_sym_hdr + n_sym_payload) + Nrise)))

    # rising edge samples
    s[0:Nrise] = p[N - 1 + np.arange(-Nrise + 1, 1, dtype=np.int16)] * np.power(
        np.sin(np.pi / 2 * np.arange(1, Nrise + 1) / Nrise), 2)
    s_ofs = p.size + Nrise

    # header modulation, if any
    for sym in range(0, n_sym_hdr):
        k = k_hdr[sym]
        (s[s_ofs + np.arange(0, N)], phi) = lora_chirp(+1, k, BW, K, OSF, t0_frac, phi)
        s_ofs = s_ofs + N

    # payload modulation
    for sym in range(0, n_sym_payload):
        k = k_payload[sym]
        (s[s_ofs + np.arange(0, N)], phi) = lora_chirp(+1, k, BW, K, OSF, t0_frac, phi)
        s_ofs = s_ofs + N

    # falling edge samples
    s[s_ofs + np.arange(0, Nrise)] = s[s_ofs + np.arange(0, Nrise)] * np.power(
        np.cos(np.pi / 2 * np.arange(0, Nrise) / Nrise), 2)

    return s, k_hdr, k_payload


def lora_header_init(SF, IH):
    if (IH):
        n_sym_hdr = np.uint16(0)
        n_bits_hdr = np.uint16(0)
    else:
        # interleaving block size, respectively for header & payload
        CR_hdr = np.uint16(4)
        DE_hdr = np.uint16(1)
        n_sym_hdr = 4 + CR_hdr
        intlv_hdr_size = (SF - 2 * DE_hdr) * (4 + CR_hdr)
        n_bits_hdr = np.uint16(intlv_hdr_size * 4 / (4 + CR_hdr) - 20)

    return n_sym_hdr, n_bits_hdr


# function [k_hdr,payload_ofs] =
def lora_header(SF, LENGTH, CR, MAC_CRC, PAYLOAD, payload_ofs):
    # header parity check matrix (reverse engineered through brute force search)

    header_FCS = np.array(([1, 1, 0, 0, 0], [1, 0, 1, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 0, 1], [0, 1, 1, 0, 0],
                           [0, 1, 0, 1, 0], [0, 1, 0, 0, 1], [0, 0, 1, 1, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 1],
                           [0, 0, 1, 1, 1],
                           [0, 1, 0, 1, 1]))

    CR_hdr = np.uint8(4)
    Hamming_hdr = np.array(([1, 0, 1, 1], [1, 1, 1, 0], [1, 1, 0, 1], [0, 1, 1, 1]), dtype=np.uint8)
    n_sym_hdr = 4 + CR_hdr
    DE_hdr = np.uint8(1)
    PPM = SF - 2 * DE_hdr
    gray_hdr = gray_lut(PPM)[0]
    intlv_hdr_size = PPM * n_sym_hdr

    # header (20 bit)
    LENGTH_bits = num2binary(LENGTH, 8)
    CR_bits = num2binary(CR, 3)
    hdr = np.concatenate((LENGTH_bits[np.arange(3, -1, -1, dtype=np.uint8)],
                          LENGTH_bits[np.arange(7, 3, -1, dtype=np.uint8)], np.array([MAC_CRC], dtype=np.uint8),
                          CR_bits[np.arange(2, -1, -1, dtype=np.uint8)], np.zeros(8, dtype=np.uint8)))
    hdr_chk_indexes = np.concatenate((np.arange(3, -1, -1, dtype=np.uint8), np.arange(7, 3, -1, dtype=np.uint8),
                                      np.arange(11, 7, -1, dtype=np.uint8)))

    hdr_chk = np.mod(hdr[hdr_chk_indexes] @ header_FCS, 2)
    hdr[12] = hdr_chk[0]
    hdr[16:20] = hdr_chk[4:-0:-1]

    # parity bit calculation
    C = np.zeros((PPM, 4 + CR_hdr))
    for k in range(0, 5):
        C[k, 0:4] = hdr[k * 4 + np.arange(0, 4)]
        C[k, 3 + np.arange(1, CR_hdr + 1)] = np.mod(C[k, 0:4] @ Hamming_hdr, 2)

    for k in range(5, PPM):
        C[k, 0:4] = PAYLOAD[payload_ofs + np.arange(0, 4, dtype=np.uint8)]
        payload_ofs = payload_ofs + 4
        C[k, 3 + np.arange(1, CR_hdr + 1)] = np.mod(C[k, 0:4] @ Hamming_hdr, 2)

    # rows flip
    C = np.flip(C, 0)

    S = np.zeros((4 + CR_hdr, PPM), dtype=np.uint8)
    for ii in range(0, PPM):
        for jj in range(0, 4 + CR_hdr):
            S[jj, np.mod(ii + jj, PPM)] = C[ii, jj]

    bits_hdr = np.reshape(S.transpose(), intlv_hdr_size, order='F')

    # bit to symbol mapping
    k_hdr = np.zeros(n_sym_hdr)
    K = np.power(2, SF)
    for sym in range(0, n_sym_hdr):
        k_hdr[sym] = K - 1 - np.power(2, (2 * DE_hdr)) * gray_hdr[
            bits_hdr[sym * PPM + np.arange(0, PPM, dtype=np.uint16)] @ np.power(2, np.arange(PPM - 1, -1, -1))]

    return k_hdr, payload_ofs


def lora_payload_init(SF, LENGTH, MAC_CRC, CR, n_bits_hdr, DST, SRC, SEQNO, MESSAGE):
    # bigger spreading factors (11 and 12) use 2 less bits per symbol
    if SF > 10:
        DE = np.uint8(1)
    else:
        DE = np.uint8(0)
    PPM = SF - 2 * DE
    n_bits_blk = PPM * 4
    n_bits_tot = 8 * LENGTH + 16 * MAC_CRC
    n_blk_tot = int(np.ceil((n_bits_tot - n_bits_hdr) / n_bits_blk))
    n_sym_blk = 4 + CR
    n_sym_payload = n_blk_tot * n_sym_blk

    byte_range = np.arange(7, -1, -1, dtype=np.uint8)
    PAYLOAD = np.zeros(int(n_bits_hdr + n_blk_tot * n_bits_blk), dtype=np.uint8)
    PAYLOAD[byte_range] = num2binary(DST, 8)
    PAYLOAD[8 + byte_range] = num2binary(SRC, 8)
    PAYLOAD[8 * 2 + byte_range] = num2binary(SEQNO, 8)
    PAYLOAD[8 * 3 + byte_range] = num2binary(LENGTH, 8)
    for k in range(0, MESSAGE.size):
        PAYLOAD[8 * (4 + k) + byte_range] = num2binary(MESSAGE[k], 8)

    if MAC_CRC:
        PAYLOAD[8 * LENGTH + np.arange(0, 16, dtype=np.uint8)] = CRC16(PAYLOAD[0:8 * LENGTH])

    # ----------------------------------------------------------- WHITENING
    W = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.uint8)
    W_fb = np.array([0, 0, 0, 1, 1, 1, 0, 1], dtype=np.uint8)
    for k in range(1, int(np.floor(len(PAYLOAD) / 8) + 1)):
        PAYLOAD[(k - 1) * 8 + np.arange(0, 8, dtype=np.int32)] = np.mod(
            PAYLOAD[(k - 1) * 8 + np.arange(0, 8, dtype=np.int32)] + W, 2)
        W1 = np.array([np.mod(np.sum(W * W_fb), 2)])
        W = np.concatenate((W1, W[0:-1]))

    return PAYLOAD, n_sym_payload


# CRC-16 calculation for LoRa (reverse engineered from a Libelium board)
def CRC16(bits):
    length = int(len(bits) / 8)
    # initial states for crc16 calculation, valid for lenghts in the range 5,255
    state_vec = np.array([46885, 27367, 35014, 54790, 18706, 15954, \
                          9784, 59350, 12042, 22321, 46211, 20984, 56450, 7998, 62433, 35799, \
                          2946, 47628, 30930, 52144, 59061, 10600, 56648, 10316, 34962, 55618, \
                          57666, 2088, 61160, 25930, 63354, 24012, 29658, 17909, 41022, 17072, \
                          42448, 5722, 10472, 56651, 40183, 19835, 21851, 13020, 35306, 42553, \
                          12394, 57960, 8434, 25101, 63814, 29049, 27264, 213, 13764, 11996, \
                          46026, 6259, 8758, 22513, 43163, 38423, 62727, 60460, 29548, 18211, \
                          6559, 61900, 55362, 46606, 19928, 6028, 35232, 29422, 28379, 55218, \
                          38956, 12132, 49339, 47243, 39300, 53336, 29575, 53957, 5941, 63650, \
                          9502, 28329, 44510, 28068, 19538, 19577, 36943, 59968, 41464, 33923, \
                          54504, 49962, 64357, 12382, 44678, 11234, 58436, 47434, 63636, 51152, \
                          29296, 61176, 33231, 32706, 27862, 11005, 41129, 38527, 32824, 20579, \
                          37742, 22493, 37464, 56698, 29428, 27269, 7035, 27911, 55897, 50485, \
                          10543, 38817, 54183, 52989, 24549, 33562, 8963, 38328, 13330, 24139, \
                          5996, 8270, 49703, 60444, 8277, 43598, 1693, 60789, 32523, 36522, \
                          17339, 33912, 23978, 55777, 34725, 2990, 13722, 60616, 61229, 19060, \
                          58889, 43920, 9043, 10131, 26896, 8918, 64347, 42307, 42863, 7853, \
                          4844, 60762, 21736, 62423, 53096, 19242, 55756, 26615, 53246, 11257, \
                          2844, 47011, 10022, 13541, 18296, 44005, 23544, 18733, 23770, 33147, \
                          5237, 45754, 4432, 22560, 40752, 50620, 32260, 2407, 26470, 2423, \
                          33831, 34260, 1057, 552, 56487, 62909, 4753, 7924, 40021, 7849, \
                          4895, 10401, 32039, 40207, 63952, 10156, 53647, 51938, 16861, 46769, \
                          7703, 9288, 33345, 16184, 56808, 30265, 10696, 4218, 7708, 32139, \
                          34174, 32428, 20665, 3869, 43003, 6609, 60431, 22531, 11704, 63584, \
                          13620, 14292, 37000, 8503, 38414, 38738, 10517, 48783, 30506, 63444, \
                          50520, 34666, 341, 34793, 2623], dtype=np.uint16)
    crc_tmp = num2binary(state_vec[length - 5], 16)
    # crc_poly = [1 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 1]
    # 	for j = 1:numel(bits)/8
    # 		for k = 1:8
    # 			add = crc_tmp(1)
    # 			crc_tmp = [crc_tmp(2: ),bits((j-1)*8+9-k)]
    # 			if add
    # 				crc_tmp = mod(crc_tmp+crc_poly(2: ),2)
    #
    #
    #
    # 	CRC=crc_tmp(16:-1:1)
    pos = 0
    pos4 = 4
    pos11 = 11
    for j in range(0, length):
        for k in range(0, 8):
            add = crc_tmp[pos]
            crc_tmp[pos] = bits[j * 8 + 7 - k]
            if add:
                crc_tmp[pos4] = 1 - crc_tmp[pos4]
                crc_tmp[pos11] = 1 - crc_tmp[pos11]
                crc_tmp[pos] = 1 - crc_tmp[pos]

            pos = np.mod(pos + 1, 16)
            pos4 = np.mod(pos4 + 1, 16)
            pos11 = np.mod(pos11 + 1, 16)

    CRC = crc_tmp[np.mod(pos + np.arange(15, -1, -1, dtype=np.int32), 16)]
    return CRC


# gray and reversed-gray mappings

def gray_lut(n):
    pow_n = np.power(2, n)
    g = np.zeros(pow_n, dtype=np.uint16)

    vec = np.atleast_2d(np.arange(0, pow_n, dtype=np.uint16)).transpose()
    vec = np.flip((vec.view(np.uint8)), 1)
    vec = np.unpackbits(vec).reshape(pow_n, 16)
    vec = vec[:, -n:]
    vec = vec.transpose()
    support_x = np.zeros((n, pow_n))
    support_x[1:, :] = vec[:-1, :]

    ig = np.matmul(np.power(2, np.arange(n - 1, -1, -1)), np.mod(vec + support_x, 2))
    ig = ig.astype(int)
    g[ig] = np.arange(0, pow_n, dtype=np.uint16)
    return g, ig


# function k_payload = \
def lora_payload(SF, CR, n_sym_payload, PAYLOAD, payload_ofs):
    # varargout = {DST, SRC, SEQNO, MESSAGE}

    # hamming parity check matrices
    Hamming_P1 = np.array(([1], [1], [1], [1]), dtype=np.uint8)
    Hamming_P2 = np.array(([1, 0], [1, 1], [1, 1], [0, 1]), dtype=np.uint8)
    Hamming_P3 = np.array(([1, 0, 1], [1, 1, 1], [1, 1, 0], [0, 1, 1]), dtype=np.uint8)
    Hamming_P4 = np.array(([1, 0, 1, 1], [1, 1, 1, 0], [1, 1, 0, 1], [0, 1, 1, 1]), dtype=np.uint8)

    if CR == 1:
        Hamming = Hamming_P1
    elif CR == 2:
        Hamming = Hamming_P2
    elif CR == 3:
        Hamming = Hamming_P3
    elif CR == 4:
        Hamming = Hamming_P4

    if SF > 10:
        DE = 1
    else:
        DE = 0

    PPM = SF - 2 * DE
    n_sym_blk = (4 + CR)
    intlv_blk_size = PPM * n_sym_blk
    gray = gray_lut(PPM)[0]
    K = np.power(2, SF)
    n_blk_tot = int(n_sym_payload / n_sym_blk)
    C = np.zeros((PPM, 4 + CR))
    S = np.zeros((4 + CR, PPM))
    k_payload = np.zeros(n_sym_payload)
    for blk in range(0, n_blk_tot):
        for k in range(0, PPM):
            C[k, 0:4] = PAYLOAD[payload_ofs + np.arange(0, 4, dtype=np.uint8)]
            payload_ofs = payload_ofs + 4
            C[k, 4: 4 + CR] = np.mod(C[k, 0:4] @ Hamming, 2)

        # row flip
        C = np.flip(C, 0)

        # interleaving
        for ii in range(0, PPM):
            for jj in range(0, 4 + CR):
                S[jj, np.mod(ii + jj, PPM)] = C[ii, jj]

        bits_blk = np.reshape(S.transpose(), intlv_blk_size, order='F')

        # bit to symbol mapping
        for sym in range(0, n_sym_blk):
            k_payload[(blk) * n_sym_blk + sym] = K - 1 - np.power(2, (2 * DE)) * gray[int(
                bits_blk[(sym * PPM + np.arange(0, PPM, dtype=np.uint16))] @ np.power(2, np.arange(PPM - 1, -1, -1)))]

    return k_payload


def chirp(f_ini, N, Ts, Df, t0_frac=0, phi_ini=0):
    t = (t0_frac + np.arange(0, N, dtype=np.int32)) * Ts
    T = N * Ts
    s = np.exp(1j * (phi_ini + 2 * np.pi * f_ini * t + np.pi * Df * np.power(t, 2)))
    phi_fin = phi_ini + 2 * np.pi * f_ini * T + np.pi * Df * np.power(T, 2)
    return s, phi_fin


def lora_packet_rx(s, SF, BW, OSF, Trise, p_ofs_est, Cfo_est):
    truncated = False
    fs = BW * OSF
    Ts = 1 / fs
    N = np.power(2, SF) * OSF
    Nrise = np.ceil(Trise * fs)
    d0 = chirp(BW / 2, N, Ts, -BW / (N * Ts))[0]
    DST = -1
    SRC = -1
    SEQNO = -1
    CR = -1
    HAS_CRC = -1
    # demodula i simboli dell'header
    # n_sym_hdr == 8 o 0 in caso di header implicito
    # numero di campioni del preambolo
    ofs = int(Nrise + 12.25 * N)

    CR_hdr = 4
    n_sym_hdr = 4 + CR_hdr
    k_hdr_est = np.zeros((n_sym_hdr))
    _internal["header_ifft"] = []  # [Andreas] add data
    for i in range(0, n_sym_hdr):

        try:
            temp = np.exp(-1j * 2 * np.pi * Cfo_est * Ts * (ofs + np.arange(0, N))) * s[
                p_ofs_est + ofs + np.arange(0, N, dtype=np.int32)] * d0
            temp_nodown = np.exp(-1j * 2 * np.pi * Cfo_est * Ts * (ofs + np.arange(0, N))) * s[
                p_ofs_est + ofs + np.arange(0, N, dtype=np.int32)]  # * d0

        except IndexError:
            k_hdr_est = None
            MAC_CRC_OK = False
            HDR_FCS_OK = False
            k_payload_est = None
            MSG = None
            truncated = True
            return k_hdr_est, HDR_FCS_OK, k_payload_est, MAC_CRC_OK, MSG, DST, SRC, SEQNO, CR, HAS_CRC, truncated, ofs

        ofs = ofs + N

        if _nelora:
            # if 'nelora_model' not in locals() or nelora_model is None:
            #     from rfsr.GLoRiPHY.Baseline import load_model_nelora, predict_nelora
            #     nelora_model, neloar_opts = load_model_nelora(fs=1e6) # only supports 1e6
            #
            from rfsr.GLoRiPHY.Baseline import predict_nelora
            nelora_model, neloar_opts = load_model_nelora(1e6)
            symbol = torch.tensor(temp_nodown).unsqueeze(0).to(torch.complex64).to(neloar_opts.device)
            x = predict_nelora(symbol, nelora_model, neloar_opts)

            import numpy
            ifft_result = np.abs(np.fft.ifft(temp[0:-1:OSF]))
            old_method_pos = numpy.argmax(ifft_result) - 1
            _internal["header_ifft"].append(ifft_result)  # [Andreas] add data

            pos = (2 ** SF) - x[0] - 1
            k_hdr_est[i] = pos

            print(f"NeLoRa: {k_hdr_est[i]}, expected: {old_method_pos}, Cfo_est={Cfo_est}, p_ofs_est={p_ofs_est}")

            print()

        if _gloriphy:
            from rfsr.GLoRiPHY.GLoRiPHY_source import predict_gloriphy, load_model_gloriphy
            gloriphy_model, gloriphy_opts = load_model_gloriphy(fs=1e6)
            symbol = torch.tensor(temp_nodown).unsqueeze(0).to(torch.complex64).to(gloriphy_opts.device)
            x = predict_gloriphy(symbol, gloriphy_model, gloriphy_opts)

            import numpy
            ifft_result = np.abs(np.fft.ifft(temp[0:-1:OSF]))
            old_method_pos = numpy.argmax(ifft_result) - 1
            _internal["header_ifft"].append(ifft_result)  # [Andreas] add data

            pos = (2 ** SF) - x[0] - 1
            k_hdr_est[i] = pos

            print(f"GLoRiPHY: {k_hdr_est[i]}, expected: {old_method_pos}, Cfo_est={Cfo_est}, p_ofs_est={p_ofs_est}")

            print()

        if _loratrimmer:
            from rfsr.LoRaTrimmer import decode_loratrimmer
            # temp = np.exp(-1j * 2 * np.pi * Cfo_est * Ts * (ofs + np.arange(0, N))) * s[p_ofs_est + ofs + np.arange(0, N, dtype=np.int32)] #* d0
            # temp = s[p_ofs_est + ofs + np.arange(0, N, dtype=np.int32)]
            import rfsr
            rfsr.LoRaTrimmer.main.fs = fs  # set sampling rate
            pos = (2 ** SF) - decode_loratrimmer(temp_nodown, SF)

            import numpy
            # ifft_result = numpy.abs(numpy.fft.ifft((temp_nodown * d0)[0:-1:OSF]))
            # test = 1024 - numpy.argmax(ifft_result)
            # test2 = numpy.argmax(ifft_result)

            print()

            ifft_result = np.abs(np.fft.ifft(temp[0:-1:OSF]))
            old_method_pos = numpy.argmax(ifft_result) - 1

            _internal["header_ifft"].append(ifft_result)  # [Andreas] add data
            k_hdr_est[i] = pos  # - 1

            print(f"LoRaTrimmer: {k_hdr_est[i]}, expected: {old_method_pos}, Cfo_est={Cfo_est}, p_ofs_est={p_ofs_est}")

            print()
        else:
            if _windowing and _zeropadding:
                print("Windowing & Zero-padding used")
                # Get the data slice to be transformed
                data_slice = temp[0:-1:OSF]
                data_length = len(data_slice)

                # Generate and apply the Hanning window
                # The window function reduces spectral leakage and should be applied in the frequency domain.
                window = np.hanning(data_length)
                windowed_data = data_slice * window

                # Perform IFFT on the windowed data once
                ifft_result = np.abs(np.fft.ifft(windowed_data))

                # --- Configuration for Zero-Padding ---
                # Pad the array to 4 times its original length for high resolution
                PADDING_FACTOR = 4
                # -------------------------------------

                # Get the data slice to be transformed
                data_slice = windowed_data
                data_length = len(data_slice)

                # --- ROBUST ZERO-PADDING FIX START ---
                # Calculate the final length of the IFFT result
                N_padded = data_length * PADDING_FACTOR

                # np.fft.ifft(a, n=N) automatically zero-pads 'a' up to length 'N'.
                # This is the cleanest way to interpolate the spatial/time domain result.
                ifft_result = np.abs(np.fft.ifft(data_slice, n=N_padded))
                # --- ROBUST ZERO-PADDING FIX END ---

            elif _windowing:

                print("Windowing used")
                # Get the data slice to be transformed
                data_slice = temp[0:-1:OSF]
                data_length = len(data_slice)

                # Generate and apply the Hanning window
                # The window function reduces spectral leakage and should be applied in the frequency domain.
                window = np.hanning(data_length)
                windowed_data = data_slice * window

                # Perform IFFT on the windowed data once
                ifft_result = np.abs(np.fft.ifft(windowed_data))
            elif _zeropadding:
                # --- Configuration for Zero-Padding ---
                # Pad the array to 4 times its original length for high resolution
                PADDING_FACTOR = 4
                # -------------------------------------

                print("Zero-Padding used")

                # Get the data slice to be transformed
                data_slice = temp[0:-1:OSF]
                data_length = len(data_slice)

                # --- ROBUST ZERO-PADDING FIX START ---
                # Calculate the final length of the IFFT result
                N_padded = data_length * PADDING_FACTOR

                # np.fft.ifft(a, n=N) automatically zero-pads 'a' up to length 'N'.
                # This is the cleanest way to interpolate the spatial/time domain result.
                ifft_result = np.abs(np.fft.ifft(data_slice, n=N_padded))
                # --- ROBUST ZERO-PADDING FIX END ---

            else:
                ifft_result = np.abs(np.fft.ifft(temp[0:-1:OSF]))

            pos = np.argmax(ifft_result)

            if _zeropadding:
                pos = int(round(np.argmax(ifft_result) / PADDING_FACTOR))
            # print(f"Pos: {pos}")

            _internal["header_ifft"].append(ifft_result)  # [Andreas] add data
            k_hdr_est[i] = pos - 1

        # ofs = ofs + N
        # pos = np.argmax(np.abs(np.fft.ifft(temp[0:-1:OSF])))
        # _internal["header_ifft"].append(np.abs(np.fft.ifft(temp[0:-1:OSF])))  # [Andreas] add data
        # k_hdr_est[i] = pos - 1

        # [Andreas] start
        if PLOTTING:
            import matplotlib.pyplot as plt
            plt.figure()

            # Compute the two IFFT magnitudes
            y1 = np.abs(np.fft.ifft(temp[0:-1:OSF]))
            y2 = np.abs(np.fft.ifft(temp[0:-1]))

            # Create a common x-axis length (normalized from 0 to 1)
            x1 = np.linspace(0, 1, len(y1))
            x2 = np.linspace(0, 1, len(y2))

            # Plot both
            plt.plot(x1, y1, label='Downsampled (OSF)', alpha=0.8)
            # plt.plot(x2, y2, label='Full', alpha=0.8)

            # plt.show()
            os.makedirs("results/", exist_ok=True)
            plt.savefig(f"results/h_{_filename}_{i}.svg")
            plt.close()
        # [Andreas] end

    _internal["k_hdr_est"] = k_hdr_est.astype(np.int16)  # [Andreas] store data

    # in case of header checksum failure, we assume the message to be lost/corrupted [ToDo: implement implicit header mode]
    print(f"k_hr_est: {k_hdr_est}")
    (HDR_FCS_OK, LENGTH, HAS_CRC, CR, PAYLOAD_bits_hdr) = lora_header_decode(SF, k_hdr_est)
    if not HDR_FCS_OK:
        k_hdr_est = None
        MAC_CRC_OK = False
        k_payload_est = None
        MSG = None

        return k_hdr_est, HDR_FCS_OK, k_payload_est, MAC_CRC_OK, MSG, DST, SRC, SEQNO, CR, HAS_CRC, truncated, ofs
    else:
        n_bits_hdr = (PAYLOAD_bits_hdr.size)
        n_sym_payload = lora_payload_n_sym(SF, LENGTH, HAS_CRC, CR, n_bits_hdr)

        k_payload_est = np.zeros((n_sym_payload))
        _internal["payload_ifft"] = []  # [Andreas] add data
        for i in range(0, n_sym_payload):
            try:
                temp = np.exp(-1j * 2 * np.pi * Cfo_est * Ts * (ofs + np.arange(0, N, dtype=np.int32))) * s[
                    p_ofs_est + ofs + np.arange(0, N, dtype=np.int32)] * d0
                temp_nodown = np.exp(-1j * 2 * np.pi * Cfo_est * Ts * (ofs + np.arange(0, N, dtype=np.int32))) * s[
                    p_ofs_est + ofs + np.arange(0, N, dtype=np.int32)]  # * d0
            except:
                k_hdr_est = None
                MAC_CRC_OK = False
                k_payload_est = None
                MSG = None

                truncated = True
                return k_hdr_est, HDR_FCS_OK, k_payload_est, MAC_CRC_OK, MSG, DST, SRC, SEQNO, CR, HAS_CRC, truncated, ofs

            ofs = ofs + N

            if _nelora:
                symbol = torch.tensor(temp_nodown).unsqueeze(0).to(torch.complex64).to(neloar_opts.device)
                x = predict_nelora(symbol, nelora_model, neloar_opts)

                import numpy
                ifft_result = np.abs(np.fft.ifft(temp[0:-1:OSF]))
                old_method_pos = numpy.argmax(ifft_result) - 1
                _internal["payload_ifft"].append(ifft_result)  # [Andreas] add data

                pos = (2 ** SF) - x[0] - 1
                k_payload_est[i] = pos

                print(
                    f"NeLoRa: {k_payload_est[i]}, expected: {old_method_pos}, Cfo_est={Cfo_est}, p_ofs_est={p_ofs_est}")

                print()

            if _gloriphy:
                symbol = torch.tensor(temp_nodown).unsqueeze(0).to(torch.complex64).to(gloriphy_opts.device)
                x = predict_gloriphy(symbol, gloriphy_model, gloriphy_opts)

                import numpy
                ifft_result = np.abs(np.fft.ifft(temp[0:-1:OSF]))
                old_method_pos = numpy.argmax(ifft_result) - 1
                _internal["payload_ifft"].append(ifft_result)  # [Andreas] add data

                pos = (2 ** SF) - x[0] - 1
                k_payload_est[i] = pos

                print(
                    f"GLoRiPHY: {k_payload_est[i]}, expected: {old_method_pos}, Cfo_est={Cfo_est}, p_ofs_est={p_ofs_est}")

                print()

            if _loratrimmer:
                from rfsr.LoRaTrimmer import decode_loratrimmer
                # temp = np.exp(-1j * 2 * np.pi * Cfo_est * Ts * (ofs + np.arange(0, N))) * s[p_ofs_est + ofs + np.arange(0, N, dtype=np.int32)] #* d0
                # temp = s[p_ofs_est + ofs + np.arange(0, N, dtype=np.int32)]
                rfsr.LoRaTrimmer.main.fs = fs  # set sampling rate
                pos = (2 ** SF) - decode_loratrimmer(temp_nodown, SF)

                import numpy
                # ifft_result = numpy.abs(numpy.fft.ifft((temp_nodown * d0)[0:-1:OSF]))
                # test = 1024 - numpy.argmax(ifft_result)
                # test2 = numpy.argmax(ifft_result)

                print()

                ifft_result = np.abs(np.fft.ifft(temp[0:-1:OSF]))
                old_method_pos = numpy.argmax(ifft_result) - 1

                _internal["payload_ifft"].append(ifft_result)  # [Andreas] add data
                k_payload_est[i] = pos  # - 1

                print(
                    f"LoRaTrimmer: {k_payload_est[i]}, expected: {old_method_pos}, Cfo_est={Cfo_est}, p_ofs_est={p_ofs_est}")

            else:

                if _windowing and _zeropadding:
                    print("Windowing & Zero-padding used")
                    # Get the data slice to be transformed
                    data_slice = temp[0:-1:OSF]
                    data_length = len(data_slice)

                    # Generate and apply the Hanning window
                    # The window function reduces spectral leakage and should be applied in the frequency domain.
                    window = np.hanning(data_length)
                    windowed_data = data_slice * window

                    # Perform IFFT on the windowed data once
                    ifft_result = np.abs(np.fft.ifft(windowed_data))

                    # --- Configuration for Zero-Padding ---
                    # Pad the array to 4 times its original length for high resolution
                    PADDING_FACTOR = 4
                    # -------------------------------------

                    # Get the data slice to be transformed
                    data_slice = windowed_data
                    data_length = len(data_slice)

                    # --- ROBUST ZERO-PADDING FIX START ---
                    # Calculate the final length of the IFFT result
                    N_padded = data_length * PADDING_FACTOR

                    # np.fft.ifft(a, n=N) automatically zero-pads 'a' up to length 'N'.
                    # This is the cleanest way to interpolate the spatial/time domain result.
                    ifft_result = np.abs(np.fft.ifft(data_slice, n=N_padded))
                    # --- ROBUST ZERO-PADDING FIX END ---
                elif _windowing:
                    # Get the data slice to be transformed
                    data_slice = temp[1:-1:OSF]
                    data_length = len(data_slice)

                    # Generate and apply the Hanning window
                    # The window function reduces spectral leakage and should be applied in the frequency domain.
                    window = np.hanning(data_length)
                    windowed_data = data_slice * window

                    # Perform IFFT on the windowed data once
                    ifft_result = np.abs(np.fft.ifft(windowed_data))
                elif _zeropadding:
                    # --- Configuration for Zero-Padding ---
                    # Pad the array to 4 times its original length for high resolution
                    PADDING_FACTOR = 4
                    # -------------------------------------

                    print("Zero-Padding used")

                    # Get the data slice to be transformed
                    data_slice = temp[1:-1:OSF]
                    data_length = len(data_slice)

                    # --- ROBUST ZERO-PADDING FIX START ---
                    # Calculate the final length of the IFFT result
                    N_padded = data_length * PADDING_FACTOR

                    # np.fft.ifft(a, n=N) automatically zero-pads 'a' up to length 'N'.
                    # This is the cleanest way to interpolate the spatial/time domain result.
                    ifft_result = np.abs(np.fft.ifft(data_slice, n=N_padded))
                    # --- ROBUST ZERO-PADDING FIX END ---
                else:
                    ifft_result = np.abs(np.fft.ifft(temp[1:-1:OSF]))

                pos = np.argmax(ifft_result)

                if _zeropadding:
                    pos = int(round(np.argmax(ifft_result) / PADDING_FACTOR))

                # print(f"Pos: {pos}")
                _internal["payload_ifft"].append(ifft_result)  # [Andreas] add data
                k_payload_est[i] = pos - 1

            # pos = np.argmax(np.abs(np.fft.ifft(temp[1:-1:OSF])))
            # _internal["payload_ifft"].append(np.abs(np.fft.ifft(temp[1:-1:OSF]))) # [Andreas] add data
            # k_payload_est[i] = pos - 1

            # pos = np.argmax(np.abs(np.fft.ifft(temp[1:-1:OSF], n=2*len(temp[1:-1:OSF])))) //2
            # _internal["payload_ifft"].append(np.abs(np.fft.ifft(temp[1:-1:OSF], n=2*len(temp[1:-1:OSF])))) # [Andreas] add data
            # if np.isnan(pos):
            #     pass
            # k_payload_est[i] = pos - 1

            # pos = np.argmax(np.abs(np.fft.ifft(temp[0:-1:OSF])))
            # _internal["payload_ifft"].append(np.abs(np.fft.ifft(temp[0:-1:OSF]))) # [Andreas] add data
            # k_payload_est[i] = pos - 1

            # [Andreas] start
            if PLOTTING:
                import matplotlib.pyplot as plt
                plt.figure()
                y1 = np.abs(np.fft.ifft(temp[0:-1:OSF]))
                # y1 = np.abs(np.fft.ifft(temp))
                x1 = np.linspace(0, 1, len(y1))
                plt.plot(x1, y1, label='Downsampled (OSF)', alpha=0.8)
                os.makedirs("results/", exist_ok=True)
                plt.savefig(f"results/p_{_filename}_{i}.svg")
                plt.close()
        # [Andreas] end

        _internal["k_payload_est"] = k_payload_est.astype(np.int16)  # [Andreas] store data

        print(f"k_payload_est: {k_payload_est.astype(np.int16)}")
        (MAC_CRC_OK, DST, SRC, SEQNO, MSG, HAS_CRC) = lora_payload_decode(SF, k_payload_est, PAYLOAD_bits_hdr, HAS_CRC,
                                                                          CR, LENGTH)
        # if MAC_CRC_OK:
        #
        # 	#fprintf('DST: #d SRC: #d SEQ: #d LEN: #d DATA: "',...
        # 		#DST,SRC,SEQNO,LENGTH)
        # 	for i in range(1,len(MSG)+1):
        # 		if MSG[i-1]>=32 and MSG[i-1]<=127:
        # 			#fprintf('#c',MSG(i))
        # 			pass
        # 		else:
        # 			pass
        # 			#fprintf('\\#d',MSG(i))
        return k_hdr_est, HDR_FCS_OK, k_payload_est, MAC_CRC_OK, MSG, DST, SRC, SEQNO, CR, HAS_CRC, truncated, ofs


def lora_header_decode(SF, k_hdr):
    if SF < 7:
        FCS_OK = False
        return

    DE_hdr = 1
    PPM = SF - 2 * DE_hdr
    (_, degray) = gray_lut(PPM)

    # np.fft(downsample(s_rx.*chirp)) demodulates the signal
    CR_hdr = 4
    n_sym_hdr = 4 + CR_hdr
    intlv_hdr_size = PPM * n_sym_hdr
    bits_est = np.zeros((intlv_hdr_size), dtype=np.uint8)
    K = np.power(2, SF)
    for sym in range(0, n_sym_hdr):
        # gray decode
        bin = int(np.round((K - 1 - k_hdr[sym]) / 4))
        bits_est[(sym) * PPM + np.arange(0, PPM, dtype=np.int32)] = num2binary(degray[np.mod(bin, np.power(2, PPM))],
                                                                               PPM)

    # interleaver del brevetto
    S = np.reshape(bits_est, (PPM, 4 + CR_hdr), order='F').transpose()
    C = np.zeros((PPM, 4 + CR_hdr), dtype=np.uint8)
    for ii in range(0, PPM):
        for jj in range(0, 4 + CR_hdr):
            C[ii, jj] = S[jj, np.mod(ii + jj, PPM)]

    # row flip
    C = np.flip(C, 0)

    # header parity check matrix (reverse engineered through brute force search)
    header_FCS = np.array(([1, 1, 0, 0, 0], [1, 0, 1, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 0, 1], [0, 1, 1, 0, 0], \
                           [0, 1, 0, 1, 0], [0, 1, 0, 0, 1], [0, 0, 1, 1, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 1],
                           [0, 0, 1, 1, 1], \
                           [0, 1, 0, 1, 1]))

    FCS_chk = np.mod(np.concatenate((C[0, np.arange(3, -1, -1, dtype=np.int32)], C[1, np.arange(3, -1, -1)],
                                     C[2, np.arange(3, -1, -1, dtype=np.int32)], np.array([C[3, 0]], dtype=np.uint8),
                                     C[4, np.arange(3, -1, -1, dtype=np.int32)])) @ [
                         np.concatenate((header_FCS, np.eye(5)), 0)], 2)
    FCS_OK = not np.any(FCS_chk)
    if not FCS_OK:
        LENGTH = -1
        HAS_CRC = False
        CR = -1
        PAYLOAD_bits = -1


    else:
        # header decoding
        # Header=char(['len:',C(1,4:-1:1)+'0',C(2,4:-1:1)+'0',...
        # 	' CR:',C(3,4:-1:2)+'0',...
        # 	' MAC-CRC:',C(3,1)+'0',...
        # 	' HDR-FCS:',C(4,4:-1:1)+'0',C(5,4:-1:1)+'0'])
        # fprintf('Header Decode: #s [OK]\n',Header)

        LENGTH = bit2uint8(np.concatenate((C[1, 0:4], C[0, 0:4])))
        HAS_CRC = C[2, 0]  # HAS_CRC
        CR = bit2uint8(C[2, 1:4])
        FCS_HDR = bit2uint8(np.concatenate((C[4, 0:4], C[3, 0:4])))

        n_bits_hdr = PPM * 4 - 20
        PAYLOAD_bits = np.zeros((1, n_bits_hdr), dtype=np.uint8)
        for i in range(5, PPM):
            # C(i,4+(1:CR_hdr)) = mod(C(i,1:4)*Hamming_hdr,2)
            PAYLOAD_bits[0, (i - 5) * 4 + np.arange(0, 4, dtype=np.int32)] = C[i, 0:4]

    return FCS_OK, LENGTH, HAS_CRC, CR, PAYLOAD_bits


def num2binary(num, length=0):
    num = np.array([num], dtype=np.uint16)
    num = np.flip(num.view(np.uint8))
    num = np.unpackbits(num)
    return num[-length:]


def lora_payload_n_sym(SF, LENGTH, MAC_CRC, CR, n_bits_hdr):
    # bigger spreading factors (11 and 12) use 2 less bits per symbol
    if SF > 10:
        DE = 1

    else:
        DE = 0
    PPM = SF - 2 * DE
    n_bits_blk = PPM * 4
    n_bits_tot = 8 * LENGTH + 16 * MAC_CRC
    n_blk_tot = np.ceil((n_bits_tot - n_bits_hdr) / n_bits_blk)
    n_sym_blk = 4 + CR
    n_sym_payload = int(n_blk_tot * n_sym_blk)
    return n_sym_payload


def lora_payload_decode(SF, k_payload, PAYLOAD_hdr_bits, HAS_CRC, CR, LENGTH_FROM_HDR):
    # hamming parity check matrices
    Hamming_P1 = np.array(([1], [1], [1], [1]), dtype=np.uint8)
    Hamming_P2 = np.array(([1, 0], [1, 1], [1, 1], [0, 1]), dtype=np.uint8)
    Hamming_P3 = np.array(([1, 0, 1], [1, 1, 1], [1, 1, 0], [0, 1, 1]), dtype=np.uint8)
    Hamming_P4 = np.array(([1, 0, 1, 1], [1, 1, 1, 0], [1, 1, 0, 1], [0, 1, 1, 1]), dtype=np.uint8)

    if CR == 1:
        Hamming = Hamming_P1
    elif CR == 2:
        Hamming = Hamming_P2
    elif CR == 3:
        Hamming = Hamming_P3
    elif CR == 4:
        Hamming = Hamming_P4

    if SF > 10:
        DE = 1
    else:
        DE = 0
    PPM = SF - 2 * DE
    n_sym_blk = (4 + CR)
    n_bits_blk = PPM * 4
    intlv_blk_size = PPM * n_sym_blk
    [_, degray] = gray_lut(PPM)
    K = np.power(2, SF)
    n_sym_payload = len(k_payload)
    n_blk_tot = int(n_sym_payload / n_sym_blk)
    try:
        PAYLOAD = np.concatenate(
            (np.squeeze(PAYLOAD_hdr_bits), np.zeros((int(n_bits_blk * n_blk_tot)), dtype=np.uint8)))
    except ValueError:
        PAYLOAD = np.zeros((int(n_bits_blk * n_blk_tot)), dtype=np.uint8)
    payload_ofs = (PAYLOAD_hdr_bits.size)
    for blk in range(0, n_blk_tot):

        bits_blk = np.zeros((intlv_blk_size))
        for sym in range(0, n_sym_blk):
            bin = round((K - 2 - k_payload[(blk) * n_sym_blk + sym]) / np.power(2, (2 * DE)))
            # gray decode
            bits_blk[(sym) * PPM + np.arange(0, PPM, dtype=np.int32)] = num2binary(
                degray[int(np.mod(bin, np.power(2, PPM)))], PPM)

        # interleaving

        S = np.reshape(bits_blk, (PPM, (4 + CR)), order='F').transpose()
        C = np.zeros((PPM, 4 + CR), dtype=np.uint8)
        for ii in range(0, PPM):
            for jj in range(0, 4 + CR):
                C[ii, jj] = S[jj, np.mod(ii + jj, PPM)]

        # row flip
        C = np.flip(C, 0)

        for k in range(0, PPM):
            PAYLOAD[payload_ofs + np.arange(0, 4, dtype=np.int32)] = C[k, 0:4]
            payload_ofs = payload_ofs + 4

    # ----------------------------------------------------------- WHITENING
    W = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.uint8)
    W_fb = np.array([0, 0, 0, 1, 1, 1, 0, 1], dtype=np.uint8)
    for k in range(1, int(np.floor(len(PAYLOAD) / 8) + 1)):
        PAYLOAD[(k - 1) * 8 + np.arange(0, 8, dtype=np.int32)] = np.mod(
            PAYLOAD[(k - 1) * 8 + np.arange(0, 8, dtype=np.int32)] + W, 2)
        W1 = np.array([np.mod(np.sum(W * W_fb), 2)])
        W = np.concatenate((W1, W[0:-1]))

    # NOTE HOW THE TOTAL LENGTH IS 4 BYTES + THE PAYLOAD LENGTH
    # INDEED, THE FIRST 4 BYTES ENCODE DST, SRC, SEQNO AND LENGTH INFOS
    DST = bit2uint8(PAYLOAD[0:8])
    SRC = bit2uint8(PAYLOAD[8 + np.arange(0, 8, dtype=np.int32)])
    SEQNO = bit2uint8(PAYLOAD[8 * 2 + np.arange(0, 8, dtype=np.int32)])
    LENGTH = bit2uint8(PAYLOAD[8 * 3 + np.arange(0, 8, dtype=np.int32)])
    if (LENGTH == 0):
        LENGTH = LENGTH_FROM_HDR

    MSG_LENGTH = LENGTH - 4
    if (((LENGTH + 2) * 8 > len(PAYLOAD) and HAS_CRC) or (LENGTH * 8 > len(PAYLOAD) and not (HAS_CRC)) or LENGTH < 4):
        MAC_CRC_OK = False

        return MAC_CRC_OK, DST, SRC, SEQNO, None, HAS_CRC

    MSG = np.zeros((int(MSG_LENGTH)), dtype=np.uint8)
    for i in range(0, int(MSG_LENGTH)):
        MSG[i] = bit2uint8(PAYLOAD[8 * (4 + i) + np.arange(0, 8, dtype=np.int32)])

    if not HAS_CRC:
        MAC_CRC_OK = True
    else:
        # fprintf('CRC-16: 0x#02X#02X ',...
        # PAYLOAD(8*LENGTH+(1:8))*2.^(0:7)',...
        # PAYLOAD(8*LENGTH+8+(1:8))*2.^(0:7)')
        temp = CRC16(PAYLOAD[0:LENGTH * 8])
        temp = np.power(2, np.arange(0, 8, dtype=np.int32)) @ (np.reshape(temp, (8, 2), order='F'))
        # fprintf('(CRC-16: 0x#02X#02X)',temp(1),temp(2))

        if np.any(PAYLOAD[8 * LENGTH + np.arange(0, 16, dtype=np.int32)] != CRC16(PAYLOAD[0:8 * LENGTH])):
            # fprintf(' [CRC FAIL]\n')
            MAC_CRC_OK = False
        else:
            # fprintf(' [CRC OK]\n')
            MAC_CRC_OK = True

    return MAC_CRC_OK, DST, SRC, SEQNO, MSG, HAS_CRC


def bit2uint8(bits):
    return np.packbits(bits, bitorder='little')[0]


# LoRa preamble (reverse engineered from a real signal, is actually different from the one described in the patent)
def lora_preamble(n_preamble, k1, k2, BW, K, OSF, t0_frac=0, phi0=0):
    (u0, phi) = lora_chirp(+1, 0, BW, K, OSF, t0_frac, phi0)
    (u1, phi) = lora_chirp(+1, k1, BW, K, OSF, t0_frac, phi)
    (u2, phi) = lora_chirp(+1, k2, BW, K, OSF, t0_frac, phi)
    (d0, phi) = lora_chirp(-1, 0, BW, K, OSF, t0_frac, phi)
    (d0_4, phi) = chirp(BW / 2, K * OSF / 4, 1 / (BW * OSF), -np.power(BW, 2) / K, t0_frac, phi)
    s = np.concatenate((np.tile(u0, (n_preamble)), u1, u2, d0, d0, d0_4))
    return s, phi


# generate a lora chirp
def lora_chirp(mu, k, BW, K, OSF, t0_frac=0, phi0=0):
    fs = BW * OSF
    Ts = 1 / fs
    # number of samples in one period T
    N = K * OSF
    T = N * Ts
    # derivative of the instant frequency
    Df = mu * BW / T
    if k > 0:
        (s1, phi) = chirp(mu * BW * (1 / 2 - k / K), k * OSF, Ts, Df, t0_frac, phi0)
        (s2, phi) = chirp(-mu * BW / 2, (K - k) * OSF, Ts, Df, t0_frac, phi)
        s = np.concatenate((s1, s2))
    else:
        (s, phi) = chirp(-mu * BW / 2, K * OSF, Ts, Df, t0_frac, phi0)
    return s, phi


def samples_decoding(s, BW, N, Ts, K, OSF, Nrise, SF, Trise):
    max_packets = int(np.ceil(Ts * s.size / min_time_lora_packet))
    pack_array = np.empty(shape=(max_packets,), dtype=LoRaPacket)

    OSF = int(OSF)
    N = int(N)
    K = int(K)
    SF = int(SF)
    cumulative_index = 0
    last_index = 0
    received = 0

    while True:
        if s.size < N:
            print("size")
            break

        (success, payload, last_index, truncated, HDR_FCS_OK, MAC_CRC_OK, DST, SRC, SEQNO, CR, HAS_CRC,
         offset) = rf_decode(s[cumulative_index:], BW, N, Ts, K, OSF, Nrise, SF, Trise)

        if truncated:
            # print("Truncated")
            break

        if (success):
            # print("success")
            # print(payload)
            # print("message","".join([chr(int(item)) for item in payload]))
            # print("FCS Check", HDR_FCS_OK)
            # print("MAC CRC",MAC_CRC_OK)
            # print("PAYLOAD LENGTH", len(payload))
            pack_array[received] = LoRaPacket(payload, SRC, DST, SEQNO, HDR_FCS_OK, HAS_CRC, MAC_CRC_OK, CR, 0, SF, BW)
            received = received + 1

        if (last_index == -1):
            break

        else:
            # cumulative_index = cumulative_index + last_index + 10*N
            # cumulative_index = cumulative_index + last_index + 28 * N
            if (offset != -1):
                cumulative_index = cumulative_index + last_index + offset
            else:
                cumulative_index = cumulative_index + last_index + 28 * N
    return pack_array[:received]


def rf_decode(s, BW, N, Ts, K, OSF, Nrise, SF, Trise):
    truncated = False
    payload = None
    last_index = -1
    success = False
    HDR_FCS_OK = None
    MAC_CRC_OK = None
    DST = -1
    SRC = -1
    SEQNO = -1
    CR = -1
    HAS_CRC = -1
    ns = len(s)
    # base upchirp & downchirp
    u0 = chirp(-BW / 2, N, Ts, BW / (N * Ts))[0]

    d0 = np.conj(u0)

    # parameters and state variables in the synch block
    m_phases = 2
    m_phase = 0
    m_vec = -1 * np.ones((2 * m_phases, 6))  # peak values positioning in the DFT
    # oversampling factor for fine-grained esteem
    OSF_fine_sync = 4  # [Andreas] Maybe increase zero-padding?

    missed_sync = True
    sync_metric = np.Inf

    offset = -1
    main_loop = True
    s_ofs = 0

    #         s = s(620000:1000000)
    #         ns = numel(s)
    while main_loop:

        # sample window for a full chirp
        try:

            s_win = s[np.arange(s_ofs, s_ofs + N, dtype=np.int32)]
        except IndexError:
            success = False
            payload = None

            return success, payload, last_index, truncated, HDR_FCS_OK, MAC_CRC_OK, DST, SRC, SEQNO, CR, HAS_CRC, offset

        # multiplication of the signal with a downchirp and a upchirp, respectively.

        Su = np.abs(np.fft.fft(s_win * d0))
        Sd = np.abs(np.fft.fft(s_win * u0))
        m_u = np.argmax(Su)
        m_d = np.argmax(Sd)

        # convert the positioning in values in the range [-N/2,N/2-1]
        m_u = np.mod(m_u - 1 + N / 2, N) - N / 2
        m_d = np.mod(m_d - 1 + N / 2, N) - N / 2

        m_vec[np.arange(m_phase * 2, m_phase * 2 + 2), 1:] = m_vec[np.arange(m_phase * 2, m_phase * 2 + 2), :-1]
        m_vec[np.arange(m_phase * 2, m_phase * 2 + 2), 0] = np.array(
            [m_u, m_d])  # Numpy automatically converts the row into a column

        # three upchirpsm followed by two upchirps shifted by
        # 8 e 16 bits, further followed by two downchirps, correspond to
        # a preamble
        if np.abs(m_vec[m_phase * 2 + 1, 0] - m_vec[m_phase * 2 + 1, 1]) <= 1 and \
                np.abs(m_vec[m_phase * 2, 2] - m_vec[m_phase * 2, 3] - 8) <= 1 and \
                np.abs(m_vec[m_phase * 2, 3] - m_vec[m_phase * 2, 4] - 8) <= 1 and \
                np.abs(m_vec[m_phase * 2, 4] - m_vec[m_phase * 2, 5]) <= 1:

            missed_sync = False
            # keyboard
            tmp = np.sum(np.abs(m_vec[m_phase * 2 + 1, 1:2]) + np.abs(m_vec[m_phase * 2, 5:6]))
            if tmp < sync_metric:
                sync_metric = tmp

                # fprintf('phase: #g\n',m_phase)
                # display(m_vec(m_phase*2+(1:2),:))
                Nu = 2
                # fine-grained estimation of the positions of the maximums in the DFT
                m_u0 = 0
                for i in range(1, Nu + 1):
                    try:
                        Su = np.abs(np.fft.fft(
                            ((s[np.arange(s_ofs - (4 + i) * N, s_ofs - (4 + i) * N + N, dtype=np.int32)]) * d0),
                            N * OSF_fine_sync))
                    except IndexError:
                        truncated = True
                        return success, payload, last_index, truncated, HDR_FCS_OK, MAC_CRC_OK, DST, SRC, SEQNO, CR, HAS_CRC, offset

                    m_u = np.argmax(Su)
                    if (m_u > 0 and m_u < N * OSF_fine_sync - 1):
                        m_u = m_u + 0.5 * (Su[m_u - 1] - Su[m_u + 1]) / (Su[m_u - 1] - 2 * Su[m_u] + Su[m_u + 1])

                    m_u0 = m_u0 + np.mod(m_u - 1 + N * OSF_fine_sync / 2, N * OSF_fine_sync) - N * OSF_fine_sync / 2

                m_u0 = m_u0 / Nu

                Nd = 2
                m_d0 = 0
                for i in range(1, Nd + 1):
                    Sd = np.abs(
                        np.fft.fft(((s[np.arange(s_ofs - (i - 1) * N, s_ofs - (i - 1) * N + N, dtype=np.int32)]) * u0),
                                   int(N * OSF_fine_sync)))
                    m_d = np.argmax(Sd)
                    if m_d > 1 and m_d < N * OSF_fine_sync:
                        try:
                            m_d = m_d + 0.5 * (Sd[m_d - 1] - Sd[m_d + 1]) / (Sd[m_d - 1] - 2 * Sd[m_d] + Sd[m_d + 1])
                        except IndexError:
                            pass

                    m_d0 = m_d0 + np.mod(m_d - 1 + N * OSF_fine_sync / 2, N * OSF_fine_sync) - N * OSF_fine_sync / 2

                m_d0 = m_d0 / Nd

                # Cfo_est: frequency error
                # t_est: timing error
                Cfo_est = (m_u0 + m_d0) / 2 * BW / K / OSF_fine_sync
                _internal["Cfo_est"] = Cfo_est  # [Andreas] store info

                t_est = (
                                    m_d0 - m_u0) * OSF / 2 / OSF_fine_sync + s_ofs - 11 * N - Nrise  # n_pr = 8 + 2 syncword + 1 downchirp
                _internal["t_est"] = t_est  # [Andreas] store info
                break

        m_phase = np.mod(m_phase + 1, m_phases)
        s_ofs = s_ofs + N / m_phases
        if s_ofs + N > ns:
            main_loop = False
            success = False
            return success, payload, last_index, truncated, HDR_FCS_OK, MAC_CRC_OK, DST, SRC, SEQNO, CR, HAS_CRC, offset

    if not missed_sync:

        missed_sync = True

        # symbol-level receiver
        p_ofs_est = int(np.ceil(t_est))
        _internal["p_ofs_est"] = p_ofs_est  # [Andreas] store info

        last_index = p_ofs_est

        t0_frac_est = np.mod(-t_est, 1)

        # keyboard
        (k_hdr_est, HDR_FCS_OK, k_payload_est, MAC_CRC_OK, MSG, DST, SRC, SEQNO, CR, HAS_CRC, truncated,
         offset) = lora_packet_rx(s, SF, BW, OSF, Trise, p_ofs_est, Cfo_est)

        if (not truncated) and (HDR_FCS_OK and MAC_CRC_OK):
            n_sym_hdr = len(k_hdr_est)
            n_sym_payload = len(k_payload_est)
            rx_success = True
            success = True
            payload = MSG

        return success, payload, last_index, truncated, HDR_FCS_OK, MAC_CRC_OK, DST, SRC, SEQNO, CR, HAS_CRC, offset


# SUPPORT FUNCTION TO ENCODE A LORA PACKET
def complex_lora_packet(K, n_pr, IH, CR, MAC_CRC, SRC, DST, SEQNO, BW, OSF, SF, Trise, N, Ts, fs, Cfo_PPM, f0, MESSAGE,
                        t0_frac, phi0):
    k1 = K - 8
    k2 = K - 16
    SF = np.uint8(SF)
    p = lora_packet(BW, OSF, SF, k1, k2, n_pr, IH, CR, MAC_CRC, SRC, DST, SEQNO, MESSAGE, Trise, t0_frac, phi0)[0]
    size_p = p.size

    ntail = N
    Ttail = ntail * Ts
    T = np.ceil(size_p * Ts + Ttail)  # WHOLE NUMBER OF SECONDS
    ns = int(T * fs)
    # 先建立全零缓冲区，再把完整 LoRa 波形写入其中。这里的 0 表示理想
    # 静默基带，不是从接收机采集到的 off-packet 环境噪声。
    s = np.zeros(ns, dtype=np.complex64)
    # p_ofs=ceil((ns-np-ntail)*rand)
    # 固定保留 10000 个高采样率前导静默样点。SyntheticLoRaDataset 对
    # 低采样率输入整体加 AWGN 后，这一段才会变成纯 AWGN 噪声。
    p_ofs = 10000

    t = np.arange(0, size_p) * Ts
    # Cfo_TX = Cfo_PPM*1e-6*f0*(2*rand-1)
    # Cfo = Cfo_TX+Cfo_PPM*1e-6*f0*(2*rand-1)
    Cfo_TX = Cfo_PPM * 1e-6 * f0 * 1
    Cfo = Cfo_TX + Cfo_PPM * 1e-6 * f0 * 1
    s[p_ofs + np.arange(0, size_p)] = s[p_ofs + np.arange(0, size_p)] + p * np.exp(1j * (2 * np.pi * Cfo * t + phi0))
    s = s[np.arange(0, p_ofs + size_p)]
    return s


# ENCODER FUNCTION. GIVEN A PAYLOAD (IN BYTES), AND THE INTENDED TRANSMISSION PARAMETERS, GENERATES THE SAMPLES FOR THE CORRESPONDING LORA PACKET
def encode(f0, SF, BW, payload, fs, src, dst, seqn, cr=1, enable_crc=1, implicit_header=0, preamble_bits=8):
    OSF = fs / BW
    Ts = 1 / fs
    Nrise = np.ceil(Trise * fs)
    K = np.power(2, SF)
    N = K * OSF
    # t0_frac = rand
    t0_frac = 0
    # phi0 = 2*pi*rand
    phi0 = 0
    complex_samples = complex_lora_packet(K, preamble_bits, implicit_header, cr, enable_crc, src, dst, seqn, BW, OSF,
                                          SF, Trise, N,
                                          Ts, fs, Cfo_PPM, f0, payload, t0_frac, phi0)
    return complex_samples


# DECODER FUNCTION. LOOKS FOR LORA PACKETS IN THE INPUT COMPLEX SAMPLES. RETURNS ALL THE PACKETS FOUND IN THE SAMPLES.

def decode(complex_samples, SF, BW, fs, filename=""):
    _internal = {}  # [Andreas] reset internal state tracking
    global _filename
    _filename = filename  # [Andreas]

    OSF = fs / BW
    Ts = 1 / fs
    Nrise = np.ceil(Trise * fs)

    K = np.power(2, SF)
    N = K * OSF
    return samples_decoding(complex_samples, BW, N, Ts, K, OSF, Nrise, SF, Trise)


# CLASS TO CONVENIENTLY ENCAPSULATE LORA PACKETS
class LoRaPacket:
    def __init__(self, payload, src, dst, seqn, hdr_ok, has_crc, crc_ok, cr, ih, SF, BW):
        self.payload = payload
        self.src = np.uint8(src)
        self.dst = np.uint8(dst)
        self.seqn = np.uint8(seqn)
        self.hdr_ok = np.uint8(hdr_ok)
        self.has_crc = np.uint8(has_crc)
        self.crc_ok = np.uint8(crc_ok)
        self.cr = np.uint8(cr)
        self.ih = np.uint8(ih)
        self.SF = np.uint8(SF)
        self.BW = BW

    def __eq__(self, other):
        payload_eq = np.all(self.payload == other.payload)
        src_eq = (self.src == other.src)
        dst_eq = (self.dst == other.dst)
        seqn_eq = (self.seqn == other.seqn)
        hdr_ok_eq = (self.hdr_ok == other.hdr_ok)
        has_crc_eq = (self.has_crc == other.has_crc)
        crc_ok_eq = (self.crc_ok == other.crc_ok)
        cr_eq = (self.cr == other.cr)
        ih_eq = (self.ih == other.ih)
        SF_eq = (self.SF == other.SF)
        BW_eq = (self.BW == other.BW)

        return payload_eq and src_eq and dst_eq and seqn_eq and hdr_ok_eq and has_crc_eq and crc_ok_eq and cr_eq and ih_eq and SF_eq and BW_eq

    def __repr__(self):
        desc = "LoRa Packet Info:\n"
        sf_desc = "Spreading Factor: " + str(self.SF) + "\n"
        bw_desc = "Bandwidth: " + str(self.BW) + "\n"
        if not self.hdr_ok:
            hdr_chk = "Header Integrity Check Failed" + "\n"
            return desc + sf_desc + bw_desc + hdr_chk

        else:
            if self.ih:
                ih_desc = "Implicit Header ON" + "\n"
                if self.has_crc:
                    if self.crc_ok:
                        crc_check = "Payload Integrity Check OK" + "\n"
                        pl_str = "Payload: " + str(self.payload) + "\n"
                        pl_len = "Payload Length: " + str(self.payload.size) + "\n"
                        return desc + sf_desc + bw_desc + ih_desc + crc_check + pl_len + pl_str
                    else:
                        crc_check = "Payload Integrity Check Failed" + "\n"
                        return desc + sf_desc + bw_desc + ih_desc + crc_check
                else:
                    crc_check = "CRC Disabled for this packet. Payload may be corrupted" + "\n"
                    pl_str = "Payload: " + str(self.payload) + "\n"
                    pl_len = "Payload Length: " + str(self.payload.size) + "\n"
                    return desc + sf_desc + bw_desc + ih_desc + pl_len + pl_str
            else:
                ih_desc = "Explicit Header ON" + "\n"
                hdr_chk = "Header Integrity Check OK" + "\n"
                src_desc = "Source: " + str(self.src) + "\n"
                dest_desc = "Destination: " + str(self.dst) + "\n"
                seq_desc = "Sequence number: " + str(self.seqn) + "\n"
                cr_desc = "Coding Rate: " + str(self.cr) + "\n"
                if self.has_crc:
                    if self.crc_ok:
                        crc_check = "Payload Integrity Check OK" + "\n"
                        pl_str = "Payload: " + str(self.payload) + "\n"
                        pl_len = "Payload Length: " + str(self.payload.size) + "\n"
                        return desc + sf_desc + bw_desc + hdr_chk + ih_desc + src_desc + dest_desc + seq_desc + cr_desc + crc_check + pl_len + pl_str
                    else:
                        crc_check = "Payload Integrity Check Failed" + "\n"
                        return desc + sf_desc + bw_desc + hdr_chk + ih_desc + src_desc + dest_desc + seq_desc + cr_desc + crc_check
                else:
                    crc_check = "CRC Check Disabled for this packet. Payload may be corrupted." + "\n"
                    pl_str = "Payload: " + str(self.payload) + "\n"
                    pl_len = "Payload Length: " + str(self.payload.size) + "\n"
                    return desc + sf_desc + bw_desc + hdr_chk + ih_desc + src_desc + dest_desc + seq_desc + cr_desc + crc_check + pl_len + pl_str


def compute_snr_from_clean(x_signal: np.ndarray, x_signal_with_noise: np.ndarray):
    """
    Computes SNR given a clean signal and its noisy version.

    Args:
        x_signal (np.ndarray): Clean signal (complex-valued).
        x_signal_with_noise (np.ndarray): Noisy signal (complex-valued).

    Returns:
        snr_linear (float): SNR in linear scale.
        snr_db (float): SNR in dB.
    """
    if x_signal.shape != x_signal_with_noise.shape:
        raise ValueError("x_signal and x_signal_with_noise must have the same shape")

    x_noise = x_signal_with_noise - x_signal

    signal_power = np.mean(np.abs(x_signal) ** 2)
    noise_power = np.mean(np.abs(x_noise) ** 2)

    if noise_power == 0:
        raise ZeroDivisionError("Noise power is zero, SNR is infinite")

    snr_linear = signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear)

    return snr_linear, snr_db


def bit_distance(a: int, b: int) -> int:
    # XOR highlights different bits

    diff = a ^ b
    # Count number of 1s in the XOR result
    return bin(diff & 0xFFFF).count("1")  # mask for int16


def symbol_hamming_distance(sym1: int, sym2: int, sf: int) -> int:
    """
    Compute the Hamming distance between two LoRa symbols.

    Parameters
    ----------
    sym1 : int
        First symbol value (0 <= sym < 2^sf).
    sym2 : int
        Second symbol value (0 <= sym < 2^sf).
    sf : int
        Spreading factor (7 <= sf <= 12).

    Returns
    -------
    int
        Hamming distance (number of differing bits).
    """
    if not (7 <= sf <= 12):
        raise ValueError("Spreading factor must be between 7 and 12")

    if not (0 <= sym1 < 2 ** sf and 0 <= sym2 < 2 ** sf):
        raise ValueError(f"Symbols must be in range 0 .. {2 ** sf - 1}")

    # XOR highlights differing bits
    diff = sym1 ^ sym2

    # Count set bits (number of 1's in binary)
    return bin(diff).count("1")

    # if __name__ == "__main__":
    #
    #
    #     for snr in range(-30, 0, 2):
    #
    #         center_freq = 915e6
    #         sf = 12
    #         bw = 125e3
    #         payload = np.array(range(16)).astype(np.uint8) #np.ones(16, dtype=np.uint8)
    #         sample_rate = 0.25e6
    #         src = 0
    #         dst = 1
    #         seqn = 7
    #
    #         x = encode(center_freq, sf, bw, payload, sample_rate, src, dst, seqn, 4, 1, 0, 8)
    #
    #         # filename = 'data/iq_samples.bin'
    #         # x = x.astype(np.complex128)
    #         # x.tofile(filename)
    #
    #         x = awgn(x, snr)
    #
    #         from scipy import signal
    #         def resample(data, up=2, down=1):
    #             f_poly = signal.resample_poly(data, up, down)
    #             return f_poly
    #
    #
    #         ups = 1
    #         x = resample(x, ups, 1)
    #
    #
    #         test = decode(x, sf, bw, sample_rate*ups)
    #
    #
    #         if len(test) == 0:
    #             print(f"SNR={snr}: failed")
    #         else:
    #             print(f"SNR={snr}: success")
    #

    # print(type(x[0]))
    # x = x.astype(np.complex64)
    # print(type(x[0]))
    # x.tofile("id_output.c64")
    # # simulate the signal above, or use your own signal
    # #

    # import numpy as np
    #
    # for snr in [-10]: #[-35, -33, -31, -29, -27, -25, -23, -21, -19, -17, -15, -13, -11]:
    #
    #
    #     filename = f'data/iq_samples_snr_{snr}.bin'
    #     samples = np.fromfile(filename, dtype=np.complex128)
    #
    #
    #     sf = 12
    #     bw = 125e3
    #     sample_rate = 250e3
    #
    #
    #     test = decode(samples, sf, bw, sample_rate)
    #
    #     if len(test) == 0:
    #         print(f"SNR={snr}: failed")
    #     else:
    #         print(f"SNR={snr}: success")
    #
    #     print()
    #
    print()
