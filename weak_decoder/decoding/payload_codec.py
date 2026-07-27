"""复刻 gr-lora_sdr payload 编解码链的离线工具。

这个模块不负责同步、dechirp 或 FFT，也不直接读取 IQ。它的作用是把
LoRa PHY payload bytes、nibble/codeword 流和 demod symbol 序列对齐起来：

* 正向：payload bytes -> whitening/CRC/Hamming/interleaver/gray -> symbols。
  用于把当前 packet decoder 产生的 payload 候选投影成它理论上应该对应的
  FFT bin/symbol 轨迹。
* 反向：header symbols + payload symbols -> header/payload bytes/CRC。
  用于离线检查某条 hard-decision symbol 序列是否能解出合法 payload。

输入通常来自 header-first demod、symbol-level selector 或 two-stage decoder
当前 packet 内部生成的候选。这个模块不学习 session byte template，也不使用
payload 结构先验。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .header_first_demod import (
    HeaderDecodeResult,
    decode_explicit_header,
    deinterleave_hard,
    gray_demapping,
    hamming_decode_hard,
)


WHITENING_SEQ = (
    0xFF, 0xFE, 0xFC, 0xF8, 0xF0, 0xE1, 0xC2, 0x85,
    0x0B, 0x17, 0x2F, 0x5E, 0xBC, 0x78, 0xF1, 0xE3,
    0xC6, 0x8D, 0x1A, 0x34, 0x68, 0xD0, 0xA0, 0x40,
    0x80, 0x01, 0x02, 0x04, 0x08, 0x11, 0x23, 0x47,
    0x8E, 0x1C, 0x38, 0x71, 0xE2, 0xC4, 0x89, 0x12,
    0x25, 0x4B, 0x97, 0x2E, 0x5C, 0xB8, 0x70, 0xE0,
    0xC0, 0x81, 0x03, 0x06, 0x0C, 0x19, 0x32, 0x64,
    0xC9, 0x92, 0x24, 0x49, 0x93, 0x26, 0x4D, 0x9B,
    0x37, 0x6E, 0xDC, 0xB9, 0x72, 0xE4, 0xC8, 0x90,
    0x20, 0x41, 0x82, 0x05, 0x0A, 0x15, 0x2B, 0x56,
    0xAD, 0x5B, 0xB6, 0x6D, 0xDA, 0xB5, 0x6B, 0xD6,
    0xAC, 0x59, 0xB2, 0x65, 0xCB, 0x96, 0x2C, 0x58,
    0xB0, 0x61, 0xC3, 0x87, 0x0F, 0x1F, 0x3E, 0x7D,
    0xFB, 0xF6, 0xED, 0xDB, 0xB7, 0x6F, 0xDE, 0xBD,
    0x7A, 0xF5, 0xEB, 0xD7, 0xAE, 0x5D, 0xBA, 0x74,
    0xE8, 0xD1, 0xA2, 0x44, 0x88, 0x10, 0x21, 0x43,
    0x86, 0x0D, 0x1B, 0x36, 0x6C, 0xD8, 0xB1, 0x63,
    0xC7, 0x8F, 0x1E, 0x3C, 0x79, 0xF3, 0xE7, 0xCE,
    0x9C, 0x39, 0x73, 0xE6, 0xCC, 0x98, 0x31, 0x62,
    0xC5, 0x8B, 0x16, 0x2D, 0x5A, 0xB4, 0x69, 0xD2,
    0xA4, 0x48, 0x91, 0x22, 0x45, 0x8A, 0x14, 0x29,
    0x52, 0xA5, 0x4A, 0x95, 0x2A, 0x54, 0xA9, 0x53,
    0xA7, 0x4E, 0x9D, 0x3B, 0x77, 0xEE, 0xDD, 0xBB,
    0x76, 0xEC, 0xD9, 0xB3, 0x67, 0xCF, 0x9E, 0x3D,
    0x7B, 0xF7, 0xEF, 0xDF, 0xBF, 0x7E, 0xFD, 0xFA,
    0xF4, 0xE9, 0xD3, 0xA6, 0x4C, 0x99, 0x33, 0x66,
    0xCD, 0x9A, 0x35, 0x6A, 0xD4, 0xA8, 0x51, 0xA3,
    0x46, 0x8C, 0x18, 0x30, 0x60, 0xC1, 0x83, 0x07,
    0x0E, 0x1D, 0x3A, 0x75, 0xEA, 0xD5, 0xAA, 0x55,
    0xAB, 0x57, 0xAF, 0x5F, 0xBE, 0x7C, 0xF9, 0xF2,
    0xE5, 0xCA, 0x94, 0x28, 0x50, 0xA1, 0x42, 0x84,
    0x09, 0x13, 0x27, 0x4F, 0x9F, 0x3F, 0x7F,
)


@dataclass(frozen=True)
class PayloadDecodeResult:
    """payload symbol 硬解码后的结果。

    payload_bytes 是去白化后的真实 payload；crc_bytes 是接在 payload 后的
    CRC 字节；crc_valid 表示按指定 crc_mode 重新计算后是否匹配。
    """

    payload_bytes: bytes
    crc_bytes: bytes
    crc_computed: int
    crc_received: int
    crc_valid: bool
    decoded_nibbles: tuple[int, ...]
    codewords: tuple[int, ...]


@dataclass(frozen=True)
class ExplicitFrameDecodeResult:
    """explicit-header LoRa frame 的硬解码结果。

    GNU Radio 接收链会把 8 个 explicit header symbols 当成一个低码率
    interleaver block 解码。解出的前 5 个 nibble 是 PHY header 字段；
    剩余 nibble 已经属于 payload/CRC 流的开头。residual search 必须保留
    这些 header tail nibbles，否则后面的 payload byte 会整体错位。
    """

    header: HeaderDecodeResult
    payload: PayloadDecodeResult
    header_tail_nibbles: tuple[int, ...]
    payload_symbol_nibbles: tuple[int, ...]
    frame_nibbles: tuple[int, ...]


def crc16(data: bytes | bytearray | Sequence[int]) -> int:
    """CRC-16(poly=0x1021, init=0)，与 gr-lora_sdr 的 crc_verif 对齐。"""
    crc = 0
    for byte in data:
        new_byte = int(byte) & 0xFF
        for _ in range(8):
            if (((crc & 0x8000) >> 8) ^ (new_byte & 0x80)):
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
            new_byte = (new_byte << 1) & 0xFF
    return int(crc)


def _int_to_bits_msb(value: int, n_bits: int) -> list[int]:
    return [
        (int(value) >> bit) & 1
        for bit in range(int(n_bits) - 1, -1, -1)
    ]


def _bits_to_int(bits: Iterable[int | bool]) -> int:
    out = 0
    for bit in bits:
        out = (out << 1) | int(bool(bit))
    return int(out)


def payload_bytes_to_whitened_nibbles(
    payload: bytes | bytearray | Sequence[int],
) -> list[int]:
    """对 payload 做 gr-lora_sdr whitening，并输出低/高 nibble 流。

    输入：原始 payload byte 序列。
    输出：每个 byte 去低 4 bit、再取高 4 bit 的 whitened nibble 序列。
    这个序列后面会继续进入 CRC、Hamming 和 interleaver。
    """
    nibbles: list[int] = []
    for offset, value in enumerate(payload):
        whitened = (int(value) & 0xFF) ^ int(WHITENING_SEQ[offset])
        nibbles.append(whitened & 0x0F)
        nibbles.append((whitened >> 4) & 0x0F)
    return nibbles


def grlora_crc_nibbles(payload: bytes | bytearray | Sequence[int]) -> list[int]:
    """生成 gr-lora_sdr add_crc_impl 风格的 CRC nibbles。"""
    data = bytes(int(v) & 0xFF for v in payload)
    crc = crc16(data[: max(0, len(data) - 2)])
    if len(data) >= 2:
        crc ^= int(data[-1])
        crc ^= int(data[-2]) << 8
    return [
        crc & 0x000F,
        (crc & 0x00F0) >> 4,
        (crc & 0x0F00) >> 8,
        (crc & 0xF000) >> 12,
    ]


def sx1276_crc_nibbles(payload: bytes | bytearray | Sequence[int]) -> list[int]:
    """生成标准 SX1276 风格的 CRC nibbles，低 nibble 在前。"""
    crc = crc16(bytes(int(v) & 0xFF for v in payload))
    return [
        crc & 0x000F,
        (crc & 0x00F0) >> 4,
        (crc & 0x0F00) >> 8,
        (crc & 0xF000) >> 12,
    ]


def compute_header_nibbles(
    payload_len: int,
    cr: int,
    has_crc: bool,
) -> list[int]:
    """构造 explicit header 的 5 个 nibbles。

    输入来自已知/解出的 PHY 参数：payload_len、CR、是否带 CRC。
    输出对应 gr-lora_sdr header_impl 使用的 3 个字段 nibble + 2 个校验 nibble。
    """
    n0 = (int(payload_len) >> 4) & 0x0F
    n1 = int(payload_len) & 0x0F
    n2 = ((int(cr) & 0x07) << 1) | int(bool(has_crc))
    c4 = ((n0 & 0b1000) >> 3) ^ ((n0 & 0b0100) >> 2) ^ ((n0 & 0b0010) >> 1) ^ (n0 & 0b0001)
    c3 = ((n0 & 0b1000) >> 3) ^ ((n1 & 0b1000) >> 3) ^ ((n1 & 0b0100) >> 2) ^ ((n1 & 0b0010) >> 1) ^ (n2 & 0b0001)
    c2 = ((n0 & 0b0100) >> 2) ^ ((n1 & 0b1000) >> 3) ^ (n1 & 0b0001) ^ ((n2 & 0b1000) >> 3) ^ ((n2 & 0b0010) >> 1)
    c1 = ((n0 & 0b0010) >> 1) ^ ((n1 & 0b0100) >> 2) ^ (n1 & 0b0001) ^ ((n2 & 0b0100) >> 2) ^ ((n2 & 0b0010) >> 1) ^ (n2 & 0b0001)
    c0 = (n0 & 0b0001) ^ ((n1 & 0b0010) >> 1) ^ ((n2 & 0b1000) >> 3) ^ ((n2 & 0b0100) >> 2) ^ ((n2 & 0b0010) >> 1) ^ (n2 & 0b0001)
    return [n0, n1, n2, int(c4), int((c3 << 3) | (c2 << 2) | (c1 << 1) | c0)]


def encode_hamming_nibble(nibble: int, cr_app: int) -> int:
    """按 hamming_enc_impl 的硬编码路径编码一个 nibble。"""
    data_bin = _int_to_bits_msb(int(nibble) & 0x0F, 4)
    if int(cr_app) != 1:
        p0 = data_bin[3] ^ data_bin[2] ^ data_bin[1]
        p1 = data_bin[2] ^ data_bin[1] ^ data_bin[0]
        p2 = data_bin[3] ^ data_bin[2] ^ data_bin[0]
        p3 = data_bin[3] ^ data_bin[1] ^ data_bin[0]
        full = (
            (data_bin[3] << 7)
            | (data_bin[2] << 6)
            | (data_bin[1] << 5)
            | (data_bin[0] << 4)
            | (p0 << 3)
            | (p1 << 2)
            | (p2 << 1)
            | p3
        )
        return int(full >> (4 - int(cr_app)))
    p4 = data_bin[0] ^ data_bin[1] ^ data_bin[2] ^ data_bin[3]
    return int(
        (data_bin[3] << 4)
        | (data_bin[2] << 3)
        | (data_bin[1] << 2)
        | (data_bin[0] << 1)
        | p4
    )


def hamming_encode_nibbles(
    nibbles: Sequence[int],
    sf: int,
    cr: int,
) -> list[int]:
    """对 nibble 流做 Hamming 编码，并保留首块 CR=4/8 规则。

    gr-lora_sdr/LoRa 的前 sf-2 个 codeword 使用更强的 header-like 保护，
    之后才按 payload CR 编码。这个细节会影响重编码后的 symbol 对齐。
    """
    out: list[int] = []
    for idx, nibble in enumerate(nibbles):
        cr_app = 4 if int(idx) < int(sf) - 2 else int(cr)
        out.append(encode_hamming_nibble(int(nibble), cr_app=cr_app))
    return out


def interleave_codewords(
    codewords: Sequence[int],
    sf: int,
    cr: int,
    ldro: bool,
    frame_len_nibbles: int | None = None,
) -> list[int]:
    """复刻 interleaver_impl 的硬路径，输出 gray 输入符号。

    输入：Hamming 后的 codeword 序列。
    输出：尚未 gray_demap 的 symbol 整数序列。
    """
    total_len = int(frame_len_nibbles) if frame_len_nibbles is not None else len(codewords)
    cw_cnt = 0
    out: list[int] = []
    values = [int(v) for v in codewords]
    pos = 0
    while pos < len(values):
        first_or_ldro = (cw_cnt < int(sf) - 2) or bool(ldro)
        cw_len = 4 + (4 if cw_cnt < int(sf) - 2 else int(cr))
        sf_app = int(sf) - 2 if first_or_ldro else int(sf)
        remaining = len(values) - pos
        nitems = min(remaining, sf_app)
        if nitems < sf_app and cw_cnt + nitems != total_len:
            break

        cw_bin: list[list[int]] = []
        for i in range(sf_app):
            if i >= nitems:
                cw_bin.append(_int_to_bits_msb(0, cw_len))
            else:
                cw_bin.append(_int_to_bits_msb(values[pos + i], cw_len))
            cw_cnt += 1
        pos += nitems

        for i in range(cw_len):
            row = [0 for _ in range(int(sf))]
            for j in range(sf_app):
                row[j] = cw_bin[(i - j - 1) % sf_app][i]
            if cw_cnt == int(sf) - 2 or bool(ldro):
                row[sf_app] = sum(row[:sf_app]) % 2
            out.append(_bits_to_int(row))
    return out


def gray_demap_symbols(symbol_values: Sequence[int], sf: int) -> list[int]:
    """复刻 gray_demap_impl：binary-coded symbols -> modulator symbols。"""
    out: list[int] = []
    mask = (1 << int(sf)) - 1
    for value in symbol_values:
        v = int(value) & mask
        g = v
        for shift in range(1, int(sf)):
            g ^= v >> shift
        out.append((int(g) + 1) & mask)
    return out


def interleaved_to_fft_demod_symbols(
    interleaved_values: Sequence[int],
    sf: int,
    ldro: bool,
) -> list[int]:
    """把 interleaver 输出转换成本地 demod 看到的 hard symbol values。

    返回值对应 run_header_first_demod/phase_guided_demod 里记录的
    symbol_value，而不是原始 FFT bin。后续评分时会再映射回 canonical bin。
    """
    n_bins = 1 << int(sf)
    modulator_symbols = gray_demap_symbols(interleaved_values, sf=int(sf))
    out: list[int] = []
    for idx, raw_bin in enumerate(modulator_symbols):
        divisor = 4 if idx < 8 or bool(ldro) else 1
        out.append(((int(raw_bin) - 1) % n_bins) // divisor)
    return out


def encode_explicit_frame_symbols(
    payload: bytes | bytearray | Sequence[int],
    sf: int,
    cr: int,
    has_crc: bool,
    ldro: bool,
    crc_mode: str = "grlora",
) -> tuple[list[int], list[int]]:
    """把 payload bytes 重编码成 explicit header symbols 和 payload symbols。

    输入：
      payload: 一个完整 payload byte 候选，通常来自当前 packet 的 hard decision
        或 decoder beam。
      sf/cr/has_crc/ldro/crc_mode: header-first 解码得到的 PHY 参数，或实验
        命令行指定的参数。

    输出：
      (header_symbols, payload_symbols)。二者都是本地 demod 语义下的
      symbol_value 序列。Phase-MAP 会拿 payload_symbols 去真实 FFT 证据里
      查对应 bin 的相位/幅度，从而给这个 byte 候选打分。
    """
    header_modulator, payload_modulator = encode_explicit_frame_modulator_symbols(
        payload,
        sf=int(sf),
        cr=int(cr),
        has_crc=bool(has_crc),
        ldro=bool(ldro),
        crc_mode=str(crc_mode),
    )
    modulator_symbols = header_modulator + payload_modulator
    n_bins = 1 << int(sf)
    demod_symbols = [
        ((int(raw_bin) - 1) % n_bins) // (4 if idx < 8 or bool(ldro) else 1)
        for idx, raw_bin in enumerate(modulator_symbols)
    ]
    return demod_symbols[:8], demod_symbols[8:]


def encode_explicit_frame_modulator_symbols(
    payload: bytes | bytearray | Sequence[int],
    sf: int,
    cr: int,
    has_crc: bool,
    ldro: bool,
    crc_mode: str = "grlora",
) -> tuple[list[int], list[int]]:
    """把 raw PHY payload 编码成 LoRa 调制器直接使用的 symbol IDs。

    与 :func:`encode_explicit_frame_symbols` 不同，本函数不把调制器 symbol
    换算成本地接收机记录的 reduced-rate ``symbol_value``。返回值可以直接
    交给 LoRa upchirp 调制器生成 IQ。

    ``payload`` 已经是完整 PHY payload；本函数不会添加 DST/SRC/SEQ/LENGTH
    等应用层私有头。PHY explicit header 和可选 PHY CRC 仍按 LoRa 规则生成。
    """
    payload_bytes = bytes(int(v) & 0xFF for v in payload)

    # 1) 先构造 PHY header nibbles，再拼接 whitened payload nibbles。
    nibbles = compute_header_nibbles(
        payload_len=len(payload_bytes),
        cr=int(cr),
        has_crc=bool(has_crc),
    )
    nibbles.extend(payload_bytes_to_whitened_nibbles(payload_bytes))

    # 2) 如果打开 PHY CRC，把 CRC 也作为 nibble 流的一部分参与编码。
    if bool(has_crc):
        if str(crc_mode).lower() == "sx1276":
            nibbles.extend(sx1276_crc_nibbles(payload_bytes))
        else:
            nibbles.extend(grlora_crc_nibbles(payload_bytes))

    # 3) 复刻 TX PHY 编码：Hamming -> interleaver -> gray demap/demod symbol。
    codewords = hamming_encode_nibbles(nibbles, sf=int(sf), cr=int(cr))
    interleaved = interleave_codewords(
        codewords,
        sf=int(sf),
        cr=int(cr),
        ldro=bool(ldro),
        frame_len_nibbles=len(nibbles),
    )
    symbols = gray_demap_symbols(interleaved, sf=int(sf))
    return symbols[:8], symbols[8:]


def reencoded_payload_known_prefix_symbols(
    payload_len: int,
    has_crc: bool,
    sf: int,
    cr: int,
    ldro: bool,
) -> int:
    """计算由 payload/CRC bytes 完全约束的 payload symbol 前缀长度。

    explicit header 解码会从 header block 转发 `sf-2-5` 个 payload nibbles。
    剩下的 payload/CRC nibbles 由 payload interleaver blocks 承载。只有完整
    block 能由 bytes 唯一确定；最后一个不满 block 可能包含 padding codewords，
    padding 改变不一定影响最终 payload/CRC，因此不应拿来做强约束评分。
    """
    header_tail = max(0, int(sf) - 2 - 5)
    needed_nibbles = int(payload_len) * 2 + (4 if bool(has_crc) else 0)
    payload_side_nibbles = max(0, needed_nibbles - header_tail)
    sf_app = int(sf) - 2 if bool(ldro) else int(sf)
    full_blocks = payload_side_nibbles // max(1, sf_app)
    return int(full_blocks * (int(cr) + 4))


def payload_symbols_to_nibbles(
    symbol_values: Iterable[int],
    sf: int,
    cr: int,
    ldro: bool,
) -> list[int]:
    """把 payload LoRa symbols 硬解码成 whitened payload/CRC nibbles。"""
    values = [int(v) for v in symbol_values]
    cw_len = int(cr) + 4
    nibbles: list[int] = []
    for start in range(0, len(values), cw_len):
        block = values[start:start + cw_len]
        if len(block) < cw_len:
            break
        gray = gray_demapping(block)
        codewords = deinterleave_hard(
            gray,
            sf=int(sf),
            is_header=False,
            cr=int(cr),
            ldro=bool(ldro),
        )
        nibbles.extend(hamming_decode_hard(
            codewords,
            is_header=False,
            cr=int(cr),
        ))
    return [int(v) & 0xF for v in nibbles]


def explicit_header_tail_nibbles(
    header_symbol_values: Iterable[int],
    sf: int,
) -> tuple[int, ...]:
    """取出 explicit header block 里跟在 PHY header 后面的 payload nibbles。

    gr-lora_sdr 的 header_decoder 只消费前 5 个 decoded nibbles 作为 PHY
    header 字段；同一个 interleaver block 里剩余 nibbles 会继续作为
    payload/CRC 数据流的开头。比如 SF10 时会贡献 3 个 tail nibbles。
    """
    gray_symbols = gray_demapping(header_symbol_values)
    codewords = deinterleave_hard(
        gray_symbols,
        sf=int(sf),
        is_header=True,
        cr=4,
        ldro=False,
    )
    decoded = hamming_decode_hard(
        codewords,
        is_header=True,
        cr=4,
    )
    return tuple((int(v) & 0xF) for v in decoded[5:])


def nibbles_to_dewhitened_bytes(
    nibbles: Sequence[int],
    payload_len: int,
    has_crc: bool,
) -> tuple[bytes, bytes]:
    """把低/高 nibble 重新组 byte，并对 payload 部分执行 dewhitening。"""
    total_bytes = int(payload_len) + (2 if bool(has_crc) else 0)
    out: list[int] = []
    for offset in range(total_bytes):
        ni = 2 * offset
        if ni + 1 >= len(nibbles):
            break
        low = int(nibbles[ni]) & 0xF
        high = int(nibbles[ni + 1]) & 0xF
        if offset < int(payload_len):
            low ^= WHITENING_SEQ[offset] & 0x0F
            high ^= (WHITENING_SEQ[offset] & 0xF0) >> 4
        out.append(((high & 0xF) << 4) | (low & 0xF))
    payload = bytes(out[: int(payload_len)])
    crc_part = bytes(out[int(payload_len): int(payload_len) + 2])
    return payload, crc_part


def decode_payload_symbols(
    symbol_values: Iterable[int],
    sf: int,
    cr: int,
    ldro: bool,
    payload_len: int,
    has_crc: bool,
    crc_mode: str = "grlora",
) -> PayloadDecodeResult:
    """把 payload symbols 硬解码成 payload bytes，并检查 CRC。

    输入通常来自 argmax/phase-guided 已经选好的 payload symbol_value 序列。
    输出包含 payload_bytes、CRC 字节、重新计算的 CRC 和 crc_valid。
    """
    nibbles = payload_symbols_to_nibbles(symbol_values, sf=sf, cr=cr, ldro=ldro)
    payload, crc_part = nibbles_to_dewhitened_bytes(
        nibbles,
        payload_len=int(payload_len),
        has_crc=bool(has_crc),
    )
    crc_value = 0
    crc_received = 0
    crc_valid = not bool(has_crc)
    if bool(has_crc) and len(crc_part) >= 2 and len(payload) >= int(payload_len):
        if str(crc_mode).lower() == "sx1276":
            crc_value = crc16(payload)
        else:
            crc_value = crc16(payload[: max(0, int(payload_len) - 2)])
            crc_value ^= payload[int(payload_len) - 1]
            crc_value ^= payload[int(payload_len) - 2] << 8
        crc_received = int(crc_part[0]) + (int(crc_part[1]) << 8)
        crc_valid = bool(crc_received == crc_value)
    return PayloadDecodeResult(
        payload_bytes=payload,
        crc_bytes=crc_part,
        crc_computed=int(crc_value),
        crc_received=int(crc_received),
        crc_valid=bool(crc_valid),
        decoded_nibbles=tuple(nibbles),
        codewords=(),
    )


def decode_explicit_frame_symbols(
    header_symbol_values: Iterable[int],
    payload_symbol_values: Iterable[int],
    sf: int,
    bw: float,
    ldro_mode: int,
    crc_mode: str = "grlora",
) -> ExplicitFrameDecodeResult:
    """把 explicit header symbols + payload symbols 硬解码成完整 frame 结果。

    这是 residual search / 离线验证使用的对齐解码入口。它复刻 GNU Radio
    接收链的 block 顺序：

    symbols -> gray -> deinterleave -> hamming -> header_decoder
            -> dewhitening -> crc_verif

    关键细节：header_decoder 会先转发 `decoded_header[5:]`，然后才接上
    payload symbols 解出的 nibbles。早期如果只解 payload symbols，会漏掉
    这些 header tail nibbles，导致 payload byte 全部错位。
    """
    header_symbols = tuple(int(v) for v in header_symbol_values)
    payload_symbols = tuple(int(v) for v in payload_symbol_values)
    header = decode_explicit_header(
        header_symbols,
        sf=int(sf),
        bw=float(bw),
        ldro_mode=int(ldro_mode),
    )
    tail = explicit_header_tail_nibbles(header_symbols, sf=int(sf))
    payload_nibbles = tuple(
        payload_symbols_to_nibbles(
            payload_symbols,
            sf=int(sf),
            cr=int(header.cr),
            ldro=bool(header.ldro),
        )
    )
    frame_nibbles = tuple(tail) + tuple(payload_nibbles)
    payload_bytes, crc_part = nibbles_to_dewhitened_bytes(
        frame_nibbles,
        payload_len=int(header.payload_len),
        has_crc=bool(header.has_crc),
    )

    crc_value = 0
    crc_received = 0
    crc_valid = not bool(header.has_crc)
    if (
        bool(header.has_crc)
        and len(crc_part) >= 2
        and len(payload_bytes) >= int(header.payload_len)
    ):
        if str(crc_mode).lower() == "sx1276":
            crc_value = crc16(payload_bytes)
        else:
            crc_value = crc16(payload_bytes[: max(0, int(header.payload_len) - 2)])
            crc_value ^= payload_bytes[int(header.payload_len) - 1]
            crc_value ^= payload_bytes[int(header.payload_len) - 2] << 8
        crc_received = int(crc_part[0]) + (int(crc_part[1]) << 8)
        crc_valid = bool(crc_received == crc_value)

    payload = PayloadDecodeResult(
        payload_bytes=payload_bytes,
        crc_bytes=crc_part,
        crc_computed=int(crc_value),
        crc_received=int(crc_received),
        crc_valid=bool(crc_valid),
        decoded_nibbles=tuple(frame_nibbles),
        codewords=(),
    )
    return ExplicitFrameDecodeResult(
        header=header,
        payload=payload,
        header_tail_nibbles=tuple(tail),
        payload_symbol_nibbles=tuple(payload_nibbles),
        frame_nibbles=tuple(frame_nibbles),
    )
