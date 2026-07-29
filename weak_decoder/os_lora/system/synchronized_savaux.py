"""使用已有 FrameSync 结果驱动 Savaux 的头部优先解调。

本模块只负责“同步之后”的符号判决，不执行包检测，也不读取 reference metadata。
调用方必须先完成 packet detection、帧定位以及 CFO/STO/SFO 估计，再把
``GrloraFrameSyncResult`` 和同一条 IQ 交给 :func:`demod_synchronized_savaux`。

解调顺序与 LoRa 显式头部接收机一致：先判决 8 个 PHY header symbols，解析出
payload 长度、编码率、CRC 和 LDRO，再按解析结果继续判决 payload symbols。
符号游标使用已有 ``advance_symbol_cursor`` 逐符号补偿累计 SFO。

当前主链只保留论文 Savaux 的 branch 相干合并和功率最大判决，不接入 GLS、
CRC 引导或其他候选重排，便于把 RFSR 的影响与解调器本身分开审查。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ...baselines.savaux_oversampled.paper_oversampled_demod import (
    paper_oversampled_spectrum,
)
from ...chirp import bin_to_grlora_symbol
from ...decoding.header_first_demod import (
    HeaderDecodeResult,
    advance_symbol_cursor,
    decode_explicit_header,
)
from ...synchronization.grlora_frame_sync import GrloraFrameSyncResult


@dataclass(frozen=True)
class SynchronizedSavauxSymbol:
    """一个同步后 LoRa symbol 的 Savaux 硬判决及其定位诊断。"""

    # stage 为 header 或 payload；两个 index 分别对应整帧和当前阶段内的位置。
    stage: str
    frame_symbol_index: int
    stage_symbol_index: int
    # start_sample 是 FrameSync/SFO 游标给出的区间起点；实际 Savaux 观察还会
    # 在 _demod_symbol 中加 R/2 的中心采样原点偏移。
    start_sample: int
    selected_bin: int
    symbol_value: int
    # 第一名与第二名候选功率比，只用于诊断，不参与门控或重判。
    peak_margin_db: float
    # 记录 SFO 累积量以及本 symbol 之后是否增减一个采样点。
    sfo_cum_before: float
    sfo_sample_adjust_after: int


@dataclass(frozen=True)
class SynchronizedSavauxResult:
    """头部优先 Savaux 解调的包级结果。"""

    # status 区分同步无效、头部/载荷截断、头部校验失败和完整成功。
    status: str
    synchronized: bool
    header_valid: bool
    decoded_payload_length: int
    decoded_cr: int
    decoded_has_crc: bool
    decoded_ldro: bool
    decoded_payload_symbol_count: int
    symbols: tuple[SynchronizedSavauxSymbol, ...]
    error: str | None = None

    @property
    def header_symbols(self) -> tuple[SynchronizedSavauxSymbol, ...]:
        return tuple(item for item in self.symbols if item.stage == "header")

    @property
    def payload_symbols(self) -> tuple[SynchronizedSavauxSymbol, ...]:
        return tuple(item for item in self.symbols if item.stage == "payload")


def _empty_result(
    status: str,
    *,
    synchronized: bool,
    symbols: tuple[SynchronizedSavauxSymbol, ...] = (),
    error: str | None = None,
) -> SynchronizedSavauxResult:
    """构造尚未得到有效 PHY header 时的统一失败结果。"""

    return SynchronizedSavauxResult(
        status=status,
        synchronized=synchronized,
        header_valid=False,
        decoded_payload_length=0,
        decoded_cr=0,
        decoded_has_crc=False,
        decoded_ldro=False,
        decoded_payload_symbol_count=0,
        symbols=symbols,
        error=error,
    )


def _demod_symbol(
    samples: np.ndarray,
    *,
    start_sample: int,
    header_start_sample: int,
    sf: int,
    os_factor: int,
    cfo_int: int,
    cfo_frac: float,
    stage: str,
    frame_symbol_index: int,
    stage_symbol_index: int,
    payload_ldro: bool,
    sfo_cum_before: float,
    sfo_sample_adjust_after: int,
) -> SynchronizedSavauxSymbol:
    """按同步参数截取一个 symbol，并执行论文 Savaux 硬判决。"""

    # gr-lora_sdr 的同步起点位于采样区间边界；Savaux branch 观测与普通
    # header-first FFT 一样，从一个 ADC 半相位（R/2 samples）处取中心原点。
    origin_shift = int(os_factor) // 2
    combined, _branches, _ = paper_oversampled_spectrum(
        samples=samples,
        start_sample=int(start_sample) + origin_shift,
        sf=int(sf),
        os_factor=int(os_factor),
        cfo_int=int(cfo_int),
        cfo_frac=float(cfo_frac),
        header_start_sample=int(header_start_sample) + origin_shift,
        cfo_correction_mode="continuous",
    )
    # Savaux Eq.37 已把全部 polyphase branch 做相位对齐和相干合并；
    # 接收机在合并频谱上直接选择功率最大的候选 bin。
    scores = np.abs(combined).astype(np.float64) ** 2
    selected_bin = int(np.argmax(scores))
    selected_power = float(scores[selected_bin])
    second_power = (
        float(np.partition(scores, -2)[-2]) if scores.size > 1 else 0.0
    )
    is_header = str(stage) == "header"
    return SynchronizedSavauxSymbol(
        stage=str(stage),
        frame_symbol_index=int(frame_symbol_index),
        stage_symbol_index=int(stage_symbol_index),
        start_sample=int(start_sample),
        selected_bin=selected_bin,
        symbol_value=bin_to_grlora_symbol(
            selected_bin,
            sf=int(sf),
            is_header=is_header,
            ldro=bool(payload_ldro),
        ),
        peak_margin_db=float(
            10.0
            * math.log10(
                (selected_power + 1e-30) / (second_power + 1e-30)
            )
        ),
        sfo_cum_before=float(sfo_cum_before),
        sfo_sample_adjust_after=int(sfo_sample_adjust_after),
    )


def _advance(
    cursor: int,
    sfo_cum: float,
    frame_sync: GrloraFrameSyncResult,
    sf: int,
    os_factor: int,
) -> tuple[int, float, int]:
    """复用 header-first 解调器的逐符号 SFO 游标推进规则。"""

    return advance_symbol_cursor(
        cursor,
        sf=int(sf),
        os_factor=int(os_factor),
        sfo_cum=float(sfo_cum),
        sfo_hat=float(frame_sync.sfo_hat),
    )


def demod_synchronized_savaux(
    samples: np.ndarray,
    frame_sync: GrloraFrameSyncResult,
    *,
    sf: int,
    bw_hz: float,
    os_factor: int,
    ldro_mode: int = 1,
    max_payload_symbols: int = 512,
) -> SynchronizedSavauxResult:
    """不使用参考边界，按 FrameSync 估计完成 explicit header 和 payload 解调。"""

    if not frame_sync.valid:
        return _empty_result("sync_invalid", synchronized=False)
    if int(os_factor) <= 0:
        raise ValueError("os_factor must be positive")
    values = np.asarray(samples, dtype=np.complex64)
    # fine_payload_start_sample 沿用 gr-lora_sdr 历史命名；显式头部模式下它实际是
    # data region 起点，也就是第 0 个 PHY header symbol 的起点。
    header_start = int(frame_sync.fine_payload_start_sample)
    cursor = header_start
    sfo_cum = float(frame_sync.sfo_cum_initial)
    symbols: list[SynchronizedSavauxSymbol] = []

    # explicit header 固定占 8 个 symbols，并始终采用 reduced-rate 映射。
    try:
        for index in range(8):
            next_cursor, next_sfo, sample_adjust = _advance(
                cursor, sfo_cum, frame_sync, sf, os_factor
            )
            symbols.append(
                _demod_symbol(
                    values,
                    start_sample=cursor,
                    header_start_sample=header_start,
                    sf=sf,
                    os_factor=os_factor,
                    cfo_int=frame_sync.cfo_int_est,
                    cfo_frac=frame_sync.cfo_frac_est,
                    stage="header",
                    frame_symbol_index=index,
                    stage_symbol_index=index,
                    payload_ldro=False,
                    sfo_cum_before=sfo_cum,
                    sfo_sample_adjust_after=sample_adjust,
                )
            )
            cursor, sfo_cum = next_cursor, next_sfo
    except ValueError as exc:
        return _empty_result(
            "truncated_header",
            synchronized=True,
            symbols=tuple(symbols),
            error=str(exc),
        )

    # 只依据刚得到的 8 个硬判决解析头部，reference metadata 不在本模块中出现。
    header: HeaderDecodeResult = decode_explicit_header(
        [item.symbol_value for item in symbols],
        sf=int(sf),
        bw=float(bw_hz),
        ldro_mode=int(ldro_mode),
    )
    if not header.header_valid:
        return _empty_result(
            "header_invalid",
            synchronized=True,
            symbols=tuple(symbols),
        )
    if not 0 <= int(header.payload_symbol_count) <= int(max_payload_symbols):
        return _empty_result(
            "payload_count_invalid",
            synchronized=True,
            symbols=tuple(symbols),
            error=f"decoded payload symbol count is {header.payload_symbol_count}",
        )

    # payload 数量和 LDRO 都来自已解出的头部；随后沿用同一 CFO 与 SFO 状态。
    try:
        for payload_index in range(int(header.payload_symbol_count)):
            frame_index = 8 + payload_index
            next_cursor, next_sfo, sample_adjust = _advance(
                cursor, sfo_cum, frame_sync, sf, os_factor
            )
            symbols.append(
                _demod_symbol(
                    values,
                    start_sample=cursor,
                    header_start_sample=header_start,
                    sf=sf,
                    os_factor=os_factor,
                    cfo_int=frame_sync.cfo_int_est,
                    cfo_frac=frame_sync.cfo_frac_est,
                    stage="payload",
                    frame_symbol_index=frame_index,
                    stage_symbol_index=payload_index,
                    payload_ldro=bool(header.ldro),
                    sfo_cum_before=sfo_cum,
                    sfo_sample_adjust_after=sample_adjust,
                )
            )
            cursor, sfo_cum = next_cursor, next_sfo
    except ValueError as exc:
        return SynchronizedSavauxResult(
            status="truncated_payload",
            synchronized=True,
            header_valid=True,
            decoded_payload_length=int(header.payload_len),
            decoded_cr=int(header.cr),
            decoded_has_crc=bool(header.has_crc),
            decoded_ldro=bool(header.ldro),
            decoded_payload_symbol_count=int(header.payload_symbol_count),
            symbols=tuple(symbols),
            error=str(exc),
        )

    return SynchronizedSavauxResult(
        status="ok",
        synchronized=True,
        header_valid=True,
        decoded_payload_length=int(header.payload_len),
        decoded_cr=int(header.cr),
        decoded_has_crc=bool(header.has_crc),
        decoded_ldro=bool(header.ldro),
        decoded_payload_symbol_count=int(header.payload_symbol_count),
        symbols=tuple(symbols),
    )
