"""Sym-FEC-style symbol-level FEC baseline for LoRa payload blocks.

This module is an audit-friendly implementation inspired by the public
description of:

    Weiwei Chen, Xianjin Xia, Shuai Wang, Xianjun Deng, Jiehong Wu, and
    Caishi Huang. "Sym-FEC: Enhancing Error Correction in LoRa PHY With a
    Symbol-Level FEC Decoder." IEEE Transactions on Mobile Computing, 2026.

The public abstract describes Sym-FEC as a symbol-level decoder that connects
the received symbol spectra with coding correlations inside a LoRa coding
block.  The key idea is "signal copy retrieval": evidence from one received
symbol can be converted into evidence for code bits/codewords in the same
interleaver block.

The implementation below follows that observable interface without relying on
private paper text:

* convert every received symbol spectrum into a likelihood over gr-lora_sdr
  hard symbol values;
* use the LoRa diagonal deinterleaver relation to project each symbol's
  likelihood into bit/copy likelihoods for every Hamming codeword in the block;
* choose valid Hamming codewords, then refine them with exact block-wide symbol
  spectrum scores;
* reconstruct payload symbol values/raw FFT bins and let the existing local
  payload codec perform final byte/CRC validation.

It does not use payload templates, counters, cross-packet priors, or CRC-guided
symbol selection.  CRC is only reported after the symbol-level FEC decision.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import numpy as np

from ...chirp import bin_to_grlora_symbol, positive_mod, signed_fft_bin
from ...decoding.header_first_demod import bits_to_int, int_to_bits_msb
from ...decoding.payload_codec import (
    PayloadDecodeResult,
    decode_explicit_frame_symbols,
    decode_payload_symbols,
    encode_hamming_nibble,
    gray_demap_symbols,
)


BitMetric = Literal["max", "logsumexp"]


def _logsumexp(values: Sequence[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("-inf")
    vmax = float(np.max(arr))
    if not math.isfinite(vmax):
        return vmax
    return float(vmax + math.log(float(np.sum(np.exp(arr - vmax)))))


def _reduce_scores(values: Sequence[float], metric: BitMetric | str) -> float:
    if not values:
        return float("-inf")
    mode = str(metric)
    if mode == "logsumexp":
        return _logsumexp(values)
    if mode != "max":
        raise ValueError(f"unknown bit metric: {metric}")
    return float(max(values))


def _as_power(spectrum_or_power: np.ndarray) -> np.ndarray:
    arr = np.asarray(spectrum_or_power)
    if np.iscomplexobj(arr):
        return (np.abs(arr).astype(np.float64) ** 2).astype(np.float64)
    return np.asarray(arr, dtype=np.float64)


def _relative_db_scores(power: np.ndarray, floor_db: float) -> np.ndarray:
    values = np.asarray(power, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("symbol spectrum must be one-dimensional")
    if values.size == 0:
        raise ValueError("empty symbol spectrum")
    db = 10.0 * np.log10(np.maximum(values, 1e-30))
    rel = db - float(np.max(db))
    return np.maximum(rel, -abs(float(floor_db))).astype(np.float64)


def _symbol_alphabet_size(sf: int, is_header: bool, ldro: bool) -> int:
    return 1 << (int(sf) - 2 if bool(is_header) or bool(ldro) else int(sf))


def _canonical_raw_bin(symbol_value: int, sf: int, is_header: bool, ldro: bool) -> int:
    n_bins = 1 << int(sf)
    divisor = 4 if bool(is_header) or bool(ldro) else 1
    return int((int(symbol_value) * divisor + 1) % n_bins)


def _codewords_to_payload_symbol_values(
    codewords: Sequence[int],
    sf: int,
    cr: int,
    ldro: bool,
) -> tuple[int, ...]:
    """Project one payload interleaver block back to demod symbol values."""

    sf_i = int(sf)
    cr_i = int(cr)
    cw_len = cr_i + 4
    sf_app = sf_i - 2 if bool(ldro) else sf_i
    if len(codewords) != sf_app:
        raise ValueError(f"payload block needs {sf_app} codewords, got {len(codewords)}")

    cw_bin = [int_to_bits_msb(int(cw), cw_len) for cw in codewords]
    interleaved: list[int] = []
    for row_idx in range(cw_len):
        row = [False for _ in range(sf_i)]
        for col_idx in range(sf_app):
            row[col_idx] = cw_bin[positive_mod(row_idx - col_idx - 1, sf_app)][row_idx]
        if bool(ldro):
            row[sf_app] = bool(sum(1 for bit in row[:sf_app] if bit) % 2)
        interleaved.append(bits_to_int(row))

    modulator_raw = gray_demap_symbols(interleaved, sf=sf_i)
    divisor = 4 if bool(ldro) else 1
    n_bins = 1 << sf_i
    return tuple(int(((raw_bin - 1) % n_bins) // divisor) for raw_bin in modulator_raw)


@dataclass(frozen=True)
class SymFECConfig:
    """Tunable parameters for the local Sym-FEC-style baseline."""

    score_floor_db: float = 30.0
    bit_metric: BitMetric = "max"
    codeword_candidates: int = 4
    refine_iterations: int = 2
    bit_score_weight: float = 1.0
    exact_symbol_weight: float = 1.0


@dataclass(frozen=True)
class SymFECSymbolEvidence:
    """One received symbol spectrum compressed into demod-symbol likelihoods."""

    symbol_index: int
    raw_scores: np.ndarray
    score_by_value: np.ndarray
    best_raw_bin_by_value: np.ndarray
    argmax_raw_bin: int
    argmax_symbol_value: int
    peak_margin_db: float
    top_raw_bins: tuple[int, ...]
    top_symbol_values: tuple[int, ...]

    def score_symbol(self, symbol_value: int) -> float:
        value = int(symbol_value)
        if 0 <= value < self.score_by_value.size:
            return float(self.score_by_value[value])
        return float("-inf")

    def best_raw_for_symbol(self, symbol_value: int, sf: int, ldro: bool) -> int:
        value = int(symbol_value)
        if 0 <= value < self.best_raw_bin_by_value.size:
            raw_bin = int(self.best_raw_bin_by_value[value])
            if raw_bin >= 0:
                return raw_bin
        return _canonical_raw_bin(value, sf=sf, is_header=False, ldro=ldro)


@dataclass(frozen=True)
class SymFECCodewordCandidate:
    """Candidate valid Hamming codeword for one deinterleaved codeword row."""

    nibble: int
    codeword: int
    bit_score: float


@dataclass(frozen=True)
class SymFECCodewordDecision:
    """Ranked codeword choices for one codeword position."""

    codeword_index: int
    candidates: tuple[SymFECCodewordCandidate, ...]
    selected: SymFECCodewordCandidate
    margin: float


@dataclass(frozen=True)
class SymFECBlockResult:
    """Decoded result for one LoRa payload interleaver block."""

    block_index: int
    symbol_offset: int
    symbol_count: int
    nibbles: tuple[int, ...]
    codewords: tuple[int, ...]
    symbol_values: tuple[int, ...]
    raw_fft_bins: tuple[int, ...]
    argmax_symbol_values: tuple[int, ...]
    argmax_raw_fft_bins: tuple[int, ...]
    changed_symbols: int
    bit_score: float
    exact_symbol_score: float
    total_score: float
    codeword_decisions: tuple[SymFECCodewordDecision, ...]


@dataclass(frozen=True)
class SymFECPayloadResult:
    """Packet payload-level result from the Sym-FEC-style baseline."""

    selected_symbol_values: tuple[int, ...]
    selected_raw_fft_bins: tuple[int, ...]
    argmax_symbol_values: tuple[int, ...]
    argmax_raw_fft_bins: tuple[int, ...]
    blocks: tuple[SymFECBlockResult, ...]
    payload_decode: PayloadDecodeResult | None
    crc_valid: bool
    decode_error: str


def build_symfec_symbol_evidence(
    spectrum_or_power: np.ndarray,
    sf: int,
    symbol_index: int = 0,
    is_header: bool = False,
    ldro: bool = False,
    score_floor_db: float = 30.0,
    top_count: int = 8,
) -> SymFECSymbolEvidence:
    """Convert one raw FFT spectrum into symbol-value likelihoods."""

    power = _as_power(spectrum_or_power)
    n_bins = 1 << int(sf)
    if power.size != n_bins:
        raise ValueError(f"spectrum has {power.size} bins, expected {n_bins}")

    raw_scores = _relative_db_scores(power, floor_db=float(score_floor_db))
    alphabet = _symbol_alphabet_size(sf, is_header=is_header, ldro=ldro)
    score_by_value = np.full(alphabet, float("-inf"), dtype=np.float64)
    best_raw = np.full(alphabet, -1, dtype=np.int64)
    for raw_bin, score in enumerate(raw_scores):
        value = bin_to_grlora_symbol(
            raw_bin,
            sf=int(sf),
            is_header=bool(is_header),
            ldro=bool(ldro),
        )
        if 0 <= value < alphabet and float(score) > float(score_by_value[value]):
            score_by_value[value] = float(score)
            best_raw[value] = int(raw_bin)

    argmax_raw = int(np.argmax(raw_scores))
    argmax_symbol = bin_to_grlora_symbol(
        argmax_raw,
        sf=int(sf),
        is_header=bool(is_header),
        ldro=bool(ldro),
    )
    sorted_bins = tuple(int(v) for v in np.argsort(raw_scores)[::-1][: max(1, int(top_count))])
    sorted_values = tuple(
        int(bin_to_grlora_symbol(v, sf=int(sf), is_header=bool(is_header), ldro=bool(ldro)))
        for v in sorted_bins
    )
    second = float(np.partition(raw_scores, -2)[-2]) if raw_scores.size > 1 else float("-inf")
    return SymFECSymbolEvidence(
        symbol_index=int(symbol_index),
        raw_scores=raw_scores,
        score_by_value=score_by_value,
        best_raw_bin_by_value=best_raw,
        argmax_raw_bin=argmax_raw,
        argmax_symbol_value=int(argmax_symbol),
        peak_margin_db=float(raw_scores[argmax_raw] - second) if math.isfinite(second) else 0.0,
        top_raw_bins=sorted_bins,
        top_symbol_values=sorted_values,
    )


def build_symfec_evidences(
    spectra_or_powers: Sequence[np.ndarray],
    sf: int,
    ldro: bool,
    config: SymFECConfig | None = None,
) -> tuple[SymFECSymbolEvidence, ...]:
    """Build payload symbol evidences for a packet."""

    cfg = config or SymFECConfig()
    return tuple(
        build_symfec_symbol_evidence(
            spectrum_or_power=item,
            sf=int(sf),
            symbol_index=idx,
            is_header=False,
            ldro=bool(ldro),
            score_floor_db=float(cfg.score_floor_db),
        )
        for idx, item in enumerate(spectra_or_powers)
    )


def _row_bit_scores(
    evidence: SymFECSymbolEvidence,
    sf_app: int,
    bit_metric: BitMetric | str,
) -> np.ndarray:
    """Return [row_bit_position, bit] scores from one symbol spectrum."""

    out = np.full((int(sf_app), 2), float("-inf"), dtype=np.float64)
    buckets: list[list[list[float]]] = [[[], []] for _ in range(int(sf_app))]
    for symbol_value, score in enumerate(evidence.score_by_value):
        score_f = float(score)
        if not math.isfinite(score_f):
            continue
        interleaved_value = int(symbol_value) ^ (int(symbol_value) >> 1)
        for bit_pos in range(int(sf_app)):
            bit = (interleaved_value >> (int(sf_app) - 1 - bit_pos)) & 1
            buckets[bit_pos][bit].append(score_f)

    for bit_pos in range(int(sf_app)):
        for bit in (0, 1):
            out[bit_pos, bit] = _reduce_scores(buckets[bit_pos][bit], bit_metric)
        best = float(np.max(out[bit_pos]))
        if math.isfinite(best):
            out[bit_pos] -= best
    return out


def _block_bit_scores(
    evidences: Sequence[SymFECSymbolEvidence],
    sf: int,
    cr: int,
    ldro: bool,
    config: SymFECConfig,
) -> np.ndarray:
    """Return [codeword_index, codeword_bit_index, bit] copy scores."""

    cw_len = int(cr) + 4
    sf_app = int(sf) - 2 if bool(ldro) else int(sf)
    if len(evidences) != cw_len:
        raise ValueError(f"block needs {cw_len} symbol evidences, got {len(evidences)}")

    row_scores = np.stack(
        [
            _row_bit_scores(evidence, sf_app=sf_app, bit_metric=config.bit_metric)
            for evidence in evidences
        ],
        axis=0,
    )
    out = np.full((sf_app, cw_len, 2), float("-inf"), dtype=np.float64)
    for codeword_idx in range(sf_app):
        for bit_idx in range(cw_len):
            row_bit_pos = positive_mod(bit_idx - codeword_idx - 1, sf_app)
            out[codeword_idx, bit_idx, :] = row_scores[bit_idx, row_bit_pos, :]
    return out


def _enumerate_codeword_decisions(
    copy_scores: np.ndarray,
    cr: int,
    config: SymFECConfig,
) -> tuple[SymFECCodewordDecision, ...]:
    cw_len = int(cr) + 4
    keep = max(1, int(config.codeword_candidates))
    decisions: list[SymFECCodewordDecision] = []
    for codeword_idx in range(copy_scores.shape[0]):
        candidates: list[SymFECCodewordCandidate] = []
        for nibble in range(16):
            codeword = encode_hamming_nibble(nibble, cr_app=int(cr))
            bits = int_to_bits_msb(codeword, cw_len)
            score = 0.0
            for bit_idx, bit in enumerate(bits):
                score += float(copy_scores[codeword_idx, bit_idx, int(bit)])
            candidates.append(
                SymFECCodewordCandidate(
                    nibble=int(nibble),
                    codeword=int(codeword),
                    bit_score=float(score),
                )
            )
        candidates.sort(key=lambda item: item.bit_score, reverse=True)
        selected = candidates[0]
        second_score = candidates[1].bit_score if len(candidates) > 1 else float("-inf")
        decisions.append(
            SymFECCodewordDecision(
                codeword_index=int(codeword_idx),
                candidates=tuple(candidates[:keep]),
                selected=selected,
                margin=float(selected.bit_score - second_score)
                if math.isfinite(second_score)
                else 0.0,
            )
        )
    return tuple(decisions)


def _exact_block_symbol_score(
    evidences: Sequence[SymFECSymbolEvidence],
    symbol_values: Sequence[int],
) -> float:
    score = 0.0
    for evidence, symbol_value in zip(evidences, symbol_values):
        score += evidence.score_symbol(int(symbol_value))
    return float(score)


def _decision_bit_score(decisions: Sequence[SymFECCodewordDecision]) -> float:
    return float(sum(float(item.selected.bit_score) for item in decisions))


def _retarget_decision(
    decision: SymFECCodewordDecision,
    candidate: SymFECCodewordCandidate,
) -> SymFECCodewordDecision:
    return SymFECCodewordDecision(
        codeword_index=int(decision.codeword_index),
        candidates=decision.candidates,
        selected=candidate,
        margin=float(decision.margin),
    )


def _combined_block_score(
    evidences: Sequence[SymFECSymbolEvidence],
    decisions: Sequence[SymFECCodewordDecision],
    sf: int,
    cr: int,
    ldro: bool,
    config: SymFECConfig,
) -> tuple[float, float, float, tuple[int, ...]]:
    codewords = tuple(int(item.selected.codeword) for item in decisions)
    symbol_values = _codewords_to_payload_symbol_values(
        codewords,
        sf=int(sf),
        cr=int(cr),
        ldro=bool(ldro),
    )
    bit_score = _decision_bit_score(decisions)
    exact_score = _exact_block_symbol_score(evidences, symbol_values)
    total = (
        float(config.bit_score_weight) * bit_score
        + float(config.exact_symbol_weight) * exact_score
    )
    return float(total), float(bit_score), float(exact_score), tuple(symbol_values)


def _coordinate_refine_block(
    evidences: Sequence[SymFECSymbolEvidence],
    decisions: tuple[SymFECCodewordDecision, ...],
    sf: int,
    cr: int,
    ldro: bool,
    config: SymFECConfig,
) -> tuple[SymFECCodewordDecision, ...]:
    current = list(decisions)
    best_total, _bit, _exact, _symbols = _combined_block_score(
        evidences, current, sf=sf, cr=cr, ldro=ldro, config=config
    )
    for _ in range(max(0, int(config.refine_iterations))):
        improved = False
        for idx, decision in enumerate(tuple(current)):
            local_best = current[idx]
            local_best_total = best_total
            for candidate in decision.candidates:
                trial = list(current)
                trial[idx] = _retarget_decision(decision, candidate)
                total, _bit, _exact, _symbols = _combined_block_score(
                    evidences, trial, sf=sf, cr=cr, ldro=ldro, config=config
                )
                if total > local_best_total + 1e-9:
                    local_best_total = total
                    local_best = trial[idx]
            if local_best is not current[idx]:
                current[idx] = local_best
                best_total = local_best_total
                improved = True
        if not improved:
            break
    return tuple(current)


def decode_symfec_block(
    evidences: Sequence[SymFECSymbolEvidence],
    sf: int,
    cr: int,
    ldro: bool,
    block_index: int = 0,
    symbol_offset: int = 0,
    config: SymFECConfig | None = None,
) -> SymFECBlockResult:
    """Decode one payload interleaver block using symbol-level FEC evidence."""

    cfg = config or SymFECConfig()
    cw_len = int(cr) + 4
    if len(evidences) != cw_len:
        raise ValueError(f"Sym-FEC block needs {cw_len} symbols, got {len(evidences)}")

    copy_scores = _block_bit_scores(
        evidences,
        sf=int(sf),
        cr=int(cr),
        ldro=bool(ldro),
        config=cfg,
    )
    decisions = _enumerate_codeword_decisions(copy_scores, cr=int(cr), config=cfg)
    decisions = _coordinate_refine_block(
        evidences,
        decisions,
        sf=int(sf),
        cr=int(cr),
        ldro=bool(ldro),
        config=cfg,
    )
    total_score, bit_score, exact_score, symbol_values = _combined_block_score(
        evidences,
        decisions,
        sf=int(sf),
        cr=int(cr),
        ldro=bool(ldro),
        config=cfg,
    )
    raw_bins = tuple(
        int(evidence.best_raw_for_symbol(value, sf=int(sf), ldro=bool(ldro)))
        for evidence, value in zip(evidences, symbol_values)
    )
    argmax_symbols = tuple(int(evidence.argmax_symbol_value) for evidence in evidences)
    argmax_bins = tuple(int(evidence.argmax_raw_bin) for evidence in evidences)
    return SymFECBlockResult(
        block_index=int(block_index),
        symbol_offset=int(symbol_offset),
        symbol_count=int(cw_len),
        nibbles=tuple(int(item.selected.nibble) for item in decisions),
        codewords=tuple(int(item.selected.codeword) for item in decisions),
        symbol_values=tuple(int(v) for v in symbol_values),
        raw_fft_bins=raw_bins,
        argmax_symbol_values=argmax_symbols,
        argmax_raw_fft_bins=argmax_bins,
        changed_symbols=int(sum(int(a != b) for a, b in zip(symbol_values, argmax_symbols))),
        bit_score=float(bit_score),
        exact_symbol_score=float(exact_score),
        total_score=float(total_score),
        codeword_decisions=decisions,
    )


def decode_symfec_payload_from_evidences(
    evidences: Sequence[SymFECSymbolEvidence],
    sf: int,
    cr: int,
    ldro: bool,
    config: SymFECConfig | None = None,
    header_symbol_values: Sequence[int] | None = None,
    bw: float = 125000.0,
    ldro_mode: int = 2,
    payload_len: int | None = None,
    has_crc: bool | None = None,
    crc_mode: str = "grlora",
) -> SymFECPayloadResult:
    """Decode a packet payload from precomputed symbol evidences."""

    cfg = config or SymFECConfig()
    cw_len = int(cr) + 4
    blocks: list[SymFECBlockResult] = []
    selected_symbols: list[int] = []
    selected_bins: list[int] = []
    argmax_symbols: list[int] = []
    argmax_bins: list[int] = []

    for offset in range(0, len(evidences), cw_len):
        block_evidences = list(evidences[offset:offset + cw_len])
        if len(block_evidences) < cw_len:
            for evidence in block_evidences:
                selected_symbols.append(int(evidence.argmax_symbol_value))
                selected_bins.append(int(evidence.argmax_raw_bin))
                argmax_symbols.append(int(evidence.argmax_symbol_value))
                argmax_bins.append(int(evidence.argmax_raw_bin))
            break
        block = decode_symfec_block(
            block_evidences,
            sf=int(sf),
            cr=int(cr),
            ldro=bool(ldro),
            block_index=len(blocks),
            symbol_offset=offset,
            config=cfg,
        )
        blocks.append(block)
        selected_symbols.extend(block.symbol_values)
        selected_bins.extend(block.raw_fft_bins)
        argmax_symbols.extend(block.argmax_symbol_values)
        argmax_bins.extend(block.argmax_raw_fft_bins)

    payload_decode: PayloadDecodeResult | None = None
    crc_valid = False
    decode_error = ""
    try:
        if header_symbol_values is not None:
            decoded = decode_explicit_frame_symbols(
                header_symbol_values=tuple(int(v) for v in header_symbol_values),
                payload_symbol_values=tuple(selected_symbols),
                sf=int(sf),
                bw=float(bw),
                ldro_mode=int(ldro_mode),
                crc_mode=str(crc_mode),
            )
            payload_decode = decoded.payload
            crc_valid = bool(decoded.payload.crc_valid)
        elif payload_len is not None and has_crc is not None:
            payload_decode = decode_payload_symbols(
                selected_symbols,
                sf=int(sf),
                cr=int(cr),
                ldro=bool(ldro),
                payload_len=int(payload_len),
                has_crc=bool(has_crc),
                crc_mode=str(crc_mode),
            )
            crc_valid = bool(payload_decode.crc_valid)
    except Exception as exc:  # Keep baseline evaluation running packet by packet.
        decode_error = f"{type(exc).__name__}: {exc}"

    return SymFECPayloadResult(
        selected_symbol_values=tuple(int(v) for v in selected_symbols),
        selected_raw_fft_bins=tuple(int(v) for v in selected_bins),
        argmax_symbol_values=tuple(int(v) for v in argmax_symbols),
        argmax_raw_fft_bins=tuple(int(v) for v in argmax_bins),
        blocks=tuple(blocks),
        payload_decode=payload_decode,
        crc_valid=bool(crc_valid),
        decode_error=decode_error,
    )


def decode_symfec_payload_from_spectra(
    spectra_or_powers: Sequence[np.ndarray],
    sf: int,
    cr: int,
    ldro: bool,
    config: SymFECConfig | None = None,
    header_symbol_values: Sequence[int] | None = None,
    bw: float = 125000.0,
    ldro_mode: int = 2,
    payload_len: int | None = None,
    has_crc: bool | None = None,
    crc_mode: str = "grlora",
) -> SymFECPayloadResult:
    """Convenience API: spectra/powers in, selected payload symbols out."""

    cfg = config or SymFECConfig()
    evidences = build_symfec_evidences(
        spectra_or_powers=spectra_or_powers,
        sf=int(sf),
        ldro=bool(ldro),
        config=cfg,
    )
    return decode_symfec_payload_from_evidences(
        evidences=evidences,
        sf=int(sf),
        cr=int(cr),
        ldro=bool(ldro),
        config=cfg,
        header_symbol_values=header_symbol_values,
        bw=float(bw),
        ldro_mode=int(ldro_mode),
        payload_len=payload_len,
        has_crc=has_crc,
        crc_mode=str(crc_mode),
    )


def symfec_symbol_rows(
    result: SymFECPayloadResult,
    sf: int,
    ldro: bool,
) -> list[dict[str, int | float]]:
    """Flatten a payload result into per-symbol diagnostics."""

    rows: list[dict[str, int | float]] = []
    for block in result.blocks:
        for local_idx, raw_bin in enumerate(block.raw_fft_bins):
            symbol_idx = int(block.symbol_offset + local_idx)
            selected_symbol = int(block.symbol_values[local_idx])
            argmax_bin = int(block.argmax_raw_fft_bins[local_idx])
            argmax_symbol = int(block.argmax_symbol_values[local_idx])
            rows.append(
                {
                    "payload_symbol_index": symbol_idx,
                    "block_index": int(block.block_index),
                    "block_local_symbol_index": int(local_idx),
                    "selected_raw_fft_bin": int(raw_bin),
                    "selected_signed_fft_bin": signed_fft_bin(raw_bin, 1 << int(sf)),
                    "selected_symbol_value": selected_symbol,
                    "argmax_raw_fft_bin": argmax_bin,
                    "argmax_signed_fft_bin": signed_fft_bin(argmax_bin, 1 << int(sf)),
                    "argmax_symbol_value": argmax_symbol,
                    "changed_from_argmax": int(selected_symbol != argmax_symbol),
                    "block_total_score": float(block.total_score),
                    "block_bit_score": float(block.bit_score),
                    "block_exact_symbol_score": float(block.exact_symbol_score),
                }
            )
    return rows


__all__ = [
    "BitMetric",
    "SymFECBlockResult",
    "SymFECCodewordCandidate",
    "SymFECCodewordDecision",
    "SymFECConfig",
    "SymFECPayloadResult",
    "SymFECSymbolEvidence",
    "build_symfec_evidences",
    "build_symfec_symbol_evidence",
    "decode_symfec_block",
    "decode_symfec_payload_from_evidences",
    "decode_symfec_payload_from_spectra",
    "symfec_symbol_rows",
]
