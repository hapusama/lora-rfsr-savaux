"""LoRaTrimmer paper-style weak-signal demodulation baseline."""

from .paper_loratrimmer_demod import (
    CfoCorrectionMode,
    LoRaTrimmerDemodResult,
    LoRaTrimmerMatrices,
    build_loratrimmer_matrices,
    demod_loratrimmer_symbol,
    loratrimmer_metric,
    loratrimmer_metric_from_symbol,
)

__all__ = [
    "CfoCorrectionMode",
    "LoRaTrimmerDemodResult",
    "LoRaTrimmerMatrices",
    "build_loratrimmer_matrices",
    "demod_loratrimmer_symbol",
    "loratrimmer_metric",
    "loratrimmer_metric_from_symbol",
]
