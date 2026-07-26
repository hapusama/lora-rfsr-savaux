"""Sym-FEC-style symbol-level FEC baseline."""

from .paper_symfec_decoder import (
    BitMetric,
    SymFECBlockResult,
    SymFECCodewordCandidate,
    SymFECCodewordDecision,
    SymFECConfig,
    SymFECPayloadResult,
    SymFECSymbolEvidence,
    build_symfec_evidences,
    build_symfec_symbol_evidence,
    decode_symfec_block,
    decode_symfec_payload_from_evidences,
    decode_symfec_payload_from_spectra,
    symfec_symbol_rows,
)

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
