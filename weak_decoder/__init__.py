"""LoRa 弱包检测、帧同步与 OS-LoRa/GLS 解调基础模块。"""

from .chirp import build_upchirp, dechirp_fft, signed_fft_bin
from .synchronization import (
    DetectionEvent,
    FrameLocation,
    FrameLocatorConfig,
    FrameSyncPeak,
    GrloraBranchSyncEstimate,
    GrloraFrameSyncResult,
    PreambleDetectorConfig,
    SymbolPeak,
    WindowPeak,
    XCopyAlignment,
    XCopyConfig,
    XCopyDetection,
    XCopyDetectionBin,
    XCopyPacketDetection,
    XCopySoftFrameCandidate,
    XCopySyncResult,
    build_grlora_corrected_preamble_chirps,
    detect_preamble_runs,
    locate_frame_from_event,
    locate_xcopy_soft_frame_candidates,
    run_xcopy_paper_sync,
    run_grlora_frame_sync_validation,
    run_xcopy_sync,
    scan_preamble_windows,
    scan_periodic_preamble,
    scan_xcopy_packet_preambles,
    sync_word_to_symbols,
    xcopy_raw_symbol_rows,
)


def __getattr__(name: str):
    """按需导出采集 profile，避免 ``python -m branch4_profile`` 重复加载。"""

    if name in {"BRANCH4_PROFILE", "LoraCaptureProfile"}:
        from .branch4_profile import BRANCH4_PROFILE, LoraCaptureProfile

        globals().update(
            BRANCH4_PROFILE=BRANCH4_PROFILE,
            LoraCaptureProfile=LoraCaptureProfile,
        )
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "DetectionEvent",
    "BRANCH4_PROFILE",
    "FrameLocation",
    "FrameLocatorConfig",
    "FrameSyncPeak",
    "GrloraBranchSyncEstimate",
    "GrloraFrameSyncResult",
    "LoraCaptureProfile",
    "PreambleDetectorConfig",
    "SymbolPeak",
    "WindowPeak",
    "XCopyAlignment",
    "XCopyConfig",
    "XCopyDetection",
    "XCopyDetectionBin",
    "XCopyPacketDetection",
    "XCopySoftFrameCandidate",
    "XCopySyncResult",
    "build_upchirp",
    "build_grlora_corrected_preamble_chirps",
    "dechirp_fft",
    "detect_preamble_runs",
    "locate_frame_from_event",
    "locate_xcopy_soft_frame_candidates",
    "run_xcopy_paper_sync",
    "run_grlora_frame_sync_validation",
    "run_xcopy_sync",
    "scan_preamble_windows",
    "scan_periodic_preamble",
    "scan_xcopy_packet_preambles",
    "signed_fft_bin",
    "sync_word_to_symbols",
    "xcopy_raw_symbol_rows",
]
