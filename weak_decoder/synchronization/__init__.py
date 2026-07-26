"""Preamble detection, frame location, and gr-lora-compatible synchronization."""

from .frame_locator import (
    FrameLocation,
    FrameLocatorConfig,
    SymbolPeak,
    locate_frame_from_event,
    sync_word_to_symbols,
)
from .grlora_frame_sync import (
    FrameSyncPeak,
    GrloraBranchSyncEstimate,
    GrloraFrameSyncResult,
    build_grlora_corrected_preamble_chirps,
    run_grlora_frame_sync_validation,
)
from .preamble_detector import (
    DetectionEvent,
    PreambleDetectorConfig,
    WindowPeak,
    detect_preamble_runs,
    scan_preamble_windows,
)
from .xcopy_sync import (
    XCopyAlignment,
    XCopyConfig,
    XCopyDetection,
    XCopyDetectionBin,
    XCopyPacketDetection,
    XCopySoftFrameCandidate,
    XCopySyncResult,
    locate_xcopy_soft_frame_candidates,
    run_xcopy_paper_sync,
    run_xcopy_sync,
    scan_xcopy_packet_preambles,
    scan_periodic_preamble,
    xcopy_raw_symbol_rows,
)

__all__ = [
    "DetectionEvent",
    "FrameLocation",
    "FrameLocatorConfig",
    "FrameSyncPeak",
    "GrloraBranchSyncEstimate",
    "GrloraFrameSyncResult",
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
    "build_grlora_corrected_preamble_chirps",
    "detect_preamble_runs",
    "locate_frame_from_event",
    "locate_xcopy_soft_frame_candidates",
    "run_xcopy_paper_sync",
    "run_grlora_frame_sync_validation",
    "scan_preamble_windows",
    "run_xcopy_sync",
    "scan_periodic_preamble",
    "scan_xcopy_packet_preambles",
    "sync_word_to_symbols",
    "xcopy_raw_symbol_rows",
]
