"""Optional RF Super Resolution frontend for synchronized LoRa IQ."""

from .frontend import (
    DEFAULT_OTA_CHECKPOINT,
    DEFAULT_SYNTHETIC_CHECKPOINT,
    RFSRFrontendConfig,
    RFSRProvenance,
    RFSuperResolutionFrontend,
    default_rfsr_repo_root,
)
from .reference_phy import (
    EncodedReference,
    ReferencePhyConfig,
    UartPacketRecord,
    UartReferenceLog,
    encode_reference_phy,
    parse_uart_reference_log,
    phy_config_from_uart,
    write_reference_packet,
)

__all__ = [
    "DEFAULT_OTA_CHECKPOINT",
    "DEFAULT_SYNTHETIC_CHECKPOINT",
    "RFSRFrontendConfig",
    "RFSRProvenance",
    "RFSuperResolutionFrontend",
    "EncodedReference",
    "ReferencePhyConfig",
    "UartPacketRecord",
    "UartReferenceLog",
    "default_rfsr_repo_root",
    "encode_reference_phy",
    "parse_uart_reference_log",
    "phy_config_from_uart",
    "write_reference_packet",
]
