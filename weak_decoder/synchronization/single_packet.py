"""对一条单包 LoRa IQ 执行可复用的检测、帧定位和精同步。

输入通常是已经从连续采集文件中粗裁剪出的单包 IQ，但本模块不会读取裁剪时保存的
包起点、CFO、STO 或 SFO 真值。它从 IQ 自身重新执行以下步骤：

1. 使用周期性前导码检测器寻找候选事件；
2. 在检测粗起点附近用多条 upchirp 的非相干累积能量细化 chirp 边界；
3. 使用 sync word 和 SFD 结构先粗后细地定位帧；
4. 调用已有 gr-lora_sdr 风格 FrameSync，估计整数/小数 CFO、分数 STO 和 SFO；
5. 多个候选事件同时存在时，优先选择完整同步有效且定位得分最高的事件。

这里的职责是把现有同步组件组合成一个可直接接收内存 IQ 的 API。核心检测器、
frame locator 和 FrameSync 算法仍分别位于原有模块中，没有在此重复实现。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from ..chirp import build_upchirp, signed_fft_bin
from .frame_locator import FrameLocation, FrameLocatorConfig, locate_frame_from_event
from .grlora_frame_sync import GrloraFrameSyncResult, run_grlora_frame_sync_validation
from .preamble_detector import (
    DetectionEvent,
    PreambleDetectorConfig,
    detect_preamble_runs,
)


@dataclass(frozen=True)
class SinglePacketSyncConfig:
    """单包同步所需的物理参数和有界搜索参数。"""

    # LoRa 物理层参数。sample_rate_hz / bw_hz 必须是正整数 OSR。
    sf: int
    bw_hz: float
    sample_rate_hz: float
    center_frequency_hz: float
    preamble_symbols: int
    sync_word: int
    # detection_chirps 控制每个检测窗累积几条 chirp；scan_chirps 限制最多
    # 扫描输入开头多少条 chirp，避免单包测试误扫到后续数据。
    detection_chirps: int = 4
    scan_chirps: int = 32
    # 三类 bin 容差分别约束前导码周期性、sync word 和 SFD 匹配。
    bin_tolerance: int = 2
    sync_bin_tolerance: int = 4
    sfd_bin_tolerance: int = 4
    # FrameSync 最终验证时对校正后 bin0 的允许偏差。
    frame_sync_bin0_tolerance: int = 0

    @property
    def os_factor(self) -> int:
        """根据采样率与带宽计算整数过采样倍数。"""

        ratio = float(self.sample_rate_hz) / float(self.bw_hz)
        value = int(round(ratio))
        if value <= 0 or not math.isclose(ratio, value, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                "sample_rate_hz / bw_hz must be a positive integer, "
                f"got {ratio:.9g}"
            )
        return value

    @property
    def chirp_samples(self) -> int:
        """返回当前 SF/OSR 下一条完整 LoRa chirp 的采样点数。"""

        return (1 << int(self.sf)) * self.os_factor

    def validate(self) -> None:
        """在进入大规模 FFT 搜索前拒绝不一致的物理参数。"""

        if not 5 <= int(self.sf) <= 12:
            raise ValueError("sf must be in [5, 12]")
        if float(self.center_frequency_hz) <= 0.0:
            raise ValueError("center_frequency_hz must be positive")
        if int(self.preamble_symbols) <= 0:
            raise ValueError("preamble_symbols must be positive")
        if not 1 <= int(self.detection_chirps) <= int(self.preamble_symbols):
            raise ValueError("detection_chirps must be within the preamble")
        if int(self.scan_chirps) < int(self.preamble_symbols) + 4:
            raise ValueError("scan_chirps must cover the preamble, sync word, and SFD")
        _ = self.os_factor


@dataclass(frozen=True)
class SinglePacketSyncResult:
    """检测诊断、帧定位和最佳独立 FrameSync 估计。

    ``status`` 可能为 ``ok``、``no_preamble``、``sync_error`` 或
    ``sync_invalid``。只有 :attr:`synchronized` 为真时，下游才可以使用
    ``frame_sync`` 提供的符号起点和 CFO/STO/SFO 参数。
    """

    status: str
    event_count: int
    selected_event_index: int | None
    alignment: dict[str, float | int] | None
    frame_location: FrameLocation | None
    frame_sync: GrloraFrameSyncResult | None
    error: str | None = None

    @property
    def synchronized(self) -> bool:
        return bool(
            self.status == "ok"
            and self.frame_location is not None
            and self.frame_location.valid
            and self.frame_sync is not None
            and self.frame_sync.valid
        )


def align_event_start(
    samples: np.ndarray,
    event: DetectionEvent,
    config: PreambleDetectorConfig,
    search_radius_samples: int,
    step_samples: int,
    align_chirps: int,
) -> dict[str, float | int]:
    """在检测事件附近搜索多条前导 chirp 共同支持的稳定采样边界。

    每个候选起点都截取 ``align_chirps`` 条完整 chirp，分别 dechirp 后沿 chirp
    维度做非相干能量累加。返回总峰值功率最大的候选，同时保留次峰、峰值占比和
    置信度，供输出诊断；该过程不使用已知包边界。
    """

    chirp_samples = config.chirp_samples
    downchirp = np.conjugate(
        build_upchirp(config.sf, symbol_id=0, os_factor=config.os_factor)
    ).astype(np.complex64)
    n_required = int(align_chirps * chirp_samples)
    start_min = max(0, int(event.start_sample) - int(search_radius_samples))
    start_max = min(
        int(np.asarray(samples).size) - n_required,
        int(event.start_sample) + int(search_radius_samples),
    )
    if start_max < start_min:
        raise ValueError(
            f"event {event.event_index} does not have enough samples for alignment"
        )

    best: dict[str, float | int] | None = None
    for candidate_start in range(start_min, start_max + 1, int(step_samples)):
        block = np.asarray(
            samples[candidate_start : candidate_start + n_required],
            dtype=np.complex64,
        )
        chirps = block.reshape(int(align_chirps), chirp_samples)
        spectrum = np.fft.fft(chirps * downchirp[np.newaxis, :], axis=1)
        energy = np.sum(np.abs(spectrum) ** 2, axis=0, dtype=np.float64)
        total_power = float(np.sum(energy, dtype=np.float64))
        if total_power <= 0.0:
            continue
        peak_bin = int(np.argmax(energy))
        peak_power = float(energy[peak_bin])
        second_power = (
            float(np.partition(energy, -2)[-2]) if energy.size > 1 else 0.0
        )
        if best is None or peak_power > float(best["align_score"]):
            best = {
                "aligned_start_sample": int(candidate_start),
                "align_offset_samples": int(candidate_start - int(event.start_sample)),
                "align_peak_bin": peak_bin,
                "align_peak_signed_bin": signed_fft_bin(peak_bin, chirp_samples),
                "align_peak_power": peak_power,
                "align_second_power": second_power,
                "align_total_power": total_power,
                "align_confidence_db": float(
                    10.0
                    * math.log10(
                        (peak_power + 1e-30) / (second_power + 1e-30)
                    )
                ),
                "align_peak_share": float(peak_power / total_power),
                "align_score": peak_power,
            }
    if best is None:
        raise ValueError(f"event {event.event_index} has no alignment candidate")
    return best


def _sync_event(
    samples: np.ndarray,
    event: DetectionEvent,
    detector: PreambleDetectorConfig,
    config: SinglePacketSyncConfig,
) -> SinglePacketSyncResult:
    """对一个前导码检测事件完成边界细化、帧定位和 FrameSync。"""

    os_factor = config.os_factor
    # 先用较大步长覆盖一整条 chirp 的不确定范围，再缩小搜索半径并细扫。
    coarse_step = max(1, os_factor * 8)
    fine_step = max(1, os_factor // 4)
    alignment = align_event_start(
        samples,
        event,
        detector,
        search_radius_samples=config.chirp_samples,
        step_samples=coarse_step,
        align_chirps=min(4, int(config.preamble_symbols)),
    )
    refined_event = replace(
        event, start_sample=int(alignment["aligned_start_sample"])
    )
    alignment = align_event_start(
        samples,
        refined_event,
        detector,
        search_radius_samples=max(128, 2 * coarse_step),
        step_samples=fine_step,
        align_chirps=min(4, int(config.preamble_symbols)),
    )

    # 帧定位器同时验证前导码、两条 sync-word chirp 和 SFD；粗定位允许相邻
    # symbol 偏移，细定位则固定 symbol 组合，只优化采样起点。
    coarse_locator = FrameLocatorConfig(
        preamble_len=float(config.preamble_symbols),
        sync_word=int(config.sync_word),
        search_radius_samples=max(1, config.chirp_samples // 8),
        step_samples=coarse_step,
        preamble_bin_tol=int(config.bin_tolerance),
        sync_bin_tol=int(config.sync_bin_tolerance),
        sfd_bin_tol=int(config.sfd_bin_tolerance),
        min_preamble_peaks=max(
            3, int(config.preamble_symbols) - int(config.bin_tolerance)
        ),
        symbol_search_span=2,
    )
    coarse_location = locate_frame_from_event(
        samples,
        event,
        detector,
        coarse_locator,
        coarse_start_sample=int(alignment["aligned_start_sample"]),
    )
    fine_locator = replace(
        coarse_locator,
        search_radius_samples=max(96, 2 * coarse_step),
        step_samples=fine_step,
        symbol_search_span=0,
    )
    frame_location = locate_frame_from_event(
        samples,
        event,
        detector,
        fine_locator,
        coarse_start_sample=int(coarse_location.preamble_start_sample),
    )
    # 这里进入项目已有的 FrameSync 实现，输出后续 Savaux 所需的全部同步量。
    frame_sync = run_grlora_frame_sync_validation(
        samples,
        frame_location,
        detector,
        float(config.preamble_symbols),
        int(config.sync_word),
        bin0_tol=int(config.frame_sync_bin0_tolerance),
        center_freq=float(config.center_frequency_hz),
    )
    status = "ok" if frame_location.valid and frame_sync.valid else "sync_invalid"
    return SinglePacketSyncResult(
        status=status,
        event_count=0,
        selected_event_index=int(event.event_index),
        alignment=alignment,
        frame_location=frame_location,
        frame_sync=frame_sync,
    )


def run_single_packet_sync(
    samples: np.ndarray,
    config: SinglePacketSyncConfig,
) -> SinglePacketSyncResult:
    """不使用 manifest 包边界，从输入 IQ 独立检测并同步一个 LoRa 包。"""

    config.validate()
    values = np.asarray(samples, dtype=np.complex64)
    detector = PreambleDetectorConfig(
        sf=int(config.sf),
        bw=float(config.bw_hz),
        samp_rate=float(config.sample_rate_hz),
        win_chirps=int(config.detection_chirps),
        hop_samples=int(config.chirp_samples),
        min_periodic_peaks=max(
            2,
            int(config.preamble_symbols) - int(config.detection_chirps) + 1,
        ),
        bin_tol=int(config.bin_tolerance),
    )
    # 只扫描配置允许的输入前缀；裁剪单包通常应产生一个候选，但保留多候选处理。
    _, events = detect_preamble_runs(
        values,
        detector,
        sample_limit=min(
            values.size, int(config.scan_chirps) * config.chirp_samples
        ),
    )
    if not events:
        return SinglePacketSyncResult(
            status="no_preamble",
            event_count=0,
            selected_event_index=None,
            alignment=None,
            frame_location=None,
            frame_sync=None,
        )

    candidates: list[SinglePacketSyncResult] = []
    errors: list[str] = []
    for event in events:
        try:
            candidates.append(_sync_event(values, event, detector, config))
        except (RuntimeError, ValueError) as exc:
            errors.append(
                f"event {event.event_index}: {type(exc).__name__}: {exc}"
            )
    if not candidates:
        return SinglePacketSyncResult(
            status="sync_error",
            event_count=len(events),
            selected_event_index=None,
            alignment=None,
            frame_location=None,
            frame_sync=None,
            error="; ".join(errors),
        )

    # 排序优先级：完整同步有效 > 帧定位有效 > 定位器得分。
    best = max(
        candidates,
        key=lambda item: (
            int(item.synchronized),
            int(item.frame_location.valid) if item.frame_location is not None else 0,
            float(item.frame_location.score)
            if item.frame_location is not None
            else -math.inf,
        ),
    )
    return replace(best, event_count=len(events))
