"""Branch4 固定帧实验的发射与采集参数。

这些值来自 STM32 工程
``LoraSTMacL1_2019.03.28_修改main函数_实现classA_通用版(Branch4)/apps/main.c``。
集中维护这份 profile，可以避免采集脚本、同步脚本和实验文件名各自保存一套旧参数。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


def _number_token(value: float) -> str:
    """把数值变成适合文件名的短字符串，例如 487.7 -> 487p7。"""

    return f"{float(value):g}".replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class LoraCaptureProfile:
    """一次固定 LoRa PHY/采集配置。"""

    name: str
    center_freq_hz: float
    sf: int
    bandwidth_hz: float
    coding_rate_index: int
    preamble_symbols: int
    sync_word: int
    explicit_header: bool
    payload_crc: bool
    app_payload_bytes: int
    phy_payload_bytes: int
    fixed_fcnt: int
    tx_power_dbm: int
    tx_period_s: float
    sample_rate_sps: float

    @property
    def coding_rate_denominator(self) -> int:
        return 4 + int(self.coding_rate_index)

    @property
    def oversampling_factor(self) -> int:
        ratio = float(self.sample_rate_sps) / float(self.bandwidth_hz)
        rounded = int(round(ratio))
        if rounded <= 0 or abs(ratio - rounded) > 1e-9:
            raise ValueError(f"采样率/带宽必须为正整数，当前为 {ratio:g}")
        return rounded

    def iq_filename(self, run: int = 1) -> str:
        """生成只包含离线解码参数的紧凑 complex64 IQ 文件名。"""

        if int(run) <= 0:
            raise ValueError("run 必须为正整数")
        return (
            f"sf{self.sf}"
            f"_bw{_number_token(self.bandwidth_hz / 1e3)}"
            f"_fs{_number_token(self.sample_rate_sps / 1e3)}"
            f"_pre{self.preamble_symbols}"
            f"_sw{self.sync_word:02x}"
            f"_r{int(run):03d}.bin"
        )

    def iq_relative_path(self, condition: str, run: int = 1) -> str:
        """生成实验条件子目录下的相对采集路径。"""

        aliases = {
            "highsnr": "high_snr",
            "high_snr": "high_snr",
            "lowsnr": "low_snr",
            "low_snr": "low_snr",
            "noiseonly": "noise_only",
            "noise_only": "noise_only",
            "interference": "interference",
        }
        normalized = str(condition).strip().lower().replace("-", "_").replace(" ", "_")
        folder = aliases.get(normalized)
        if folder is None:
            choices = ", ".join(sorted(set(aliases.values())))
            raise ValueError(f"未知 condition {condition!r}，可选值：{choices}")
        return f"{folder}/{self.iq_filename(run=run)}"


BRANCH4_PROFILE = LoraCaptureProfile(
    name="b4",
    center_freq_hz=487.7e6,
    sf=10,
    bandwidth_hz=125e3,
    coding_rate_index=3,
    preamble_symbols=32,
    sync_word=0x34,
    explicit_header=True,
    payload_crc=True,
    app_payload_bytes=20,
    phy_payload_bytes=33,
    fixed_fcnt=1,
    tx_power_dbm=2,
    tx_period_s=3.0,
    sample_rate_sps=500e3,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Branch4 固定帧实验的推荐 IQ 文件名。")
    parser.add_argument(
        "--condition",
        default="high_snr",
        help="实验条件目录：high_snr、low_snr、noise_only 或 interference。",
    )
    parser.add_argument("--run", type=int, default=1, help="重复采集编号，从 1 开始。")
    args = parser.parse_args()
    print(BRANCH4_PROFILE.iq_relative_path(args.condition, args.run))


if __name__ == "__main__":
    main()
