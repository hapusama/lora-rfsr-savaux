import json
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset

from rfsr import (
    apply_hi2lora_sto,
    awgn,
    encode,
    encode_random_raw_phy,
)

# 当前文件位于 rfsr/nn/，OTA 参考文件路径会以这个目录为基准拼接。
BASE_DIR = Path(__file__).resolve().parent


# --- 合成 LoRa 内存数据集 ---
class SyntheticLoRaDataset(Dataset):
    """一次性生成并缓存成对的低采样率输入与高采样率标签。

    返回值：
        x:   float32，[2, L]，250 kSPS 的带噪 I/Q。
        y:   float32，[2, OSF*L]，高采样率干净 I/Q 标签；论文配置
             OSF=4 时为 1 MSPS。
        snr: float32，[1]，当前样本加噪时使用的 SNR。

    注意：样本在 __init__ 中一次性生成，后续 epoch 只会重复读取，
    并不会在每个 epoch 重新随机 payload、SNR 或噪声。
    """

    def __init__(
        self,
        oversampling=1,
        size=100,
        payload_length=16,
        downsampling=8,
        SF=12,
        BW=125e3,
        snr_range=(-22.0, 10.0),
    ):
        # 唯一 packet 数量；这些 packet 会全部预生成并保存在内存中。
        self.size = size

        # 应用层随机 payload 的字节数，不包含 PHY.py 自动加入的伪头和 CRC。
        self.payload_length = payload_length
        # 输出相对输入的升采样倍数；论文和公开权重使用 OSF=4。
        self.OSF = oversampling

        # 信道参数由训练入口传入；相同上下界表示固定 SNR。
        self.snr_range = (
            float(min(snr_range)),
            float(max(snr_range)),
        )

        # LoRa 物理层参数。
        self.center_freq = 915e6
        self.sf = SF
        self.bw = BW
        print(f"BW={BW}")
        # downsampling=8 时，低速率输入为 2 MSPS / 8 = 250 kSPS。
        # 后面生成标签时还会乘 OSF，因此 OSF=4 时标签为 1 MSPS。
        self.sample_rate = 2e6 / downsampling  # make it equivalent to the OTA: default: Fs=2e6  (default: downsampling /8 -> 0.25e6)
        self.src = 0
        self.dst = 1
        self.seqn = 7
        self.cr = 4
        self.enable_crc = 1
        self.implicit_header = 0
        self.preamble_bits = 8

        # 估算低、高采样率记录长度。实际训练输入仍由高采样率 y 每 OSF
        # 点抽取一次得到，input_len/output_len 没有参与后面的切片。
        self.input_len = len(
            encode(self.center_freq, self.sf, self.bw, np.ones(self.payload_length, dtype=np.uint8), self.sample_rate,
                   self.src, self.dst, self.seqn, self.cr, self.enable_crc, self.implicit_header, self.preamble_bits))
        self.output_len = self.input_len * self.OSF

        # 一次性生成整个数据集。这样训练较快，但不同 epoch 看到的是同一批
        # payload、SNR 和噪声 realization。
        x_tensors, y_tensors, snr_tensors = [], [], []
        print(f"Dataset size: {self.size}")
        for i in range(self.size):

            if i % 100 == 0:
                print(f"Added: {i} elements")

            # 生成高采样率干净标签 y。
            # randint 的上界不包含在结果中，因此 payload 字节范围实际为 0..254。
            # PHY.encode 还会在 packet 前加入固定的零样点，并生成前导码、
            # 显式头、payload 编码和 CRC。
            y = encode(self.center_freq, self.sf, self.bw,
                       np.random.randint(0, 2 ** 8 - 1, size=self.payload_length, dtype=np.uint8),
                       self.sample_rate * self.OSF, self.src, self.dst, self.seqn, self.cr, self.enable_crc,
                       self.implicit_header, self.preamble_bits)

            # 从高采样率标签直接抽取低采样率输入，保证 x/y 时间对齐。
            # 只对低采样率 x 加 AWGN，高采样率 y 保持干净。
            x = y[::self.OSF]  # extract down-sampled signal
            snr_db = np.random.uniform(*self.snr_range)
            x = awgn(x, snr_db)  # add noise
            # encode() 生成的 y 前面有固定的零值静默段；AWGN 会把 x 中对应
            # 的静默段变成纯白噪声，而 y 仍为 0。因此网络能看到“包外白噪声
            # -> 干净静默”的样本，但看不到真实现场的非高斯环境噪声。

            # 将 complex IQ 拆为两个实数通道：
            # x_tensor.shape = [2, L]，y_tensor.shape = [2, OSF*L]。
            x_tensor = torch.tensor(np.stack([x.real, x.imag]), dtype=torch.float32)
            y_tensor = torch.tensor(np.stack([y.real, y.imag]), dtype=torch.float32)
            snr_tensor = torch.tensor([snr_db], dtype=torch.float32)

            x_tensors.append(x_tensor)
            y_tensors.append(y_tensor)
            snr_tensors.append(snr_tensor)

        # 堆叠 batch 维度，最终形状分别为：
        # [size, 2, L]、[size, 2, OSF*L] 和 [size, 1]。
        self.x_batched = torch.stack(x_tensors, dim=0)
        self.y_batched = torch.stack(y_tensors, dim=0)

        # 保存每个 packet 对应的 SNR。
        self.snr_batched = torch.stack(snr_tensors, dim=0)

    def __len__(self):
        """返回预生成的唯一 packet 数量。"""
        return self.size

    def __getitem__(self, idx):
        """返回一个已经缓存好的 (低速率输入, 高速率标签, SNR) 三元组。"""
        return self.x_batched[idx], self.y_batched[idx], self.snr_batched[idx],


class ReferencePhyPretrainingDataset(Dataset):
    """按 reference metadata 的 PHY 配置构造合成预训练数据对。

    默认仅使用 metadata 中的 PHY 参数和 raw frame 长度，并在初始化时调用
    ``PHY.encode_random_raw_phy`` 生成 ``size`` 个随机 payload 和 clean
    标签 ``y``。这些样本随后缓存，后续 epoch 只会重新打乱读取顺序。
    ``random_payload=False`` 时改为读取已有 cfile，作为固定 payload 对照。
    两种模式都会按 ``oversampling`` 构造 ``x``，并只给 ``x`` 添加 STO、
    CFO 和 AWGN；高采样率标签 ``y`` 始终保持理想波形。
    """

    def __init__(
        self,
        reference_root,
        oversampling=4,
        size=250,
        snr_range=(-22.0, 10.0),
        expected_sample_rate_hz=1_000_000,
        expected_sf=12,
        expected_bandwidth_hz=125_000,
        seed=42,
        random_payload=True,
        cfo_range_hz=(-35_000.0, 35_000.0),
        sto_enabled=True,
        sto_initial_range_chips=(-0.5, 0.5),
        sto_slope_range_chips_per_symbol=(-0.05, 0.05),
    ):
        self.reference_root = Path(reference_root).expanduser().resolve()
        self.reference_dir = self.reference_root / "reference"
        self.metadata_dir = self.reference_root / "metadata"
        self.OSF = int(oversampling)
        self.size = int(size)
        self.random_payload = bool(random_payload)
        self.snr_range = (
            float(min(snr_range)),
            float(max(snr_range)),
        )
        self.cfo_range_hz = (
            float(min(cfo_range_hz)),
            float(max(cfo_range_hz)),
        )
        self.sto_enabled = bool(sto_enabled)
        self.sto_initial_range_chips = (
            float(min(sto_initial_range_chips)),
            float(max(sto_initial_range_chips)),
        )
        self.sto_slope_range_chips_per_symbol = (
            float(min(sto_slope_range_chips_per_symbol)),
            float(max(sto_slope_range_chips_per_symbol)),
        )
        self.rng = np.random.default_rng(seed)
        self.last_cfo_hz = 0.0
        self.last_initial_sto_chips = 0.0
        self.last_sto_slope_chips_per_symbol = 0.0

        if self.OSF < 1:
            raise ValueError(f"oversampling must be positive, got {self.OSF}.")
        if self.size < 1:
            raise ValueError(f"size must be positive, got {self.size}.")
        if not self.metadata_dir.is_dir():
            raise FileNotFoundError(
                f"reference root must contain metadata/: {self.reference_root}"
            )
        if not self.random_payload and not self.reference_dir.is_dir():
            raise FileNotFoundError(
                "fixed-payload mode requires reference/: "
                f"{self.reference_root}"
            )

        metadata_paths = sorted(self.metadata_dir.glob("*.json"))
        if not metadata_paths:
            raise FileNotFoundError(
                f"no metadata JSON files found in {self.metadata_dir}"
            )

        records = []
        expected_length = None
        raw_phy_config = None
        payload_length = None
        for metadata_path in metadata_paths:
            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            self._validate_metadata(
                metadata,
                metadata_path,
                expected_sample_rate_hz,
                expected_sf,
                expected_bandwidth_hz,
            )
            current_config = self._raw_phy_config_from_metadata(metadata)
            current_payload_length = int(metadata["packet"]["frame_bytes"])
            if raw_phy_config is None:
                raw_phy_config = current_config
                payload_length = current_payload_length
            elif current_config != raw_phy_config:
                raise ValueError(
                    f"{metadata_path} has a different PHY configuration."
                )
            elif current_payload_length != payload_length:
                raise ValueError(
                    f"{metadata_path} has frame_bytes={current_payload_length}, "
                    f"expected {payload_length}."
                )

            iq = metadata["iq"]
            iq_path = self.reference_root / str(iq["relative_path"])
            complex_samples = int(iq["complex_samples"])
            if not self.random_payload:
                if not iq_path.is_file():
                    raise FileNotFoundError(
                        f"reference IQ declared by {metadata_path} does not "
                        f"exist: {iq_path}"
                    )
                expected_bytes = complex_samples * np.dtype("<c8").itemsize
                if iq_path.stat().st_size != expected_bytes:
                    raise ValueError(
                        f"{iq_path} size does not match metadata: "
                        f"{iq_path.stat().st_size} != {expected_bytes}."
                    )
            if complex_samples % self.OSF:
                raise ValueError(
                    f"{iq_path} length {complex_samples} is not divisible by "
                    f"oversampling={self.OSF}."
                )
            if expected_length is None:
                expected_length = complex_samples
            elif complex_samples != expected_length:
                raise ValueError(
                    "all references must have the same length for batching; "
                    f"{iq_path} has {complex_samples}, expected {expected_length}."
                )
            records.append(
                {
                    "iq_path": iq_path,
                    "payload_id": int(metadata["packet"]["payload_id"]),
                    "complex_samples": complex_samples,
                }
            )

        self.records = tuple(records)
        self.output_len = int(expected_length or 0)
        self.input_len = self.output_len // self.OSF
        self.raw_phy_config = dict(raw_phy_config or {})
        self.payload_length = int(payload_length or 0)

        # Cache the entire synthetic dataset once. Preallocation avoids a
        # second full copy that would result from collecting then stacking.
        self.x_batched = torch.empty(
            (self.size, 2, self.input_len), dtype=torch.float32
        )
        self.y_batched = torch.empty(
            (self.size, 2, self.output_len), dtype=torch.float32
        )
        self.snr_batched = torch.empty(
            (self.size, 1), dtype=torch.float32
        )
        print(f"Dataset size: {self.size}")
        for index in range(self.size):
            if index % 10 == 0:
                print(f"Added: {index} elements")
            x, y, snr = self._generate_sample(index)
            self.x_batched[index].copy_(x)
            self.y_batched[index].copy_(y)
            self.snr_batched[index].copy_(snr)

    @staticmethod
    def _validate_metadata(
        metadata,
        metadata_path,
        expected_sample_rate_hz,
        expected_sf,
        expected_bandwidth_hz,
    ):
        if metadata.get("schema") != "lora-rfsr-reference":
            raise ValueError(f"unsupported reference schema in {metadata_path}.")
        if metadata.get("reference_kind") != "ideal_tx_complex_baseband":
            raise ValueError(
                f"{metadata_path} is not an ideal synthetic reference."
            )
        if bool(metadata.get("iq", {}).get("awgn_added", True)):
            raise ValueError(f"{metadata_path} already contains added AWGN.")
        if str(metadata.get("iq", {}).get("dtype")) != "<c8":
            raise ValueError(f"{metadata_path} must declare dtype '<c8'.")

        phy = metadata.get("phy", {})
        checks = (
            ("sample_rate_hz", expected_sample_rate_hz),
            ("sf", expected_sf),
            ("bandwidth_hz", expected_bandwidth_hz),
        )
        for key, expected in checks:
            if expected is not None and int(phy.get(key, -1)) != int(expected):
                raise ValueError(
                    f"{metadata_path} declares {key}={phy.get(key)!r}, "
                    f"expected {expected}."
                )

    @staticmethod
    def _raw_phy_config_from_metadata(metadata):
        phy = metadata["phy"]
        return {
            "fs": int(phy["sample_rate_hz"]),
            "SF": int(phy["sf"]),
            "BW": int(phy["bandwidth_hz"]),
            "cr": int(phy["cr"]),
            "enable_crc": int(bool(phy["phy_crc"])),
            "implicit_header": int(not bool(phy["explicit_header"])),
            "preamble_bits": int(phy["preamble_symbols"]),
            "sync_word": int(phy["sync_word"]),
            "ldro": bool(phy["ldro"]),
            "crc_mode": str(phy["crc_mode"]),
            "leading_silence_samples": int(phy["leading_silence_samples"]),
            "trailing_silence_samples": int(phy["trailing_silence_samples"]),
        }

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if not 0 <= int(idx) < self.size:
            raise IndexError(idx)
        return (
            self.x_batched[idx],
            self.y_batched[idx],
            self.snr_batched[idx],
        )

    def _generate_sample(self, idx):
        if not 0 <= int(idx) < self.size:
            raise IndexError(idx)

        if self.random_payload:
            encoded = encode_random_raw_phy(
                self.payload_length,
                rng=self.rng,
                **self.raw_phy_config,
            )
            y = encoded.samples
            if y.size != self.output_len:
                raise RuntimeError(
                    "random PHY output length changed: "
                    f"{y.size} != {self.output_len}."
                )
        else:
            record = self.records[int(idx) % len(self.records)]
            clean = np.memmap(
                record["iq_path"],
                dtype=np.dtype("<c8"),
                mode="r",
                shape=(int(record["complex_samples"]),),
            )
            y = np.asarray(clean)
        sampled_initial_sto = float(
            self.rng.uniform(*self.sto_initial_range_chips)
        )
        sampled_sto_slope = float(
            self.rng.uniform(*self.sto_slope_range_chips_per_symbol)
        )
        if self.sto_enabled:
            initial_sto_chips = sampled_initial_sto
            sto_slope_chips_per_symbol = sampled_sto_slope
        else:
            initial_sto_chips = 0.0
            sto_slope_chips_per_symbol = 0.0
        self.last_initial_sto_chips = initial_sto_chips
        self.last_sto_slope_chips_per_symbol = (
            sto_slope_chips_per_symbol
        )
        x_clean = apply_hi2lora_sto(
            y,
            fs=self.raw_phy_config["fs"],
            BW=self.raw_phy_config["BW"],
            SF=self.raw_phy_config["SF"],
            output_decimation=self.OSF,
            preamble_bits=self.raw_phy_config["preamble_bits"],
            leading_silence_samples=self.raw_phy_config[
                "leading_silence_samples"
            ],
            trailing_silence_samples=self.raw_phy_config[
                "trailing_silence_samples"
            ],
            initial_sto_chips=initial_sto_chips,
            sto_slope_chips_per_symbol=sto_slope_chips_per_symbol,
        )

        # 标签保持理想零 STO/CFO；仅在低采样率输入上模拟接收机频偏。
        cfo_hz = float(self.rng.uniform(*self.cfo_range_hz))
        self.last_cfo_hz = cfo_hz
        if cfo_hz != 0.0:
            high_rate_hz = float(self.raw_phy_config["fs"])
            leading_samples = int(
                self.raw_phy_config["leading_silence_samples"]
            )
            sample_offsets = (
                np.arange(x_clean.size, dtype=np.float64) * self.OSF
                - leading_samples
            )
            cfo_rotation = np.exp(
                2j * np.pi * cfo_hz * sample_offsets / high_rate_hz
            ).astype(np.complex64)
            x_clean = np.asarray(x_clean * cfo_rotation, dtype=np.complex64)

        snr_db = float(self.rng.uniform(*self.snr_range))
        signal_power = float(np.mean(np.abs(x_clean) ** 2))
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        noise_scale = np.sqrt(noise_power / 2.0)
        noise = noise_scale * (
            self.rng.standard_normal(x_clean.shape)
            + 1j * self.rng.standard_normal(x_clean.shape)
        )
        x = np.asarray(x_clean + noise, dtype=np.complex64)

        x_tensor = torch.from_numpy(
            np.stack([x.real, x.imag], axis=0)
        ).to(dtype=torch.float32)
        y_tensor = torch.from_numpy(
            np.stack([y.real, y.imag], axis=0)
        ).to(dtype=torch.float32)
        snr_tensor = torch.tensor([snr_db], dtype=torch.float32)
        return x_tensor, y_tensor, snr_tensor



class _LegacyUpstreamOTALoRaDataset(Dataset):
    """读取作者预处理后的逐包 OTA IQ 与配对参考 IQ。

    从当前 loader 可以还原出的磁盘契约如下（目录是作者代码的意图）：

        rfsr/nn/data/post/
        ├── exp1_000020_rxg18_0_fulltrim.cfile   # 2 MSPS OTA 接收 IQ
        └── signalout_000020_fulltrim.cfile      # 同一 packet 的 2 MSPS reference

    两种文件都是没有文件头的一维 NumPy complex64 流，即每个采样点按
    float32 I + float32 Q 存储。每个 dataset item 对应一个已经检测、同步
    并逐包裁剪的 ``*_fulltrim.cfile``，而不是连续 USRP 录音。

    默认 DSF=8、OSF=4 时，loader 返回：
        x: [2, L]，接收文件从 2 MSPS 抽取到 250 kSPS。
        y: [2, 4L]，reference 从 2 MSPS 抽取到 1 MSPS。
        snr: 该接收 packet 的标量 SNR（return_snr=True 时）。

    预处理必须保证 x/y 是同一 packet、覆盖相同物理时间，并在抽取后满足
    ``len(y) == OSF * len(x)``；本 loader 不执行包检测、时间对齐、幅度
    归一化或长度修正。

    当前仓库没有包含从连续 USRP IQ 生成 fulltrim 文件的流程；文件筛选、
    SNR 元数据以及可复现划分函数也依赖本文件之外的辅助代码。
    """

    def __init__(self, oversampling = 1,
                 snr_range = (-20, 20), #(-35, 20),
                 training=True,
                 trim=False,
                 return_snr=False,
                 downsampling=8):
        raise RuntimeError(
            "The incomplete upstream OTA loader is disabled. Import "
            "rfsr.nn.OTALoRaDataset to use the local views.csv contract."
        )
        self.return_snr = return_snr
        self.training = training
        self.trim = trim
        self.OSF = oversampling
        self.DSF = downsampling
        # OTA 原始 IQ 默认按 2 MSPS 记录，DSF 决定网络输入采样率：
        # DSF=8 -> 250 kSPS
        # DSF=4 -> 500 kSPS
        # DSF=2 ->   1 MSPS
        # DSF=1 ->   2 MSPS

        # 数据目录与 SNR 过滤。DATA_DIRECTORY 是作者环境中的硬编码路径；
        # filter_files_by_snr 需要返回 (接收文件路径列表, 对应 SNR 列表)。
        # 该函数未随公开仓库发布，所以这里尚不能直接发现你的本地数据。
        DATA_DIRECTORY = str(
            Path(__file__).resolve().parents[4]
            / "data"
            / "reference_phy"
            / "rfsr_db"
        )
        min_snr, max_snr = snr_range
        final_list, self.snrs = filter_files_by_snr(DATA_DIRECTORY, min_snr, max_snr)

        # 将原始 OTA 文件名映射到已经预处理、逐包裁剪的 fulltrim 文件。
        # 因此辅助函数返回的路径必须包含 "data/ota" 且以 ".cfile" 结尾。
        self.files = list(map(lambda x: x.replace("data/ota", f"{BASE_DIR}/data/post").replace(".cfile", "_fulltrim.cfile"), final_list))

        # 根据接收文件名提取 reference 编号。作者的数据中 reference 编号
        # 每 100 个 packet 循环一次；这是作者数据集特有的命名/配对规则，
        # 自建数据集可以用 manifest 显式保存 rx/ref/snr，避免依赖文件名。
        def filename2refnum(filename):
            # 示例文件名：exp1_000020_rxg18_0.cfile
            basename = os.path.basename(filename)
            # 拆分后为 ['exp1', '000020', 'rxg18', '0.cfile']。
            parts = basename.split('_')
            # 第二段是 packet/reference 序号。
            extracted_id = parts[1]
            return f"{int(extracted_id)%100:06d}"

        self.refmap = list(map(lambda x: filename2refnum(x), self.files))

        self.size = len(self.files)
        print(f"OTALoRaDataset contains {self.size} items.")

        # 按论文设置随机划分 80% 训练集、20% 测试集。应按原始 packet
        # 身份划分，再做噪声增强，避免同一个 clean packet 泄漏到两边。
        # get_reproducible_split 需要由外部辅助代码提供。
        self.train_indices, self.test_indices = get_reproducible_split(self.size, test_split=0.2)


    def __len__(self):
        """根据 training 标志返回训练子集或测试子集大小。"""
        if self.training:
            return len(self.train_indices)
        else:
            return len(self.test_indices)


    def __getitem__(self, idx):

        # DataLoader 传入的是子集索引，先还原成完整文件列表的索引。
        if self.training:
            idx = self.train_indices[idx]
        else:
            idx = self.test_indices[idx]

        # 读取一个已经预处理好的 OTA packet。np.fromfile 不解析文件头，
        # 因此源文件必须就是连续排列的 complex64，不能直接传 sc16 IQ。
        signal = np.fromfile(self.files[idx], dtype=np.complex64)

        if self.trim:
            # model3 的显存优化路径会额外裁掉两端；公开四层 model0
            # 默认 trim=False，不执行这一操作。
            TRIM_SIZE = 460_000
            signal = signal[TRIM_SIZE: -TRIM_SIZE]

        # 从 2 MSPS 接收 IQ 构造低采样率网络输入。
        signal = signal[::self.DSF] # load 2MSPS OTA signal

        # 根据 refmap 加载同一个 packet 对应的高采样率 reference/标签。
        # reference 可以是已知 payload 生成并对齐的干净基带，或对齐后的
        # 高 SNR 接收波形；仅“CRC 能过”并不代表它是无噪声标签。
        ref_filename = os.path.join(BASE_DIR, "data/post/", f"signalout_{self.refmap[idx]}_fulltrim.cfile")
        ref_signal = np.fromfile(ref_filename, dtype=np.complex64) # load 2MSPS reference signal (label)

        # 把 2 MSPS reference 调整到目标输出采样率。
        # 论文设置 DSF=8、OSF=4，因此 reference 每 2 点取 1 点，
        # 得到 1 MSPS 标签；输入则是 250 kSPS。
        if self.DSF/self.OSF < 1:
            raise RuntimeError(f"Invalid DSF/OSF ratio: {self.DSF}/{self.OSF}, needs to be >=1")
        ref_signal = ref_signal[::int(self.DSF/self.OSF)]

        if self.trim:
            # 对标签执行作者原有的可选裁剪。注意：标签是在降采样后再裁剪，
            # 而输入是在降采样前裁剪；model0 默认 trim=False。
            ref_signal = ref_signal[TRIM_SIZE: -TRIM_SIZE]

        # complex64 -> float32 双通道：
        # x.shape = [2, L]，y.shape = [2, OSF*L]。
        # 训练损失要求预测与 y 长度完全相等；当前代码不会自动 crop/pad。
        x = torch.tensor(
            np.stack([signal.real, signal.imag], axis=0),
            dtype=torch.float32
        )
        y = torch.tensor(
            np.stack([ref_signal.real, ref_signal.imag], axis=0),
            dtype=torch.float32
        )

        if self.return_snr:
            # model0v0 的 forward 接口接收 SNR，但非 gated 模型实际不使用它。
            return x, y, self.snrs[idx] # also return the snr
        else:
            return x, y


# Public OTA loading is implemented by the repository-local manifest contract.
# Keep the upstream class above private only as a readable provenance snapshot.
from .ota_dataset import OTALoRaDataset  # noqa: E402,F401

