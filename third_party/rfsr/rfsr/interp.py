import numpy as np

import torch
import torch.nn.functional as F

from scipy import signal
from scipy.interpolate import CubicSpline, make_interp_spline, PchipInterpolator

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess

    _HAS_LOWESS = True
except ImportError:
    _HAS_LOWESS = False


def upsample_nearestneighbour(data: np.ndarray, up: int) -> np.ndarray:
    return np.repeat(data, up)


def upsample_linear(data: np.ndarray, up: int) -> np.ndarray:
    n = len(data)
    if n < 2:
        return data.copy()
    x = np.arange(n)
    x_up = np.linspace(0, n - 1, n * up)
    return np.interp(x_up, x, data.real) + 1j * np.interp(x_up, x, data.imag)


def upsample_poly(data: np.ndarray, up: int = 2) -> np.ndarray:
    return signal.resample_poly(data, up, 1)


def upsample_fft(data: np.ndarray, up: int = 2) -> np.ndarray:
    return signal.resample(data, len(data) * up)


def upsample_cubicspline(data: np.ndarray, up: int = 2, boundarycondition="natural") -> np.ndarray:
    n = len(data)
    if n < 2:
        return data.copy()
    t_original = np.arange(n)
    t_new = np.linspace(0, n - 1, n * up)
    cs_real = CubicSpline(t_original, data.real, bc_type=boundarycondition)
    cs_imag = CubicSpline(t_original, data.imag, bc_type=boundarycondition)
    return cs_real(t_new) + 1j * cs_imag(t_new)


def upsample_bspline(data: np.ndarray, up: int, k: int = 3) -> np.ndarray:
    n = len(data)
    if n < 2:
        return data.copy()
    x = np.arange(n)
    x_up = np.linspace(0, n - 1, n * up)
    spline_real = make_interp_spline(x, data.real, k=k)
    spline_imag = make_interp_spline(x, data.imag, k=k)
    return spline_real(x_up) + 1j * spline_imag(x_up)


def upsample_pchip(data: np.ndarray, up: int) -> np.ndarray:
    n = len(data)
    if n < 2:
        return data.copy()
    x = np.arange(n)
    x_up = np.linspace(0, n - 1, n * up)
    pchip_real = PchipInterpolator(x, data.real)
    pchip_imag = PchipInterpolator(x, data.imag)
    return pchip_real(x_up) + 1j * pchip_imag(x_up)


def upsample_loess(data: np.ndarray, up: int, frac: float = 0.2) -> np.ndarray:
    if not _HAS_LOWESS:
        raise ImportError("statsmodels.lowess not available")
    n = len(data)
    if n < 2:
        return data.copy()
    x = np.arange(n)
    x_up = np.linspace(0, n - 1, n * up)
    real_smooth = lowess(data.real, x, frac=frac, return_sorted=False)
    imag_smooth = lowess(data.imag, x, frac=frac, return_sorted=False)
    return np.interp(x_up, x, real_smooth) + 1j * np.interp(x_up, x, imag_smooth)


class KalmanIQ:
    def __init__(self, process_var: float = 1e-5, meas_var: float = 1e-2):
        self.Q = process_var
        self.R = meas_var
        self.P = 1.0
        self.x = None

    def update(self, z: float) -> float:
        if self.x is None:
            self.x = z
            return z
        P_pred = self.P + self.Q
        K = P_pred / (P_pred + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * P_pred
        return self.x


def upsample_kalman(data: np.ndarray, up: int = 2, process_var: float = 1e-5, meas_var: float = 1e-2) -> np.ndarray:
    n = len(data)
    if n < 2:
        return data.copy()
    kf_real = KalmanIQ(process_var, meas_var)
    kf_imag = KalmanIQ(process_var, meas_var)
    real_filtered = np.array([kf_real.update(z) for z in data.real])
    imag_filtered = np.array([kf_imag.update(z) for z in data.imag])
    x = np.arange(n)
    x_up = np.linspace(0, n - 1, n * up)
    return np.interp(x_up, x, real_filtered) + 1j * np.interp(x_up, x, imag_filtered)


# --- Parameters ---
signal_len = 1000
up = 4
down = 1
filter_width = 511
beta = 14.769656459379492  # ~60 dB attenuation

# --- Generate complex signal ---
torch.manual_seed(42)
x = torch.randn(signal_len, dtype=torch.complex64)


# --- Kaiser-windowed sinc filter ---
def design_filter(up: int, down: int, width: int, beta: float):
    cutoff = 1 / max(up, down)
    t = torch.arange(-width // 2, width // 2 + 1, dtype=torch.float32)
    sinc = torch.where(t == 0, torch.tensor(1.0), torch.sin(np.pi * t * cutoff) / (np.pi * t * cutoff))

    arg = 1 - (t / (width / 2)) ** 2
    arg = torch.clamp(arg, min=0.0)
    window = torch.i0(beta * torch.sqrt(arg)) / torch.i0(torch.tensor(beta))

    h = sinc * window
    h /= h.sum()
    return h


# --- Apply filter to real and imaginary parts ---
def apply_filter(x, h):
    x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, time]
    h = h.flip(0).unsqueeze(0).unsqueeze(0)  # [1, 1, kernel]
    return F.conv1d(x, h, padding=h.shape[-1] // 2).squeeze()


# --- PyTorch resample_poly ---
def resample_poly_torch(data, up, down=1, width=511, beta=14.769):
    if isinstance(data, torch.Tensor):
        x = data
        isnumpy = False
    else:
        x = torch.tensor(data)
        isnumpy = True

    x_real = x.real
    x_imag = x.imag

    # Upsample by inserting zeros
    x_up_real = torch.zeros(x_real.shape[0] * up)
    x_up_imag = torch.zeros(x_imag.shape[0] * up)
    x_up_real[::up] = x_real
    x_up_imag[::up] = x_imag

    # Filter
    h = design_filter(up, down, width, beta)
    x_filt_real = apply_filter(x_up_real, h)
    x_filt_imag = apply_filter(x_up_imag, h)

    # Downsample
    x_down_real = x_filt_real[::down]
    x_down_imag = x_filt_imag[::down]

    if isnumpy:
        return torch.complex(x_down_real, x_down_imag).numpy()
    return torch.complex(x_down_real, x_down_imag)


def resample_poly_torch_batch(data, up, down=1, width=511, beta=14.769):
    """
    Polyphase resampling for batched complex tensors.

    Args:
        data: torch.Tensor of shape (B, L), complex dtype
        up: upsampling factor
        down: downsampling factor
        width, beta: filter design params
    Returns:
        torch.Tensor of shape (B, new_len), complex dtype
    """

    if not torch.is_complex(data):
        raise ValueError("Input must be a complex torch.Tensor of shape (B, L)")

    B, L = data.shape
    x_real, x_imag = data.real, data.imag

    # --- 升采样（在相邻原始样点之间插零）---
    # 这是 polyphase 重采样的标准中间表示，后面的低通卷积会根据原始样点
    # 重建插值点；这些零不是 packet 前后的静默段，也不是环境噪声样本。
    L_up = L * up
    x_up_real = torch.zeros(B, L_up, device=data.device, dtype=x_real.dtype)
    x_up_imag = torch.zeros(B, L_up, device=data.device, dtype=x_imag.dtype)
    x_up_real[:, ::up] = x_real
    x_up_imag[:, ::up] = x_imag

    # --- Filter ---
    h = design_filter(up, down, width, beta)  # assume returns 1D torch filter
    h = h.to(data.device, dtype=x_real.dtype)

    # Apply as 1D convolution with groups=B (process each batch independently)
    h_flip = h.view(1, 1, -1)  # shape (out_channels=1, in_channels=1, kernel_size)
    x_up_real = x_up_real.unsqueeze(1)  # (B, 1, L_up)
    x_up_imag = x_up_imag.unsqueeze(1)

    x_filt_real = F.conv1d(x_up_real, h_flip, padding=h.numel() // 2, groups=1).squeeze(1)
    x_filt_imag = F.conv1d(x_up_imag, h_flip, padding=h.numel() // 2, groups=1).squeeze(1)

    # --- Downsample ---
    x_down_real = x_filt_real[:, ::down]
    x_down_imag = x_filt_imag[:, ::down]

    return torch.complex(x_down_real, x_down_imag)


def resample_poly_torch_batch2(data: torch.Tensor, up: int, down: int = 1, width: int = 511, beta: float = 14.769):
    """
    Polyphase resampling for batched real+imag tensors.

    Args:
        data: torch.Tensor of shape (B, 2, L), real dtype
              channel 0 = real, channel 1 = imag
        up: upsampling factor
        down: downsampling factor
        width, beta: filter design params

    Returns:
        torch.Tensor of shape (B, 2, L_out), real dtype
    """

    if data.ndim != 3 or data.shape[1] != 2:
        raise ValueError("Input must have shape (B, 2, L) where channel 0=real, 1=imag")

    B, C, L = data.shape
    assert C == 2, "Second dimension must be 2 (real, imag)"

    x_real, x_imag = data[:, 0, :], data[:, 1, :]

    # --- 升采样（在相邻原始样点之间插零）---
    # 插零随后必须配合低通滤波，不能把这些临时零点理解为网络的 off-packet
    # 训练数据；函数最终返回的是滤波后的连续插值结果。
    L_up = L * up
    x_up_real = torch.zeros(B, L_up, device=data.device, dtype=data.dtype)
    x_up_imag = torch.zeros(B, L_up, device=data.device, dtype=data.dtype)
    x_up_real[:, ::up] = x_real
    x_up_imag[:, ::up] = x_imag

    # --- Filter ---
    h = design_filter(up, down, width, beta)  # assume returns 1D torch filter
    h = h.to(data.device, dtype=data.dtype)

    h_flip = h.view(1, 1, -1)  # (1, 1, K)
    # add channel dimension for conv1d
    x_up_real = x_up_real.unsqueeze(1)  # (B, 1, L_up)
    x_up_imag = x_up_imag.unsqueeze(1)

    L_filt = L_up  # expected length after filtering
    x_filt_real = F.conv1d(x_up_real, h_flip, padding=int(h.numel() // 2)).squeeze(1)[..., :L_filt]
    x_filt_imag = F.conv1d(x_up_imag, h_flip, padding=int(h.numel() // 2)).squeeze(1)[..., :L_filt]

    # --- Downsample ---
    x_down_real = x_filt_real[:, ::down]
    x_down_imag = x_filt_imag[:, ::down]

    # Stack back into (B, 2, L_out)
    return torch.stack([x_down_real, x_down_imag], dim=1)


# Inventory of interpolation functions
interp_fn_inventory = {
    "nearestneighbour": upsample_nearestneighbour,
    "linear": upsample_linear,
    "cubicspline": upsample_cubicspline,
    "bspline": upsample_bspline,
    "pchip": upsample_pchip,
    "poly": upsample_poly,
    "fft": upsample_fft,
    "pytorch": resample_poly_torch,
    # "loess": upsample_loess, # extremely slow
    # "kalman": upsample_kalman, # bad performance
}

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # Example IQ data
    data = np.array([1 + 1j, 2 + 2j, 0 + 0j, 3 + 3j, 1 + 1j], dtype=np.complex64)
    up = 4  # upsampling factor

    # Original sample positions
    x_orig = np.arange(len(data))

    # Create figure
    plt.figure(figsize=(12, 6))

    # Plot each interpolated signal
    for fn in interp_fn_inventory:
        upsampled = interp_fn_inventory[fn](data, up)
        x_up = np.linspace(0, len(data) - 1, len(upsampled))
        plt.plot(x_up, upsampled.real, label=f"{fn} (real)")
        # plt.plot(x_up, upsampled.imag, '--', label=f"{fn} (imag)") # imaginary part, disabled for readability

    # Plot original signal
    plt.plot(x_orig, data.real, 'ko-', label="original (real)")
    # plt.plot(x_orig, data.imag, 'k--o', label="original (imag)") # imaginary part, disabled for readability

    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.title(f"IQ Upsampling Comparison (factor {up})")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True)
    plt.show()
