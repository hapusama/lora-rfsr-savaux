import numpy as np
from scipy.signal import kaiserord, firwin, filtfilt, remez


def shift_frequency(signal, f_shift, fs):
    """
    Shifts a complex signal in frequency.

    Args:
        signal (np.ndarray): The complex input signal (e.g., complex64).
        f_shift (float): The frequency to shift by (in Hz).
                         A positive value shifts up.
                         A negative value shifts down.
        fs (float): The sampling frequency (in Hz).

    Returns:
        np.ndarray: The frequency-shifted complex signal.
    """
    # Create the time/sample index vector
    N = len(signal)
    n = np.arange(N)

    # Create the complex rotator
    # To shift *down* by f_shift, use a negative sign.
    # We'll make the f_shift argument positive for "up" and negative for "down".
    rotator = np.exp(1j * 2 * np.pi * f_shift / fs * n)

    # Apply the shift
    shifted_signal = signal * rotator

    return shifted_signal.astype(signal.dtype)  # Ensure output type matches input


def simple_interp_fir_lowpass(x, fs, band_hz, trans_hz=None, attn_db=60, zero_phase=True):
    """
    Complex-safe 'cheap interpolation' smoother at the original rate.
    Apply this to your complex IQ BEFORE dechirp/FFT.

    Parameters
    ----------
    x : np.ndarray (complex)
        Complex IQ array.
    fs : float
        Sampling rate in Hz.
    band_hz : float
        Highest frequency you want to preserve (low-pass passband edge) in Hz.
        For complex baseband LoRa: band_hz ≈ 0.5*BW (with a small safety margin).
    trans_hz : float or None
        Transition width in Hz. If None, uses 10% of band_hz.
    attn_db : float
        Stopband attenuation target (Kaiser). 50–80 dB is typical.
    zero_phase : bool
        If True, uses filtfilt for zero-phase (no group delay). Otherwise, linear-phase
        with group delay (N-1)/2 samples.

    Returns
    -------
    y : np.ndarray (complex)
        Filtered IQ.
    taps : np.ndarray (float)
        Real FIR taps (apply equally to I and Q).
    """
    x = np.asarray(x)
    assert np.iscomplexobj(x), "x must be complex IQ (dtype can be complex64/complex128)."

    if trans_hz is None:
        trans_hz = 0.10 * band_hz  # gentle default

    # Kaiser design
    width_norm = trans_hz / (fs / 2)  # normalized transition width (0..1)
    N, beta = kaiserord(attn_db, width_norm)
    if N % 2 == 0:  # prefer odd length for symmetric impulse and integer delay
        N += 1

    # Low-pass cutoff at passband edge
    taps = firwin(N, cutoff=band_hz, window=('kaiser', beta), fs=fs, pass_zero='lowpass')

    if zero_phase:
        # filtfilt is complex-safe; gives zero group delay
        y = filtfilt(taps, 1.0, x)
    else:
        # linear-phase with group delay (N-1)/2; keep length same as input
        y = np.convolve(x, taps, mode='same')

    return y, taps


# ---- example usage ----
# rx_iq is your complex noisy signal, fs is sampling rate, BW is LoRa bandwidth
# For LoRa complex baseband: preserve up to half the bandwidth (±BW/2)
# y, taps = interp_kernel_smoother_iq(rx_iq, fs=1_000_000, band_hz=0.5*125_000*0.95)


def precision_interp_fir_lowpass(x,
                                 fs,
                                 band_hz,
                                 trans_hz=None,
                                 attn_db=90,  # try 80–110 dB
                                 method="kaiser", zero_phase=True,
                                 max_taps=4097):
    """
    Apply an interpolation-like low-pass at the ORIGINAL rate (complex-safe).

    Parameters
    ----------
    x : 1D np.ndarray (complex)
        Complex IQ input.
    fs : float
        Sampling rate [Hz].
    band_hz : float
        Passband edge (keep everything up to this). For complex baseband LoRa,
        band_hz ≈ 0.5*BW*0.95 is a good start.
    trans_hz : float or None
        Transition width [Hz]; if None, uses 5% of band_hz.
    attn_db : float
        Target stopband attenuation [dB]. Increase to push spurs down.
    method : {"kaiser","equiripple"}
        Filter design method.
    zero_phase : bool
        If True, use filtfilt for zero group delay.
    max_taps : int
        Safety cap on filter length.

    Returns
    -------
    y : np.ndarray (complex)
        Filtered IQ (same length as x).
    taps : np.ndarray (float)
        Real FIR taps (applied to I and Q).
    """
    x = np.asarray(x)
    if not np.iscomplexobj(x):
        raise ValueError("x must be complex IQ")

    if trans_hz is None:
        trans_hz = 0.05 * band_hz  # tighter than before

    # --- Design filter ---
    if method.lower() == "kaiser":
        # Kaiser order from specs
        width_norm = trans_hz / (fs / 2.0)
        N, beta = kaiserord(attn_db, width_norm)
        # Prefer odd length (nice symmetry; exact zero delay in filtfilt anyway)
        if N % 2 == 0:
            N += 1
        N = min(N, max_taps)
        taps = firwin(
            N,
            cutoff=band_hz,
            window=('kaiser', beta),
            fs=fs,
            pass_zero='lowpass'
        )

    elif method.lower() == "equiripple":
        # Equiripple (Parks–McClellan). Strong stopband weight to approach brickwall.
        # Bands: [0, band_hz] pass; [band_hz+trans_hz, fs/2] stop
        # Leave a clean guard between pass and stop.
        bands = [0.0, band_hz, band_hz + trans_hz, fs / 2.0]
        desired = [1.0, 0.0]
        # Weighting: small passband ripple vs strong stopband attenuation.
        # Tune these two numbers if needed.
        weights = [1.0, 10.0]
        # Start from an estimate (similar to Kaiser) and cap
        width_norm = trans_hz / (fs / 2.0)
        N_est, _ = kaiserord(attn_db, width_norm)
        N = min(N_est | 1, max_taps)  # make odd and cap
        N = max(N, 63)  # avoid too-short filters
        taps = remez(N, bands, desired, weight=weights, fs=fs)

    else:
        raise ValueError("method must be 'kaiser' or 'equiripple'")

    # --- Apply (complex-safe) ---
    if zero_phase:
        y = filtfilt(taps, 1.0, x)
    else:
        # Linear-phase; keep same length
        y = np.convolve(x, taps, mode='same')

    return y, taps


# --------------------------
# Example (drop-in):
# y, taps = interp_like_lowpass_iq(
#     rx_iq, fs=1_000_000,
#     band_hz=0.5*125_000*0.95,   # preserve ±BW/2 (with margin)
#     trans_hz=0.05*125_000,      # ~5% BW transition
#     attn_db=100,                # push stopband hard
#     method="equiripple",        # or "kaiser"
#     zero_phase=True
# )


# Inventory of filter functions
filter_fn_inventory = {
    "simple_fir": simple_interp_fir_lowpass,
    "precision_fir": precision_interp_fir_lowpass,
}

"""
cheap_interp_fir_lowpass: a basic windowed-sinc FIR low-pass using a Kaiser window.
- simpler design, fewer parameters.
- transition width and attenuation are “softer”.
- good for quick smoothing (cheap version of an interpolation kernel).

precision_interp_fir_lowpass: a more flexible low-pass that can be either:
- Kaiser FIR (like above, but you can ask for more stopband attenuation), or
- Equiripple (Parks–McClellan) FIR (sharper cutoff for the same number of taps).
- gives you control over transition width, stopband attenuation, and filter length
"""
