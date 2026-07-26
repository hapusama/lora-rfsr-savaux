import numpy as np


def awgn(x, snr_db):
    """
    Add AWGN noise to a signal x to achieve desired SNR in dB.
    """
    # Compute signal power
    if np.iscomplexobj(x):
        signal_power = np.mean(np.abs(x) ** 2)
    else:
        signal_power = np.mean(x ** 2)

    # Compute noise power for desired SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    # Generate noise
    if np.iscomplexobj(x):
        noise = np.sqrt(noise_power / 2) * (np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape))
    else:
        noise = np.sqrt(noise_power) * np.random.randn(*x.shape)

    # Add noise
    y = x + noise
    return y
