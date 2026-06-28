"""
Metrics for evaluating DPD linearization performance.

Usage
-----
    import Metrics as met

    print(f"NMSE:  {met.nmse(x, y):.2f} dB")
    print(f"EVM:   {met.evm(x, y):.2f} %")
    print(f"ACPR:  {met.acpr(y, bw=20e6, fs=200e6):.2f} dBc")
    print(f"OOB:   {met.oob_nmse(x, y, bw=20e6, fs=200e6):.2f} dB")
"""

import numpy as np


def nmse(x: np.ndarray, y: np.ndarray) -> float:
    """
    Normalized Mean Squared Error in dB.

    NMSE = 10·log10( mean(|x - y|²) / mean(|x|²) )
    """
    return float(10 * np.log10(np.mean(np.abs(x - y) ** 2) / np.mean(np.abs(x) ** 2)))


def evm(x: np.ndarray, y: np.ndarray) -> float:
    """
    Error Vector Magnitude in percent.

    EVM = sqrt( mean(|x - y|²) / mean(|x|²) ) × 100
    """
    return float(np.sqrt(np.mean(np.abs(x - y) ** 2) / np.mean(np.abs(x) ** 2)) * 100)


def oob_nmse(x: np.ndarray, y: np.ndarray, bw: float, fs: float) -> float:
    """
    Out-of-Band Normalized Mean Squared Error in dB.

    Measures the residual error energy outside the signal bandwidth,
    normalized to the ideal in-band energy. Useful for quantifying
    spectral regrowth suppression.

    Parameters
    ----------
    x  : Ideal (reference) signal.
    y  : Received / pre-distorted signal.
    bw : Signal bandwidth [Hz].
    fs : Sampling frequency [Hz].
    """
    N = len(x)
    f = np.fft.fftshift(np.fft.fftfreq(N, 1 / fs))

    E = np.abs(np.fft.fftshift(np.fft.fft(x - y))) ** 2
    X = np.abs(np.fft.fftshift(np.fft.fft(x))) ** 2

    in_band_mask  = np.abs(f) <= bw / 2
    out_band_mask = ~in_band_mask

    P_ib_ref  = np.sum(X[in_band_mask])   # Ideal in-band power
    P_oob_err = np.sum(E[out_band_mask])  # Out-of-band error power

    if P_oob_err == 0:
        return -np.inf
    return float(10 * np.log10(P_oob_err / P_ib_ref))


def acpr(y: np.ndarray, bw: float, fs: float,
         offset: float = None, adj_bw: float = None,
         apply_window: bool = True) -> float:
    """
    Adjacent Channel Power Ratio (ACPR) in dBc.

    Follows standard RF measurement practice:
      1. Optional Hann windowing to reduce spectral leakage.
      2. Evaluates both lower and upper adjacent channels.
      3. Returns the worst-case (maximum) value.

    Parameters
    ----------
    y            : Time-domain complex baseband signal.
    bw           : Main channel bandwidth [Hz].
    fs           : Sampling frequency [Hz].
    offset       : Adjacent channel center offset [Hz]. Defaults to bw.
    adj_bw       : Adjacent channel bandwidth [Hz]. Defaults to bw.
    apply_window : Apply Hann window before FFT (default: True).

    Returns
    -------
    float : Worst-case ACPR in dBc.
    """
    if offset is None:
        offset = bw
    if adj_bw is None:
        adj_bw = bw

    N = len(y)
    if N == 0:
        return -np.inf

    # 1. Windowing: prevents FFT leakage from artificially degrading ACPR
    if apply_window:
        y = y * np.hanning(N)

    # 2. Transform to frequency domain
    f        = np.fft.fftshift(np.fft.fftfreq(N, 1 / fs))
    Y_mag_sq = np.abs(np.fft.fftshift(np.fft.fft(y))) ** 2

    # 3. Main channel power
    ch_mask = np.abs(f) <= bw / 2
    P_ch    = np.sum(Y_mag_sq[ch_mask])

    # 4. Adjacent channel power (upper and lower)
    upper_mask = (f >= offset - adj_bw / 2) & (f <= offset + adj_bw / 2)
    lower_mask = (f >= -offset - adj_bw / 2) & (f <= -offset + adj_bw / 2)
    P_adj = max(np.sum(Y_mag_sq[upper_mask]), np.sum(Y_mag_sq[lower_mask]))

    # 5. Numerical stability
    if P_ch <= 0:
        return np.inf
    if P_adj <= 0:
        return -np.inf

    return float(10 * np.log10(P_adj / P_ch))
