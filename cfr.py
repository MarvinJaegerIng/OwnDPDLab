import numpy as np


def calculate_papr(x: np.ndarray) -> float:
    """Return the Peak-to-Average Power Ratio of x in dB."""
    peak_power = np.max(np.abs(x) ** 2)
    avg_power  = np.mean(np.abs(x) ** 2)
    if avg_power == 0:
        return 0.0
    return 10 * np.log10(peak_power / avg_power)


def hard_clip(x: np.ndarray, threshold: float = 1.0,
              clip_value: float = 0.99, verbose: bool = True) -> np.ndarray:
    """
    CFR: Hard amplitude clipping while preserving phase.

    All samples whose magnitude exceeds `threshold` are clipped to `clip_value`.

    Parameters
    ----------
    x          : Complex baseband input signal.
    threshold  : Clipping threshold (default: 1.0).
    clip_value : Clipped amplitude value (default: 0.99).
    verbose    : Print PAPR before/after when True.
    """
    x_clipped = np.copy(x)
    clip_mask  = np.abs(x_clipped) > threshold
    peaks_over = np.sum(clip_mask)

    if peaks_over > 0:
        if verbose:
            papr_before = calculate_papr(x)
            print(f"[CFR - Hard] Clipping {peaks_over} peaks above {threshold}.")

        x_clipped[clip_mask] = (x_clipped[clip_mask] / np.abs(x_clipped[clip_mask])) * clip_value

        if verbose:
            papr_after = calculate_papr(x_clipped)
            print(f"       PAPR: {papr_before:.2f} dB -> {papr_after:.2f} dB")

    return x_clipped


def soft_clip(x: np.ndarray, threshold: float = 0.8,
              max_limit: float = 1.0, verbose: bool = True) -> np.ndarray:
    """
    CFR: Soft amplitude clipping.

    Signal is passed through linearly up to `threshold`. Above that, the
    amplitude is smoothly compressed toward `max_limit` using an asymptotic
    curve, producing less spectral regrowth than hard clipping.

    Parameters
    ----------
    x          : Complex baseband input signal.
    threshold  : Linear pass-through range (default: 0.8).
    max_limit  : Asymptotic amplitude ceiling, never exceeded (default: 1.0).
    verbose    : Print PAPR before/after when True.
    """
    amp   = np.abs(x)
    phase = np.angle(x)

    mask       = amp > threshold
    peaks_over = np.sum(mask)

    amp_clipped = np.copy(amp)
    if peaks_over > 0:
        if verbose:
            papr_before = calculate_papr(x)
            print(f"[CFR - Soft] Smoothly compressing {peaks_over} peaks above {threshold}.")

        # Soft-clip formula: y = limit - (limit - thresh)^2 / (x - thresh + limit - thresh)
        diff = max_limit - threshold
        amp_clipped[mask] = max_limit - (diff ** 2) / (amp[mask] - threshold + diff)

    x_clipped = amp_clipped * np.exp(1j * phase)

    if peaks_over > 0 and verbose:
        papr_after = calculate_papr(x_clipped)
        print(f"       PAPR: {papr_before:.2f} dB -> {papr_after:.2f} dB")

    return x_clipped
