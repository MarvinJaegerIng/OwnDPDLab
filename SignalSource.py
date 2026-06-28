import numpy as np


class OFDMSource:
    """
    Single-symbol OFDM signal generator.

    Generates a complex baseband OFDM signal with uniformly distributed
    QAM symbols on the active subcarriers. The signal is normalized to
    unit peak amplitude.

    Parameters
    ----------
    bw : float
        Occupied bandwidth [Hz]. Determines the number of active subcarriers.
    N_samples : int
        Total number of samples (= FFT size = one OFDM symbol).
    fs : float
        Sampling frequency [Hz].
    mod_order : int
        QAM modulation order (e.g. 16, 64, 256).
    noise_floor_dB : float or None
        If set, additive white Gaussian noise is added at this SNR level [dB].
    """

    def __init__(self, bw, N_samples, fs, mod_order=16, noise_floor_dB=None):
        self.fs = fs
        self.N_samples = N_samples
        self.mod_order = mod_order
        self.noise_floor_dB = noise_floor_dB
        self.bw = bw

        self.f_res = fs / N_samples
        self.N_carriers = int(bw / self.f_res)
        self.N_fft = N_samples

    def _random_qam(self, size):
        M = int(np.sqrt(self.mod_order))
        re = np.random.randint(0, M, size) * 2 - (M - 1)
        im = np.random.randint(0, M, size) * 2 - (M - 1)
        return (re + 1j * im) / np.sqrt(2 * (M**2 - 1) / 3)

    def generate(self, seed=None):
        """
        Generate one OFDM symbol.

        Parameters
        ----------
        seed : int or None
            Random seed for reproducibility.

        Returns
        -------
        x : np.ndarray (complex)
            Time-domain baseband signal, peak-normalized to 1.
        """
        if seed is not None:
            np.random.seed(seed)
        X = np.zeros(self.N_fft, dtype=complex)
        start = (self.N_fft - self.N_carriers) // 2
        X[start:start + self.N_carriers] = self._random_qam(self.N_carriers)
        X[self.N_fft // 2] = 0  # Zero DC subcarrier
        x = np.fft.ifft(np.fft.ifftshift(X))
        x = x / np.max(np.abs(x))

        if self.noise_floor_dB is not None:
            sig_power = np.mean(np.abs(x) ** 2)
            noise_power = sig_power * 10 ** (self.noise_floor_dB / 10)
            sigma = np.sqrt(noise_power / 2)
            x = x + sigma * (np.random.randn(len(x)) + 1j * np.random.randn(len(x)))

        return x
