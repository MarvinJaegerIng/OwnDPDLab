import numpy as np
import socket
import struct


class RFSoC:
    """
    Python interface to the RFSoC SDR hardware.

    Handles TCP communication, RF front-end configuration, and
    MLS-synchronized signal acquisition.

    Example usage
    -------------
        rfsoc = RFSoC("192.168.3.1", num_samples=200_000)
        y = rfsoc.measure(x)
        rfsoc.close()
    """

    _CMD_TRANSFER    = 0x01
    _CMD_DAC_VOP     = 0x02
    _CMD_ADC_ATT     = 0x03
    _CMD_DAC_NYQUIST = 0x04
    _CMD_DAC_NCO     = 0x05
    _CMD_ADC_NYQUIST = 0x06
    _CMD_ADC_NCO     = 0x07

    def __init__(self, host="192.168.3.1", port=5555,
                 num_samples=100_000, m_dac=4, m_adc=4,
                 f_clk_dac=245.76e6, f_clk_adc=245.76e6,
                 code_length=12, code_spread=4, sync_oversample=8):
        self.host = host
        self.port = port
        self.num_samples = num_samples
        self.m_dac = m_dac
        self.m_adc = m_adc
        self.f_clk_dac = f_clk_dac
        self.f_clk_adc = f_clk_adc
        self.fs_dac = f_clk_dac * m_dac
        self.fs_adc = f_clk_adc * m_adc
        self.num_in = num_samples * m_dac
        self.num_out = num_samples * m_adc

        self.code_length = code_length
        self.code_spread = code_spread
        self.sync_oversample = sync_oversample

        # Pre-compute MLS preamble (BPSK: ±1)
        from scipy.signal import max_len_seq
        seq, _ = max_len_seq(code_length)
        bpsk = (2 * seq - 1).astype(np.float64)
        self._mls_base = np.repeat(bpsk, code_spread)
        self._preamble_len = len(self._mls_base)
        self._payload_max = self.num_in - self._preamble_len

        self._sock = None
        self._connect()

    # ── Connection ──────────────────────────────────────────

    def _connect(self):
        if self._sock is not None:
            self.close()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((self.host, self.port))

    def close(self):
        if self._sock:
            self._sock.close()
            self._sock = None

    # ── RFDC Control ────────────────────────────────────────

    def set_dac_current(self, vop_uA: int):
        if self._sock is None:
            self._connect()
        self._sock.sendall(bytes([self._CMD_DAC_VOP]))
        self._sock.sendall(struct.pack('<I', int(vop_uA)))
        resp = self._recv_exact(1)
        if resp[0] != 1:
            raise RuntimeError(f"DAC VOP command failed (vop={vop_uA})")
        print(f"DAC Current → {vop_uA} µA")

    def set_adc_attenuation(self, att_dB: float):
        if self._sock is None:
            self._connect()
        self._sock.sendall(bytes([self._CMD_ADC_ATT]))
        self._sock.sendall(struct.pack('<d', float(att_dB)))
        resp = self._recv_exact(1)
        if resp[0] != 1:
            raise RuntimeError(f"ADC attenuation command failed (att={att_dB})")
        print(f"ADC Attenuation → {att_dB} dB")

    def set_dac_nyquist_zone(self, zone: int):
        if zone not in [1, 2]:
            raise ValueError("Nyquist Zone must be 1 or 2.")
        if self._sock is None:
            self._connect()
        self._sock.sendall(bytes([self._CMD_DAC_NYQUIST]))
        self._sock.sendall(struct.pack('<B', int(zone)))
        resp = self._recv_exact(1)
        if resp[0] != 1:
            raise RuntimeError("DAC Nyquist Zone configuration failed.")
        print(f"DAC Nyquist Zone → {zone}")

    def set_dac_nco_frequency(self, freq_mhz: float):
        if self._sock is None:
            self._connect()
        self._sock.sendall(bytes([self._CMD_DAC_NCO]))
        self._sock.sendall(struct.pack('<d', float(freq_mhz)))
        resp = self._recv_exact(1)
        if resp[0] != 1:
            raise RuntimeError("DAC NCO frequency configuration failed.")
        print(f"DAC NCO Frequency → {freq_mhz} MHz")

    def set_adc_nyquist_zone(self, zone: int):
        if zone not in [1, 2]:
            raise ValueError("Nyquist Zone must be 1 or 2.")
        if self._sock is None:
            self._connect()
        self._sock.sendall(bytes([self._CMD_ADC_NYQUIST]))
        self._sock.sendall(struct.pack('<B', int(zone)))
        resp = self._recv_exact(1)
        if resp[0] != 1:
            raise RuntimeError("ADC Nyquist Zone configuration failed.")
        print(f"ADC Nyquist Zone → {zone}")

    def set_adc_nco_frequency(self, freq_mhz: float):
        if self._sock is None:
            self._connect()
        self._sock.sendall(bytes([self._CMD_ADC_NCO]))
        self._sock.sendall(struct.pack('<d', float(freq_mhz)))
        resp = self._recv_exact(1)
        if resp[0] != 1:
            raise RuntimeError("ADC NCO frequency configuration failed.")
        print(f"ADC NCO Frequency → {freq_mhz} MHz")

    def set_power_dBm(self, power_dbm,
                      max_power_dbm=4.5,
                      max_vop_uA=32000,
                      min_vop_uA=2250):
        """
        Set the DAC output power via VOP (Variable Output Power).

        Physics: P ∝ I_FS²  →  I_FS = I_max * sqrt(P / P_max)
        VOP step size: 43.75 µA (per DS926).

        Parameters
        ----------
        power_dbm     : Desired output power at the SMA connector [dBm].
        max_power_dbm : Maximum achievable output power [dBm].
        max_vop_uA    : VOP current at max_power_dbm [µA].
        min_vop_uA    : Minimum allowed VOP current [µA].
        """
        if power_dbm > max_power_dbm:
            raise ValueError(f"Requested {power_dbm:.1f} dBm exceeds max {max_power_dbm} dBm")

        power_ratio = 10 ** ((power_dbm - max_power_dbm) / 10)
        vop = max_vop_uA * np.sqrt(power_ratio)

        # Clip to valid range
        vop = np.clip(vop, min_vop_uA, max_vop_uA)

        # Round to nearest VOP step (43.75 µA per DS926)
        step = 43.75
        vop = round(vop / step) * step

        # Actual level after quantization
        actual_power_dbm = max_power_dbm + 20 * np.log10(vop / max_vop_uA)

        self.set_dac_current(int(vop))
        print(f"Target {power_dbm:.1f} dBm → VOP = {vop:.2f} µA → actual ≈ {actual_power_dbm:.1f} dBm")

    # ── Data Transfer ───────────────────────────────────────

    def transfer(self, x: np.ndarray) -> np.ndarray:
        if self._sock is None:
            self._connect()
        x = np.asarray(x, dtype=np.complex128)

        # Pad or truncate to exact buffer size
        if len(x) < self.num_in:
            x = np.pad(x, (0, self.num_in - len(x)))
        elif len(x) > self.num_in:
            x = x[:self.num_in]

        x_scaled = x * 32767.0
        I = np.real(x_scaled).astype(np.int16)
        Q = np.imag(x_scaled).astype(np.int16)

        self._sock.sendall(bytes([self._CMD_TRANSFER]))
        self._sock.sendall(struct.pack('<I', self.num_in))
        self._sock.sendall(I.tobytes())
        self._sock.sendall(Q.tobytes())

        resp = self._recv_exact(4)
        num_out = struct.unpack('<I', resp)[0]
        if num_out == 0:
            raise RuntimeError("Server reported an error during transfer.")

        nbytes = num_out * 8
        I_rec = np.frombuffer(self._recv_exact(nbytes), dtype=np.float64).copy()
        Q_rec = np.frombuffer(self._recv_exact(nbytes), dtype=np.float64).copy()

        return (I_rec - 1j * Q_rec) / 32767.0

    # ── Synchronized measurement ────────────────────────────

    def measure(self, x: np.ndarray) -> np.ndarray:
        """
        Transmit x and return the time-aligned received signal.

        An MLS preamble is prepended to the payload for sub-sample
        synchronization via cross-correlation.
        """
        x = np.asarray(x, dtype=np.complex128)
        if len(x) > self._payload_max:
            raise ValueError(
                f"Signal too long: {len(x)} > {self._payload_max} "
                f"(num_in={self.num_in} - preamble={self._preamble_len})"
            )

        sig_level = np.mean(np.abs(x))
        if sig_level == 0:
            sig_level = 0.5
        preamble = self._mls_base * 0.9 * sig_level + 0j

        pad_len = self.num_in - self._preamble_len - len(x)
        tx = np.concatenate([preamble, x, np.zeros(pad_len, dtype=np.complex128)])

        y_full = self.transfer(tx)
        y_synced, _ = self._synchronize(preamble, y_full)
        return y_synced[self._preamble_len:self._preamble_len + len(x)]

    def _synchronize(self, ref: np.ndarray, y: np.ndarray):
        L = self.sync_oversample
        ref_up = self._upsample_fft(ref, L)
        y_up   = self._upsample_fft(y, L)

        ref_scaled = ref_up * (np.max(np.abs(y_up)) / np.max(np.abs(ref_up)))

        N = max(len(ref_scaled), len(y_up))
        R = np.fft.fft(ref_scaled, n=N)
        Y = np.fft.fft(y_up, n=N)
        r = np.fft.ifft(Y * np.conj(R))

        idx = np.argmax(np.abs(r))
        lag = idx if idx <= N // 2 else idx - N

        y_sync = np.roll(y_up, -lag)
        phase  = np.angle(r[idx])
        return y_sync * np.exp(-1j * phase), lag

    @staticmethod
    def _upsample_fft(x, L):
        N    = len(x)
        X    = np.fft.fft(x)
        N_up = N * L
        X_up = np.zeros(N_up, dtype=complex)
        X_up[:N // 2] = X[:N // 2]
        X_up[N_up - N // 2:] = X[N // 2:]
        if N % 2 == 0:
            X_up[N // 2]          = X[N // 2] / 2
            X_up[N_up - N // 2]   = X[N // 2] / 2
        return np.fft.ifft(X_up) * L

    # ── Helper methods ───────────────────────────────────────

    def time_axis(self, side="dac") -> np.ndarray:
        if side == "dac":
            return np.arange(self.num_in) / self.fs_dac
        return np.arange(self.num_out) / self.fs_adc

    def freq_axis(self, side="adc") -> np.ndarray:
        if side == "dac":
            return np.linspace(-self.fs_dac / 2, self.fs_dac / 2, self.num_in)
        return np.linspace(-self.fs_adc / 2, self.fs_adc / 2, self.num_out)

    @property
    def payload_max(self):
        return self._payload_max

    # ── Internal ────────────────────────────────────────────

    def _recv_exact(self, nbytes):
        chunks, received = [], 0
        while received < nbytes:
            chunk = self._sock.recv(min(nbytes - received, 65536))
            if not chunk:
                raise ConnectionError("Server closed the connection unexpectedly.")
            chunks.append(chunk)
            received += len(chunk)
        return b''.join(chunks)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        state = "connected" if self._sock else "disconnected"
        return (f"RFSoC({self.host}:{self.port}, "
                f"fs={self.fs_dac/1e6:.1f} MHz, "
                f"preamble={self._preamble_len}, {state})")
