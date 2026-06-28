import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from abc import ABC, abstractmethod
from scipy.signal import resample_poly, firwin, filtfilt
def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class DPDModel(ABC):
    """Abstract base class for all DPD models."""

    @abstractmethod
    def train(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Train model on PA input x and output y."""
        pass

    def train_oversampled(self, x, y, L=4, **kwargs):
        """Training at L-times rate — model learns without aliasing."""
        x_up = resample_poly(x, L, 1)
        y_up = resample_poly(y, L, 1)
        self.train(x_up, y_up, **kwargs)

    @abstractmethod
    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply trained model to input signal."""
        pass

    def apply_oversampled(self, x: np.ndarray, L: int = 4, fir_order: int = 101) -> np.ndarray:
        """
        Apply DPD at L-times sampling rate, anti-alias filtered.

        Parameters
        ----------
        x : np.ndarray
        L : int — Upsampling factor
        fir_order : int — Anti-alias filter order
        """
        
        N = len(x)
        x_up = resample_poly(x, L, 1)
        y_up = self.apply(x_up)
        h = firwin(fir_order, 1.0 / L)
        y_up = filtfilt(h, 1.0, y_up)
        return resample_poly(y_up, 1, L)[:N]

    def _delay(self, x: np.ndarray, l: int) -> np.ndarray:
        """Return delayed signal x[n-l] with zero padding."""
        N = len(x)
        x_delayed = np.zeros(N, dtype=complex)
        x_delayed[l:] = x[:N - l]
        return x_delayed


class PolynomialDPD(DPDModel):
    """
    Base class for polynomial DPD models (MPM, GMP).

    Subclasses only need to implement _build_basis().
    Training method is selectable via keyword argument.
    """

    def train(self, x: np.ndarray, y: np.ndarray,
              method: str = 'ls', **kwargs) -> None:
        """
        Train model using specified method.

        Parameters
        ----------
        method : str
            'ls'       – Least Squares
            'tikhonov' – Ridge Regression (lam=1e-3)
            'lasso'    – Lasso via ISTA   (lam=1e-3)
            'rls'      – Recursive LS     (lam=0.99)
            'lms'      – Least Mean Sq.   (mu=1e-4)
            'robust'   – Huber via IRLS   (epsilon=1e-2)
        """
        Phi = self._build_basis(x)
        methods = {
            'ls':       lambda: self._train_ls(Phi, y),
            'tikhonov': lambda: self._train_tikhonov(Phi, y, kwargs.get('lam', 1e-3)),
            'lasso':    lambda: self._train_lasso(Phi, y, kwargs.get('lam', 1e-3)),
            'rls':      lambda: self._train_rls(Phi, y, kwargs.get('lam', 0.99)),
            'lms':      lambda: self._train_lms(Phi, y, kwargs.get('mu', 1e-4)),
            'robust':   lambda: self._train_robust(Phi, y, kwargs.get('epsilon', 1e-2)),
        }
        if method not in methods:
            raise ValueError(f"Unknown method '{method}'. "
                             f"Choose from: {list(methods.keys())}")
        self.coef = methods[method]()

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply trained model to input signal."""
        return self._build_basis(x) @ self.coef

    @abstractmethod
    def _build_basis(self, x: np.ndarray) -> np.ndarray:
        pass

    def _train_ls(self, Phi, y):
        w, _, _, _ = np.linalg.lstsq(Phi, y, rcond=None)
        return w

    def _train_tikhonov(self, Phi, y, lam):
        PhiH = Phi.conj().T
        return np.linalg.solve(PhiH @ Phi + lam * np.eye(Phi.shape[1]), PhiH @ y)

    def _train_lasso(self, Phi, y, lam, n_iter=1000):
        M = Phi.shape[1]
        w = np.zeros(M, dtype=complex)
        step = 1.0 / np.real(np.linalg.eigvalsh(Phi.conj().T @ Phi).max())
        for _ in range(n_iter):
            grad  = Phi.conj().T @ (Phi @ w - y)
            w_new = w - step * grad
            w     = (np.maximum(np.abs(w_new) - lam * step, 0)
                     * np.exp(1j * np.angle(w_new)))
        return w

    def _train_rls(self, Phi, y, lam):
        M = Phi.shape[1]
        w = np.zeros(M, dtype=complex)
        P = np.eye(M) / 1e-4
        for n in range(len(y)):
            phi_n = Phi[n].reshape(-1, 1)
            K     = P @ phi_n / (lam + phi_n.conj().T @ P @ phi_n)
            w     = w + (K * (y[n] - phi_n.conj().T @ w)).ravel()
            P     = (P - K @ phi_n.conj().T @ P) / lam
        return w

    def _train_lms(self, Phi, y, mu):
        w = np.zeros(Phi.shape[1], dtype=complex)
        for n in range(len(y)):
            e = y[n] - Phi[n].conj() @ w
            w = w + mu * Phi[n] * np.conj(e)
        return w

    def _train_robust(self, Phi, y, epsilon, n_iter=100):
        w = np.zeros(Phi.shape[1], dtype=complex)
        for _ in range(n_iter):
            r = np.abs(Phi @ w - y)
            W = np.diag(np.where(r < epsilon, 1.0,
                                 epsilon / np.maximum(r, 1e-10)))
            PhiW = Phi.conj().T @ W
            w    = np.linalg.solve(PhiW @ Phi, PhiW @ y)
        return w


class Equalizer(DPDModel):
    """
    Complex-valued FIR equalizer, trained with least squares.

    y_eq[n] = sum_{l=0}^{L} w[l] * x[n-l]

    Parameters
    ----------
    L : int
        Filter length (number of taps - 1).
    """

    def __init__(self, L: int):
        self.L = L
        self.coef = np.zeros(L + 1, dtype=complex)

    def train(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Trains the FIR weights via LS: min ||Phi @ w - y||^2

        Parameters
        ----------
        x : np.ndarray
            Input (measured/distorted signal).
        y : np.ndarray
            Target (desired signal).
        """
        Phi = self._build_basis(x)
        self.coef, _, _, _ = np.linalg.lstsq(Phi, y, rcond=None)

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self._build_basis(x) @ self.coef

    def _build_basis(self, x: np.ndarray) -> np.ndarray:
        return np.column_stack([self._delay(x, l) for l in range(self.L + 1)])

    def impulse_response(self) -> np.ndarray:
        """Returns the trained FIR coefficients."""
        return self.coef.copy()

class MPM(PolynomialDPD):
    """Memory Polynomial Model."""

    def __init__(self, L: int, K: int):
        self.L, self.K = L, K
        self.coef = np.zeros((L + 1) * len(range(1, K + 1, 2)), dtype=complex)

    def _build_basis(self, x: np.ndarray) -> np.ndarray:
        cols = []
        for l in range(self.L + 1):
            x_l = self._delay(x, l)
            for k in range(1, self.K + 1, 2):
                cols.append(x_l * np.abs(x_l) ** (k - 1))
        return np.column_stack(cols)


class GMP(PolynomialDPD):
    """Generalized Memory Polynomial."""

    def __init__(self, Ka: int, La: int,
                       Kb: int, Lb: int, Mb: int,
                       Kc: int, Lc: int, Mc: int):
        self.Ka, self.La = Ka, La
        self.Kb, self.Lb, self.Mb = Kb, Lb, Mb
        self.Kc, self.Lc, self.Mc = Kc, Lc, Mc
        self.coef = None

    def _build_basis(self, x: np.ndarray) -> np.ndarray:
        cols = []
        for l in range(self.La + 1):
            x_l = self._delay(x, l)
            for k in range(1, self.Ka + 1, 2):
                cols.append(x_l * np.abs(x_l) ** (k - 1))
        for l in range(self.Lb + 1):
            x_l = self._delay(x, l)
            for m in range(1, self.Mb + 1):
                x_lead = self._delay(x, max(0, l - m))
                for k in range(1, self.Kb + 1, 2):
                    cols.append(x_l * np.abs(x_lead) ** (k - 1))
        for l in range(self.Lc + 1):
            x_l = self._delay(x, l)
            for m in range(1, self.Mc + 1):
                x_lag = self._delay(x, l + m)
                for k in range(1, self.Kc + 1, 2):
                    cols.append(x_l * np.abs(x_lag) ** (k - 1))
        return np.column_stack(cols)


class _ARVTDNNNet(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int],
                 activation: nn.Module):
        super().__init__()
        layers = []
        in_size = input_size

        # No batch normalization — keep the model as pure RF mathematics.
        # The activation factory is invoked per layer so each layer gets
        # its own module instance (required for stateful activations such as PReLU).
        for h in hidden_sizes:
            layers += [
                nn.Linear(in_size, h),
                activation()
            ]
            in_size = h

        # Output layer: two real-valued outputs (I, Q).
        layers.append(nn.Linear(in_size, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Mapping from string identifiers to torch activation classes.
# Extend this table if additional activations are needed.
_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "silu":       nn.SiLU,
    "relu":       nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu":       nn.GELU,
    "tanh":       nn.Tanh,
    "elu":        nn.ELU,
    "prelu":      nn.PReLU,
    "mish":       nn.Mish,
    "sigmoid":    nn.Sigmoid,
}


def _resolve_activation(activation) -> type[nn.Module]:
    """Resolve an activation specifier into an nn.Module class.

    Accepts either a string key from ``_ACTIVATIONS`` (case-insensitive)
    or an ``nn.Module`` subclass passed directly.
    """
    if isinstance(activation, str):
        key = activation.lower()
        if key not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Available: {sorted(_ACTIVATIONS.keys())}"
            )
        return _ACTIVATIONS[key]
    if isinstance(activation, type) and issubclass(activation, nn.Module):
        return activation
    raise TypeError(
        "activation must be a string identifier or an nn.Module subclass."
    )


class ARVTDNN(DPDModel):
    """Augmented Real-Valued Time Delay Neural Network DPD."""

    def __init__(self, L: int, K: int, hidden_sizes: list[int] = [32, 16],
                 lr: float = 1e-3, epochs: int = 100, batch_size: int = 256,
                 activation="silu"):
        self.L           = L
        self.K           = K  # Number of polynomial terms per tap.
        self.lr          = lr
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.device      = _get_device()
        self.activation  = _resolve_activation(activation)
        in_size = (2 + self.K) * (self.L + 1)

        self.model = _ARVTDNNNet(in_size, hidden_sizes, self.activation).to(self.device)

    def train(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Train via Adam optimizer on MPS/CUDA/CPU."""
        Phi    = self._build_features(x)
        y_real = np.stack([np.real(y), np.imag(y)], axis=1).astype(np.float32)

        loader    = DataLoader(TensorDataset(torch.from_numpy(Phi),
                                             torch.from_numpy(y_real)),
                               batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn   = nn.MSELoss()

        # =====================================================================
        # AUTOPILOT: ReduceLROnPlateau
        #   mode='min'    : monitors the loss for further decrease.
        #   factor=0.5    : halves the learning rate when progress stalls.
        #   patience=8    : tolerates 8 epochs without improvement before acting.
        #   min_lr=1e-7   : hard lower bound to prevent collapse to zero.
        # =====================================================================
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-7
        )

        self.model.train()
        for epoch in range(self.epochs):
            total = sum(
                (lambda X, y: (
                    optimizer.zero_grad(),
                    loss_fn(self.model(X.to(self.device)),
                            y.to(self.device)).backward(),
                    optimizer.step()
                ) and 0 or loss_fn(
                    self.model(X.to(self.device)),
                    y.to(self.device)).item()
                )(X, yb) for X, yb in loader
            )

            # Average loss across all batches in this epoch.
            epoch_loss = total / len(loader)

            # ReduceLROnPlateau requires the monitored metric to be passed
            # explicitly (unlike cosine schedulers, which are step-based).
            scheduler.step(epoch_loss)

            # Status output every 10 epochs, including the current learning rate.
            if (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:>4}/{self.epochs}  "
                      f"loss: {epoch_loss:.12f}  "
                      f"LR: {current_lr:.2e}  "
                      f"device: {self.device}")

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply trained model to input signal."""
        X = torch.from_numpy(self._build_features(x)).to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(X).cpu().numpy()
        return out[:, 0] + 1j * out[:, 1]

    def _build_features(self, x: np.ndarray) -> np.ndarray:
        """Build augmented feature matrix: [I | Q | |x|^1 | |x|^2 | ... | |x|^K]"""
        I, Q, A = [], [], []

        for l in range(self.L + 1):
            x_l = self._delay(x, l)
            I.append(np.real(x_l))
            Q.append(np.imag(x_l))

            # Physical domain knowledge: instantaneous magnitude.
            abs_x_l = np.abs(x_l)

            # Explicit polynomial terms (k = 1, 2, ..., K).
            for k in range(1, self.K + 1):
                A.append(abs_x_l ** k)

        return np.column_stack(I + Q + A).astype(np.float32)

