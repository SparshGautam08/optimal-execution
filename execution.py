"""Optimal liquidation of a multi-asset portfolio in the discrete-time
Almgren-Chriss framework, with correlated prices and cross-asset impact.

Setup (m assets, N trading periods of length tau = T / N):
  x_k   holdings (vector) after period k; x_0 = X (start), x_N = 0 (fully sold)
  n_k = x_{k-1} - x_k   shares sold during period k

  Unaffected price change per period: sqrt(tau) * (correlated normal) with
  covariance C per unit time, C_ij = rho_ij * sigma_i * sigma_j.
  Permanent impact: selling n_k lowers all later prices by Gamma @ n_k.
  Temporary impact: period-k trades execute at S_{k-1} - H @ n_k / tau.

  Implementation shortfall IS = X . S_0 - (cash received). Then
    E[IS]   = sum_k n_k . (Gamma @ sum_{j<k} n_j)  +  sum_k n_k . H @ n_k / tau
    Var[IS] = tau * sum_k x_k . C @ x_k
  The optimal strategy minimises E[IS] + lam * Var[IS] (lam = risk aversion).
  Both terms are quadratic in the holdings, so the optimum solves one linear
  system.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Market:
    X: np.ndarray        # starting holdings, shape (m,)
    S0: np.ndarray       # starting prices, shape (m,)
    sigma: np.ndarray    # price volatility, $/share/sqrt(day), shape (m,)
    rho: np.ndarray      # correlation matrix, shape (m, m)
    H: np.ndarray        # temporary impact, ($/share)/(share/day), shape (m, m)
    Gamma: np.ndarray    # permanent impact, ($/share)/share, shape (m, m)
    T: float             # horizon in days
    N: int               # number of trading periods

    @property
    def m(self):
        return len(self.X)

    @property
    def tau(self):
        return self.T / self.N

    @property
    def C(self):
        return self.rho * np.outer(self.sigma, self.sigma)


def _impact_matrix(mkt):
    """Q such that E[IS] = n . Q @ n for the stacked trade vector n (length m*N).
    Diagonal blocks H / tau; block (k, j) with j < k is Gamma (permanent impact
    of earlier trades on the price of later ones)."""
    m, N = mkt.m, mkt.N
    Q = np.zeros((m * N, m * N))
    for k in range(N):
        Q[k*m:(k+1)*m, k*m:(k+1)*m] = mkt.H / mkt.tau
        for j in range(k):
            Q[k*m:(k+1)*m, j*m:(j+1)*m] = mkt.Gamma
    return 0.5 * (Q + Q.T)


def _trade_operator(mkt):
    """n = D @ z + d0, where z = (x_1, ..., x_{N-1}) stacked (free holdings)."""
    m, N = mkt.m, mkt.N
    I = np.eye(m)
    D = np.zeros((m * N, m * (N - 1)))
    d0 = np.zeros(m * N)
    d0[:m] = mkt.X                       # n_1 = x_0 - x_1
    for k in range(N):                   # n_{k+1} = x_k - x_{k+1}
        if k >= 1:
            D[k*m:(k+1)*m, (k-1)*m:k*m] = I
        if k <= N - 2:
            D[k*m:(k+1)*m, k*m:(k+1)*m] = -I
    return D, d0


def optimal_trajectory(mkt, lam):
    """Holdings x_0..x_N (shape (N+1, m)) minimising E[IS] + lam * Var[IS]."""
    m, N = mkt.m, mkt.N
    Q = _impact_matrix(mkt)
    D, d0 = _trade_operator(mkt)
    R = np.kron(np.eye(N - 1), mkt.C) * mkt.tau
    A = D.T @ Q @ D + lam * R
    z = np.linalg.solve(A, -D.T @ Q @ d0)
    return np.vstack([mkt.X, z.reshape(N - 1, m), np.zeros(m)])


def cost_and_variance(mkt, x):
    """Exact E[IS] and Var[IS] of a holdings trajectory x (shape (N+1, m))."""
    n = (x[:-1] - x[1:]).ravel()
    E = n @ _impact_matrix(mkt) @ n
    V = mkt.tau * np.einsum("ki,ij,kj->", x[1:], mkt.C, x[1:])
    return E, V


def simulate_shortfall(mkt, x, n_paths, rng):
    """Monte Carlo implementation shortfall of trajectory x, by simulating the
    price paths and executing the trades. Returns an array of n_paths values."""
    n = x[:-1] - x[1:]                                    # (N, m) trades
    L = np.linalg.cholesky(mkt.C)
    xi = rng.standard_normal((n_paths, mkt.N, mkt.m)) @ L.T * np.sqrt(mkt.tau)
    perm = np.cumsum(n @ mkt.Gamma.T, axis=0)             # cumulative permanent impact
    perm_before = np.vstack([np.zeros(mkt.m), perm[:-1]])  # applied to S_{k-1}
    noise_before = np.cumsum(xi, axis=1) - xi             # noise up to S_{k-1}
    S_prev = mkt.S0 + noise_before - perm_before          # (paths, N, m)
    exec_price = S_prev - (n @ mkt.H.T) / mkt.tau
    cash = np.einsum("pkm,km->p", exec_price, n)
    return mkt.X @ mkt.S0 - cash


def twap(mkt):
    """Sell at a constant rate: holdings fall linearly to zero."""
    frac = 1 - np.arange(mkt.N + 1) / mkt.N
    return np.outer(frac, mkt.X)


def sell_immediately(mkt):
    """Sell everything in the first period."""
    x = np.zeros((mkt.N + 1, mkt.m))
    x[0] = mkt.X
    return x


def almgren_chriss_closed_form(mkt, lam):
    """Single-asset closed form (Almgren & Chriss 2000):
    x_j = sinh(kappa (T - t_j)) / sinh(kappa T) * X,
    with (2 / tau^2)(cosh(kappa tau) - 1) = lam sigma^2 / eta_tilde and
    eta_tilde = eta - gamma tau / 2."""
    eta_t = mkt.H[0, 0] - 0.5 * mkt.Gamma[0, 0] * mkt.tau
    kappa_t2 = lam * mkt.sigma[0] ** 2 / eta_t
    kappa = np.arccosh(1 + 0.5 * kappa_t2 * mkt.tau ** 2) / mkt.tau
    t = np.arange(mkt.N + 1) * mkt.tau
    return (np.sinh(kappa * (mkt.T - t)) / np.sinh(kappa * mkt.T) * mkt.X[0])[:, None]
