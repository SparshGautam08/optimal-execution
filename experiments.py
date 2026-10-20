"""Experiments for the optimal-execution project. Prints every table used in
the README and saves the figures to plots/. Runs in a few seconds."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from execution import (Market, optimal_trajectory, cost_and_variance, simulate_shortfall,
                       twap, sell_immediately, almgren_chriss_closed_form)

PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
LAM = 1e-6        # baseline risk aversion (1/$), as in Almgren & Chriss (2000)

# Chart styling: neutral ink for text, thin marks, recessive grid
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
# Diverging blue <-> red with a gray midpoint, for correlation -0.9 ... +0.9
RHO_COLORS = {-0.9: "#1c5cab", -0.5: "#6da7ec", 0.0: "#8a8985", 0.5: "#ec8a89", 0.9: "#c43a3a"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "lines.linewidth": 2, "legend.frameon": False})


def one_asset():
    """Almgren & Chriss (2000) example: 1M shares at $50, 30% annual vol
    (0.95 $/share/sqrt(day)), 5-day horizon, here split into 25 periods."""
    return Market(X=np.array([1e6]), S0=np.array([50.0]), sigma=np.array([0.95]),
                  rho=np.eye(1), H=np.array([[2.5e-6]]), Gamma=np.array([[2.5e-7]]),
                  T=5.0, N=25)


def two_assets(rho, cross=0.0, X=(1e6, 1e6)):
    """Asset A is the same as one_asset(). Asset B is equally volatile but 4x
    less liquid (4x the temporary and permanent impact). `cross` sets the
    cross-impact as a fraction of the geometric mean of the own impacts."""
    eta, gam = np.array([2.5e-6, 1e-5]), np.array([2.5e-7, 1e-6])
    H, G = np.diag(eta), np.diag(gam)
    H[0, 1] = H[1, 0] = cross * np.sqrt(eta[0] * eta[1])
    G[0, 1] = G[1, 0] = cross * np.sqrt(gam[0] * gam[1])
    return Market(X=np.array(X, dtype=float), S0=np.array([50.0, 50.0]),
                  sigma=np.array([0.95, 0.95]), rho=np.array([[1, rho], [rho, 1]]),
                  H=H, Gamma=G, T=5.0, N=25)


def utility(mkt, x, lam=LAM):
    E, V = cost_and_variance(mkt, x)
    return E, np.sqrt(V), E + lam * V


def days(mkt):
    return np.arange(mkt.N + 1) * mkt.tau


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, name), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(PLOT_DIR, exist_ok=True)
    rng = np.random.default_rng(seed=42)
    one = one_asset()

    # ---- 1. Single asset: checks ------------------------------------------
    x_opt = optimal_trajectory(one, LAM)
    gap = np.abs(x_opt - almgren_chriss_closed_form(one, LAM)).max()
    print(f"1. Solver vs Almgren-Chriss closed form: max difference {gap:.1e} shares")

    IS = simulate_shortfall(one, x_opt, 200_000, rng)
    E, sd, _ = utility(one, x_opt)
    se = IS.std(ddof=1) / np.sqrt(len(IS))
    print(f"   Monte Carlo (200,000 paths): mean shortfall ${IS.mean():,.0f} +/- {1.96*se:,.0f} "
          f"(formula ${E:,.0f}); SD ${IS.std(ddof=1):,.0f} (formula ${sd:,.0f})\n")

    # ---- 2. Single asset: strategies --------------------------------------
    print(f"2. Selling 1M shares over 5 days, risk aversion {LAM:g}")
    print(f"   {'Strategy':<20}{'Expected cost':>15}{'Std dev':>12}{'E + lam*Var':>14}")
    for name, x in [("Sell immediately", sell_immediately(one)), ("Even pace (TWAP)", twap(one)),
                    ("Optimal", x_opt)]:
        E, sd, U = utility(one, x)
        print(f"   {name:<20}{E:>15,.0f}{sd:>12,.0f}{U:>14,.0f}")
    print()

    # Figure 1: optimal trajectories for several risk aversions
    fig, ax = plt.subplots(figsize=(7, 4.3))
    lams = [(1e-8, "1e-8", "#86b6ef"), (1e-7, "1e-7", "#5598e7"), (1e-6, "1e-6 (baseline)", "#256abf"),
            (1e-5, "1e-5", "#104281")]
    ax.plot(days(one), twap(one)[:, 0] / 1e3, color=INK2, ls="--", lw=1.5, label="Even pace (TWAP)")
    for lam, label, c in lams:
        ax.plot(days(one), optimal_trajectory(one, lam)[:, 0] / 1e3, "o-", ms=3, color=c,
                label=f"Optimal, risk aversion {label}")
    ax.set_xlabel("Day")
    ax.set_ylabel("Shares still held (thousands)")
    ax.set_title("More risk aversion means selling sooner", color=INK, loc="left")
    ax.legend(fontsize=8)
    save(fig, "1_trajectories_by_risk_aversion.png")

    # Figure 2: efficient frontier
    frontier = np.array([utility(one, optimal_trajectory(one, lam))[:2]
                         for lam in np.logspace(-9, -4, 60)])
    E_tw, sd_tw, _ = utility(one, twap(one))
    E_op, sd_op, _ = utility(one, x_opt)
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(frontier[:, 1] / 1e3, frontier[:, 0] / 1e3, color=BLUE, label="Optimal strategies")
    ax.plot(sd_tw / 1e3, E_tw / 1e3, "s", ms=8, color=INK2, label="Even pace (TWAP)")
    ax.plot(sd_op / 1e3, E_op / 1e3, "o", ms=8, color=ORANGE, label="Optimal at risk aversion 1e-6")
    ax.set_xlabel("Standard deviation of cost ($ thousands)")
    ax.set_ylabel("Expected cost ($ thousands)")
    ax.set_title("Efficient frontier: every point trades cost against risk", color=INK, loc="left")
    ax.set_ylim(0, 3000)
    ax.legend(fontsize=8)
    save(fig, "2_efficient_frontier.png")

    # ---- 3. Two assets: correlation ---------------------------------------
    print("3. Two assets (A liquid, B 4x less liquid), 1M shares each")
    print(f"   {'rho':>5}  {'A held after 1 day':>19}  {'B held after 1 day':>19}  "
          f"{'lowest A':>9}  {'E + lam*Var':>12}")
    traj = {}
    for rho in RHO_COLORS:
        mkt = two_assets(rho)
        x = optimal_trajectory(mkt, LAM)
        traj[rho] = x
        print(f"   {rho:>+5.1f}  {x[5, 0]:>19,.0f}  {x[5, 1]:>19,.0f}  {x[:, 0].min():>9,.0f}  "
              f"{utility(mkt, x)[2]:>12,.0f}")
    print()

    # Figure 3: trajectories of both assets by correlation
    mkt = two_assets(0.0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
    for ax, i, name in [(axes[0], 0, "Asset A (liquid)"), (axes[1], 1, "Asset B (4x less liquid)")]:
        for rho, c in RHO_COLORS.items():
            ax.plot(days(mkt), traj[rho][:, i] / 1e3, "o-", ms=3, color=c, label=f"ρ = {rho:+.1f}")
        ax.axhline(0, color=INK2, lw=1)
        ax.set_title(name, color=INK, loc="left")
        ax.set_xlabel("Day")
    axes[0].set_ylabel("Shares still held (thousands)")
    axes[1].legend(title="Correlation", fontsize=8, title_fontsize=8)
    fig.suptitle("Correlation changes how fast to sell each asset", color=INK, x=0.01, ha="left")
    save(fig, "3_two_assets_by_correlation.png")

    # ---- 4. Hedging with an asset you don't own ---------------------------
    print("4. Holding only B (1M shares), 0 shares of A")
    print(f"   {'rho':>5}  {'peak A position':>15}  {'E + lam*Var':>12}  {'B alone':>12}  {'saved':>7}")
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for rho, c in RHO_COLORS.items():
        mkt = two_assets(rho, X=(0.0, 1e6))
        x = optimal_trajectory(mkt, LAM)
        ax.plot(days(mkt), x[:, 0] / 1e3, "o-", ms=3, color=c, label=f"ρ = {rho:+.1f}")
        # Compare with selling B on its own (never touching A)
        b_only = Market(X=np.array([1e6]), S0=np.array([50.0]), sigma=np.array([0.95]),
                        rho=np.eye(1), H=np.array([[1e-5]]), Gamma=np.array([[1e-6]]), T=5.0, N=25)
        U_b = utility(b_only, optimal_trajectory(b_only, LAM))[2]
        U = utility(mkt, x)[2]
        most_short = x[:, 0].min() if rho >= 0 else x[:, 0].max()
        print(f"   {rho:>+5.1f}  {most_short:>15,.0f}  {U:>12,.0f}  {U_b:>12,.0f}  {max(1 - U / U_b, 0.0):>6.1%}")
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xlabel("Day")
    ax.set_ylabel("Position in asset A (thousands of shares)")
    ax.set_title("Selling B, the optimal strategy hedges with A", color=INK, loc="left")
    ax.legend(title="Correlation", fontsize=8, title_fontsize=8)
    save(fig, "4_hedging_with_asset_A.png")
    print()

    # ---- 5. Cost of ignoring correlation ----------------------------------
    print("5. Cost of planning each asset separately (ignoring correlation)")
    print(f"   {'rho':>5}  {'joint plan':>12}  {'separate plans':>15}  {'extra':>7}")
    for rho in RHO_COLORS:
        mkt = two_assets(rho)
        joint = optimal_trajectory(mkt, LAM)
        separate = optimal_trajectory(two_assets(0.0), LAM)   # planned as if rho = 0
        Uj, Us = utility(mkt, joint)[2], utility(mkt, separate)[2]
        print(f"   {rho:>+5.1f}  {Uj:>12,.0f}  {Us:>15,.0f}  {Us / Uj - 1:>6.1%}")
    print()

    # ---- 6. Cross-asset impact --------------------------------------------
    print("6. Cross-asset impact (rho = 0)")
    print(f"   {'cross':>6}  {'A held after 1 day':>19}  {'B held after 1 day':>19}  {'E + lam*Var':>12}")
    cross_vals = [(0.0, "#86b6ef"), (0.5, "#2a78d6"), (0.9, "#104281")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
    for cross, c in cross_vals:
        mkt = two_assets(0.0, cross)
        x = optimal_trajectory(mkt, LAM)
        print(f"   {cross:>6.1f}  {x[5, 0]:>19,.0f}  {x[5, 1]:>19,.0f}  {utility(mkt, x)[2]:>12,.0f}")
        for ax, i in [(axes[0], 0), (axes[1], 1)]:
            ax.plot(days(mkt), x[:, i] / 1e3, "o-", ms=3, color=c, label=f"cross-impact {cross:.1f}")
    for ax, name in [(axes[0], "Asset A (liquid)"), (axes[1], "Asset B (4x less liquid)")]:
        ax.set_title(name, color=INK, loc="left")
        ax.set_xlabel("Day")
    axes[0].set_ylabel("Shares still held (thousands)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Strong cross-asset impact pushes the two sales apart in time",
                 color=INK, x=0.01, ha="left")
    save(fig, "5_cross_impact.png")
    print(f"Plots saved to {PLOT_DIR}")
