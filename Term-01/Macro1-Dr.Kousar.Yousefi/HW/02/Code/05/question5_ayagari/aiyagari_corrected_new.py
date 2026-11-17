"""
Aiyagari model: continuous (refined-grid) asset choice + lottery distribution
Runs asset grids N_A = 800, 1200, 1600 and reports convergence.

Requirements:
    pip install numpy pandas matplotlib

Model:
    u(c) = log(c)
    beta = 0.94, alpha = 0.36, delta = 0.08
    z = [0.5, 1.5], P = [[0.90, 0.10], [0.10, 0.90]]
    borrowing constraint a' >= 0
    L = E[z] under the stationary productivity distribution (= 1)

The household chooses a' on a refined action grid (4 times denser than
the distribution/state grid). The value function is interpolated linearly
at action-grid points. The stationary distribution uses lottery
interpolation between adjacent points of the state asset grid.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------- Parameters ----------------------------
BETA = 0.94
ALPHA = 0.36
DELTA = 0.08
Z = np.array([0.5, 1.5], dtype=float)
P = np.array([[0.90, 0.10],
              [0.10, 0.90]], dtype=float)
L = 1.0

A_MIN = 0.0
A_MAX = 30.0
GRID_SIZES = [800, 1200, 1600]

# Refined choice grid: number of choice intervals per state-grid interval.
CHOICE_REFINEMENT = 4

VFI_TOL = 1e-7
VFI_MAX_ITER = 2000
DIST_TOL = 1e-10
DIST_MAX_ITER = 20000
GE_TOL = 2e-5
GE_MAX_ITER = 35
K_LOWER = 4.0
K_UPPER = 6.0

OUT = Path("aiyagari_refined_results")
OUT.mkdir(exist_ok=True)


# ----------------------- Production and prices ----------------------
def prices_from_K(K):
    r = ALPHA * (K / L) ** (ALPHA - 1.0) - DELTA
    w = (1.0 - ALPHA) * (K / L) ** ALPHA
    return r, w


def utility(c):
    # Infeasible consumption is assigned a very negative utility.
    return np.where(c > 0.0, np.log(np.maximum(c, 1e-14)), -1e14)


# ---------------------- Household optimization ----------------------
def solve_household(a_grid, a_choice, r, w, V_initial=None):
    """
    V is defined on a_grid. Continuation values are linearly interpolated
    onto the denser a_choice grid.
    """
    n_a = len(a_grid)
    n_choice = len(a_choice)
    n_z = len(Z)

    if V_initial is None:
        V = np.zeros((n_z, n_a))
    else:
        V = V_initial.copy()

    # Interpolate expected continuation value for each current productivity.
    EV_choice = np.empty((n_z, n_choice))
    for iz in range(n_z):
        EV_on_state_grid = P[iz] @ V
        EV_choice[iz] = np.interp(a_choice, a_grid, EV_on_state_grid)

    policy = np.zeros((n_z, n_a))
    converged = False

    for iteration in range(1, VFI_MAX_ITER + 1):
        V_new = np.empty_like(V)

        for iz, z in enumerate(Z):
            cash = w * z + (1.0 + r) * a_grid
            # Chunk over current assets to keep memory bounded.
            chunk = 100
            for start in range(0, n_a, chunk):
                stop = min(start + chunk, n_a)
                c = cash[start:stop, None] - a_choice[None, :]
                RHS = utility(c) + BETA * EV_choice[iz][None, :]
                best_idx = np.argmax(RHS, axis=1)
                V_new[iz, start:stop] = RHS[np.arange(stop-start), best_idx]
                policy[iz, start:stop] = a_choice[best_idx]

        err = np.max(np.abs(V_new - V))
        V = V_new

        if err < VFI_TOL:
            converged = True
            break

        # Update expected continuation values for the next VFI iteration.
        for iz in range(n_z):
            EV_on_state_grid = P[iz] @ V
            EV_choice[iz] = np.interp(a_choice, a_grid, EV_on_state_grid)

    if not converged:
        print(f"  Warning: VFI did not converge; final error={err:.3e}")

    return V, policy, iteration, err


# -------------------- Stationary distribution -----------------------
def stationary_distribution(a_grid, policy, tol=DIST_TOL):
    """
    Lottery interpolation:
    For each (a,z), the chosen a' is split between its two neighboring
    asset-grid nodes so that the conditional mean asset choice is preserved.
    Productivity transitions follow P.
    """
    n_a = len(a_grid)
    n_z = len(Z)
    mu = np.zeros((n_z, n_a))

    # Start with all mass at the lowest asset point and stationary z distribution.
    eigvals, eigvecs = np.linalg.eig(P.T)
    pi_z = np.real(eigvecs[:, np.argmin(np.abs(eigvals - 1.0))])
    pi_z = pi_z / pi_z.sum()
    mu[:, 0] = pi_z

    for it in range(1, DIST_MAX_ITER + 1):
        mu_new = np.zeros_like(mu)

        for iz in range(n_z):
            ap = np.clip(policy[iz], a_grid[0], a_grid[-1])
            hi = np.searchsorted(a_grid, ap, side="right")
            hi = np.clip(hi, 1, n_a - 1)
            lo = hi - 1

            a_lo = a_grid[lo]
            a_hi = a_grid[hi]
            weight_hi = (ap - a_lo) / (a_hi - a_lo)
            weight_lo = 1.0 - weight_hi

            mass = mu[iz]
            for iz_next in range(n_z):
                trans_mass = mass * P[iz, iz_next]
                np.add.at(mu_new[iz_next], lo, trans_mass * weight_lo)
                np.add.at(mu_new[iz_next], hi, trans_mass * weight_hi)

        err = np.max(np.abs(mu_new - mu))
        mu = mu_new
        if err < tol:
            break

    total = mu.sum()
    if total > 0:
        mu /= total

    return mu, it, err


# ---------------------- General equilibrium --------------------------
def evaluate_K(K, a_grid, a_choice, V_initial=None):
    r, w = prices_from_K(K)
    V, policy, vfi_it, vfi_err = solve_household(
        a_grid, a_choice, r, w, V_initial
    )
    mu, dist_it, dist_err = stationary_distribution(a_grid, policy)

    K_supply = np.sum(mu * a_grid[None, :])
    K_demand = K
    excess = K_supply - K_demand

    result = {
        "K": K, "K_demand": K_demand, "K_supply": K_supply,
        "excess": excess, "r": r, "w": w,
        "V": V, "policy": policy, "mu": mu,
        "vfi_iterations": vfi_it, "vfi_error": vfi_err,
        "dist_iterations": dist_it, "dist_error": dist_err
    }
    return result


def solve_equilibrium(a_grid, a_choice):
    # Bisection with best-point tracking; discrete numerical choices can
    # make excess supply only approximately smooth.
    lo, hi = K_LOWER, K_UPPER
    best = None
    V_warm = None

    # Check that the excess-capital function changes sign over the bracket.
    left_check = evaluate_K(lo, a_grid, a_choice, None)
    right_check = evaluate_K(hi, a_grid, a_choice, left_check["V"])
    if left_check["excess"] * right_check["excess"] > 0:
        raise RuntimeError(
            "Initial K bracket does not contain a market-clearing root. "
            f"At K={lo:.4f}, excess={left_check['excess']:+.6f}; "
            f"at K={hi:.4f}, excess={right_check['excess']:+.6f}. "
            "Adjust K_LOWER and K_UPPER."
        )

    # Reuse the left endpoint's value function as a warm start.
    V_warm = left_check["V"]

    for it in range(1, GE_MAX_ITER + 1):
        mid = 0.5 * (lo + hi)
        res = evaluate_K(mid, a_grid, a_choice, V_warm)
        V_warm = res["V"]
        if best is None or abs(res["excess"]) < abs(best["excess"]):
            best = res

        print(
            f"    GE {it:02d}: K={mid:.7f}, "
            f"Ksup={res['K_supply']:.7f}, excess={res['excess']:+.3e}"
        )

        if abs(res["excess"]) < GE_TOL or (hi - lo) < 1e-5:
            break

        # Correct bisection direction:
        # excess > 0 means K_supply > K_demand, so raise K.
        # excess < 0 means K_supply < K_demand, so lower K.
        if res["excess"] > 0:
            lo = mid
        else:
            hi = mid

    # Re-evaluate best point from scratch for internally consistent outputs.
    final = evaluate_K(best["K"], a_grid, a_choice, best["V"])
    return final, it


# ----------------------------- Diagnostics ---------------------------
def euler_diagnostics(a_grid, policy, r, w):
    """
    Approximate Euler residual at the selected next-asset policy.
    The next-period marginal utility is interpolated over the asset grid.
    This is a diagnostic, not an exact continuous-choice FOC at grid kinks.
    """
    n_z, n_a = policy.shape
    residuals = []
    constraint_violations = []

    for iz, z in enumerate(Z):
        ap = policy[iz]
        c = w*z + (1+r)*a_grid - ap
        for ia in range(n_a):
            if c[ia] <= 0:
                continue
            expected_muc = 0.0
            for iz_next, z_next in enumerate(Z):
                ap_next = np.interp(ap[ia], a_grid, policy[iz_next])
                c_next = w*z_next + (1+r)*ap[ia] - ap_next
                if c_next > 0:
                    expected_muc += P[iz, iz_next] / c_next

            lhs = 1.0 / c[ia]
            rhs = BETA * (1+r) * expected_muc
            if ap[ia] > A_MIN + 1e-8:
                residuals.append(abs(lhs-rhs) / max(lhs, 1e-12))
            else:
                constraint_violations.append(max(rhs-lhs, 0.0) / max(lhs, 1e-12))

    return (
        float(np.mean(residuals)) if residuals else np.nan,
        float(np.mean(constraint_violations)) if constraint_violations else np.nan
    )


# -------------------------------- Main -------------------------------
def main():
    summaries = []

    for n_a in GRID_SIZES:
        print("\n" + "="*70)
        print(f"Solving model with N_A={n_a}")
        a_grid = np.linspace(A_MIN, A_MAX, n_a)
        n_choice = (n_a - 1) * CHOICE_REFINEMENT + 1
        a_choice = np.linspace(A_MIN, A_MAX, n_choice)

        eq, ge_iterations = solve_equilibrium(a_grid, a_choice)
        r, w = eq["r"], eq["w"]
        euler_int, euler_con = euler_diagnostics(a_grid, eq["policy"], r, w)

        mass_sum = eq["mu"].sum()
        upper_hit = bool(np.any(eq["policy"] >= A_MAX - 1e-10))

        row = {
            "N_A": n_a,
            "N_choice": n_choice,
            "K_demand": eq["K_demand"],
            "K_supply": eq["K_supply"],
            "market_error": abs(eq["excess"]),
            "interest_rate": r,
            "wage": w,
            "VFI_iterations": eq["vfi_iterations"],
            "VFI_error": eq["vfi_error"],
            "distribution_iterations": eq["dist_iterations"],
            "distribution_error": eq["dist_error"],
            "probability_sum": mass_sum,
            "Euler_interior_error": euler_int,
            "Euler_constraint_violation": euler_con,
            "upper_grid_boundary_hit": upper_hit,
            "GE_iterations": ge_iterations
        }
        summaries.append(row)

        # Policy function
        plt.figure(figsize=(8, 5))
        for iz, z in enumerate(Z):
            plt.plot(a_grid, eq["policy"][iz], label=f"z={z}")
        plt.plot(a_grid, a_grid, "k--", linewidth=1, label="45-degree line")
        plt.xlabel("Current assets, a")
        plt.ylabel("Next-period assets, a'")
        plt.title(f"Policy function (N_A={n_a})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT / f"policy_NA_{n_a}.png", dpi=160)
        plt.close()

        # Wealth distribution
        wealth_pdf = eq["mu"].sum(axis=0)
        plt.figure(figsize=(8, 5))
        plt.plot(a_grid, wealth_pdf)
        plt.xlabel("Assets")
        plt.ylabel("Probability mass at grid point")
        plt.title(f"Stationary wealth distribution (N_A={n_a})")
        plt.tight_layout()
        plt.savefig(OUT / f"wealth_distribution_NA_{n_a}.png", dpi=160)
        plt.close()

        print(
            f"  RESULT: K={eq['K_demand']:.8f}, K_supply={eq['K_supply']:.8f}, "
            f"|error|={abs(eq['excess']):.3e}, r={r:.8f}, w={w:.8f}"
        )

    df = pd.DataFrame(summaries)
    df.to_csv(OUT / "grid_comparison_summary.csv", index=False)

    # Overlay policy functions for the two productivity states.
    for iz, z in enumerate(Z):
        plt.figure(figsize=(8, 5))
        for row in summaries:
            n_a = int(row["N_A"])
            # Re-solve only to recover policy arrays for overlay would be wasteful;
            # the individual policy files are saved above.
        plt.title(f"Policy functions are saved separately for z={z}")
        plt.xlabel("Current assets, a")
        plt.ylabel("Next-period assets, a'")
        plt.tight_layout()
        plt.close()

    print("\nSummary:")
    print(df.to_string(index=False))
    print(f"\nAll outputs saved in: {OUT.resolve()}")


if __name__ == "__main__":
    main()
