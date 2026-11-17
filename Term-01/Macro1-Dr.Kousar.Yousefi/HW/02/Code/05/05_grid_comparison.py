import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# QUESTION 5: GRID SIZE COMPARISON
# N_A = 200, 400, 800
# ============================================================

# 1. Parameters
BETA = 0.94
ALPHA = 0.36
DELTA = 0.08

Z = np.array([0.5, 1.5])

P = np.array([[0.90, 0.10], [0.10, 0.90]])

GRID_SIZES = [200, 400, 800]

A_MIN = 0.0
A_MAX = 30.0

V_TOL = 1e-7
MU_TOL = 1e-10
GE_BRACKET_TOL = 1e-5

MAX_V_ITER = 5000
MAX_MU_ITER = 20000
MAX_GE_ITER = 45

K_LOWER = 4.0
K_UPPER = 6.0

OUT = Path("question5_grid_comparison")
OUT.mkdir(exist_ok=True)


# 2. Stationary productivity distribution
def stationary_productivity():

    A = np.vstack([P.T - np.eye(2), np.ones(2)])

    b = np.array([0.0, 0.0, 1.0])

    return np.linalg.lstsq(A, b, rcond=None)[0]


PI = stationary_productivity()
L = np.dot(PI, Z)


# 3. Factor prices
def factor_prices(K):

    r = ALPHA * (K / L) ** (ALPHA - 1) - DELTA

    w = (1 - ALPHA) * (K / L) ** ALPHA

    return r, w


# 4. Household problem: Value Function Iteration
def solve_household(r, w, a_grid, V_initial=None):

    n = len(a_grid)

    if V_initial is None:
        V = np.zeros((2, n))
    else:
        V = V_initial.copy()

    policy_index = np.zeros((2, n), dtype=int)

    resources = [(1 + r) * a_grid + w * Z[i] for i in range(2)]

    for iteration in range(MAX_V_ITER):

        V_new = np.empty_like(V)

        for iz in range(2):

            EV = P[iz, :] @ V

            consumption = resources[iz][:, None] - a_grid[None, :]

            utility = np.full((n, n), -np.inf)

            feasible = consumption > 0

            utility[feasible] = np.log(consumption[feasible])

            objective = utility + BETA * EV[None, :]

            policy_index[iz] = np.argmax(objective, axis=1)

            V_new[iz] = np.max(objective, axis=1)

        error = np.max(np.abs(V_new - V))

        V = V_new

        if error < V_TOL:
            break

    policy_assets = a_grid[policy_index]

    consumption_policy = np.empty((2, n))

    for iz in range(2):

        consumption_policy[iz] = resources[iz] - policy_assets[iz]

    return {
        "V": V,
        "policy_index": policy_index,
        "policy_assets": policy_assets,
        "consumption_policy": consumption_policy,
        "iterations": iteration + 1,
        "error": error,
    }


# 5. Stationary distribution
def stationary_distribution(policy_index, n):

    mu = np.zeros((2, n))

    mu[:, 0] = PI

    for iteration in range(MAX_MU_ITER):

        mu_new = np.zeros_like(mu)

        for iz in range(2):

            current_mass = mu[iz]

            next_indices = policy_index[iz]

            for iz_next in range(2):

                moved_mass = np.bincount(
                    next_indices, weights=current_mass * P[iz, iz_next], minlength=n
                )

                mu_new[iz_next] += moved_mass

        error = np.max(np.abs(mu_new - mu))

        mu = mu_new

        if error < MU_TOL:
            break

    return {"mu": mu, "iterations": iteration + 1, "error": error}


# 6. Aggregate capital
def aggregate_capital(mu, a_grid):

    asset_mass = np.sum(mu, axis=0)

    return np.dot(a_grid, asset_mass)


# 7. Evaluate equilibrium at a given K
def evaluate(K, a_grid, V_initial=None):

    r, w = factor_prices(K)

    household = solve_household(r, w, a_grid, V_initial)

    distribution = stationary_distribution(household["policy_index"], len(a_grid))

    Ks = aggregate_capital(distribution["mu"], a_grid)

    return {
        "K": K,
        "Ks": Ks,
        "r": r,
        "w": w,
        "excess": Ks - K,
        "market_error": abs(Ks - K),
        "household": household,
        "distribution": distribution,
    }


# 8. General equilibrium: Bisection
def general_equilibrium(a_grid):

    low = K_LOWER
    high = K_UPPER

    left = evaluate(low, a_grid)
    right = evaluate(high, a_grid)

    f_low = left["excess"]
    f_high = right["excess"]

    if f_low * f_high > 0:

        raise RuntimeError(
            "Initial K interval does not bracket equilibrium. "
            "Change K_LOWER or K_UPPER."
        )

    best = left if left["market_error"] < right["market_error"] else right

    V_initial = None

    for it in range(MAX_GE_ITER):

        K = 0.5 * (low + high)

        result = evaluate(K, a_grid, V_initial)

        V_initial = result["household"]["V"]

        f = result["excess"]

        if result["market_error"] < best["market_error"]:
            best = result

        print(
            f"Iteration {it+1:02d}: "
            f"K={K:.7f}, "
            f"Ks={result['Ks']:.7f}, "
            f"excess={f:+.7f}"
        )

        if high - low < GE_BRACKET_TOL:
            break

        if f > 0:
            low = K
        else:
            high = K

    best["GE_iterations"] = it + 1
    best["bracket_width"] = high - low

    return best


# 9. Euler diagnostic
def euler_diagnostic(result):

    r = result["r"]

    c = result["household"]["consumption_policy"]

    policy = result["household"]["policy_index"]

    interior = []
    constrained = []

    for iz in range(2):

        for ia in range(len(c[iz])):

            if c[iz, ia] <= 0:
                continue

            lhs = 1 / c[iz, ia]

            next_ia = policy[iz, ia]

            expected_mu = 0.0

            for j in range(2):

                expected_mu += P[iz, j] / c[j, next_ia]

            rhs = BETA * (1 + r) * expected_mu

            if next_ia == 0:
                constrained.append(max(rhs - lhs, 0))
            else:
                interior.append(abs(lhs - rhs))

    return (
        max(interior) if interior else np.nan,
        max(constrained) if constrained else 0.0,
    )


# 10. Run all grid sizes
results = {}
summary = []

print("Stationary productivity:", PI)
print("Average labor:", L)

for N_A in GRID_SIZES:

    print("\n" + "=" * 65)
    print(f"RUNNING MODEL WITH N_A = {N_A}")
    print("=" * 65)

    a_grid = np.linspace(A_MIN, A_MAX, N_A)

    result = general_equilibrium(a_grid)

    results[N_A] = {"result": result, "grid": a_grid}

    mu = result["distribution"]["mu"]

    wealth = np.sum(mu, axis=0)

    euler_interior, euler_constraint = euler_diagnostic(result)

    upper_hit = np.max(result["household"]["policy_index"]) == N_A - 1

    print("\nRESULTS")
    print("Capital demand:", result["K"])
    print("Capital supply:", result["Ks"])
    print("Interest rate:", result["r"])
    print("Wage:", result["w"])
    print("Market error:", result["market_error"])

    print("VFI iterations:", result["household"]["iterations"])
    print("VFI error:", result["household"]["error"])

    print("Distribution iterations:", result["distribution"]["iterations"])

    print("Distribution error:", result["distribution"]["error"])

    print("Probability sum:", np.sum(mu))

    print("Interior Euler error:", euler_interior)
    print("Constraint violation:", euler_constraint)

    print("Upper asset boundary reached:", upper_hit)

    # Individual policy graph
    policy = result["household"]["policy_assets"]

    plt.figure(figsize=(9, 6))

    plt.plot(a_grid, policy[0], label="Low productivity")
    plt.plot(a_grid, policy[1], label="High productivity")
    plt.plot(a_grid, a_grid, "--", label="45-degree line")

    plt.xlabel("Current assets")
    plt.ylabel("Next-period assets")

    plt.title(f"Asset Policy Functions (N_A={N_A})")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(OUT / f"policy_NA_{N_A}.png", dpi=300)

    plt.close()

    # Individual wealth distribution graph
    plt.figure(figsize=(9, 6))

    plt.plot(a_grid, wealth)

    plt.xlabel("Assets")
    plt.ylabel("Probability mass")

    plt.title(f"Stationary Wealth Distribution (N_A={N_A})")

    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(OUT / f"wealth_distribution_NA_{N_A}.png", dpi=300)

    plt.close()

    summary.append(
        {
            "N_A": N_A,
            "K_demand": result["K"],
            "K_supply": result["Ks"],
            "interest_rate": result["r"],
            "wage": result["w"],
            "market_error": result["market_error"],
            "VFI_error": result["household"]["error"],
            "distribution_error": result["distribution"]["error"],
            "probability_sum": np.sum(mu),
            "Euler_interior_error": euler_interior,
            "Euler_constraint_violation": euler_constraint,
            "upper_grid_boundary_hit": upper_hit,
        }
    )


# 11. Compare low productivity policy functions
plt.figure(figsize=(10, 7))

for N_A in GRID_SIZES:

    item = results[N_A]

    plt.plot(
        item["grid"],
        item["result"]["household"]["policy_assets"][0],
        label=f"N_A={N_A}",
    )

plt.plot(
    results[GRID_SIZES[-1]]["grid"],
    results[GRID_SIZES[-1]]["grid"],
    "k--",
    label="45-degree line",
)

plt.xlabel("Current assets")
plt.ylabel("Next-period assets")

plt.title("Low Productivity: Grid Comparison")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(OUT / "comparison_policy_low.png", dpi=300)

plt.close()


# 12. Compare high productivity policy functions
plt.figure(figsize=(10, 7))

for N_A in GRID_SIZES:

    item = results[N_A]

    plt.plot(
        item["grid"],
        item["result"]["household"]["policy_assets"][1],
        label=f"N_A={N_A}",
    )

plt.plot(
    results[GRID_SIZES[-1]]["grid"],
    results[GRID_SIZES[-1]]["grid"],
    "k--",
    label="45-degree line",
)

plt.xlabel("Current assets")
plt.ylabel("Next-period assets")

plt.title("High Productivity: Grid Comparison")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(OUT / "comparison_policy_high.png", dpi=300)

plt.close()


# 13. Compare wealth distributions
plt.figure(figsize=(10, 7))

for N_A in GRID_SIZES:

    item = results[N_A]

    wealth = np.sum(item["result"]["distribution"]["mu"], axis=0)

    plt.plot(item["grid"], wealth, label=f"N_A={N_A}")

plt.xlabel("Assets")
plt.ylabel("Probability mass")

plt.title("Stationary Wealth Distribution: Grid Comparison")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(OUT / "comparison_wealth_distribution.png", dpi=300)

plt.close()


# 14. Save numerical results in CSV
csv_file = OUT / "grid_comparison_summary.csv"

with open(csv_file, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))

    writer.writeheader()
    writer.writerows(summary)


# 15. Final comparison table
print("\n")
print("=" * 95)
print("FINAL GRID COMPARISON")
print("=" * 95)

print(
    f"{'N_A':>6} {'K demand':>12} {'K supply':>12} "
    f"{'Interest':>12} {'Wage':>12} {'Market error':>15}"
)

for row in summary:

    print(
        f"{row['N_A']:6d} "
        f"{row['K_demand']:12.6f} "
        f"{row['K_supply']:12.6f} "
        f"{row['interest_rate']:12.6f} "
        f"{row['wage']:12.6f} "
        f"{row['market_error']:15.6e}"
    )

print("\nAll graphs and CSV saved in:")
print(OUT.resolve())

print("\nPROGRAM COMPLETED")
