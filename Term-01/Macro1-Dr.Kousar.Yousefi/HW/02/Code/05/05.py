import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# QUESTION 5: AIYAGARI MODEL
# Complete Python implementation
# ============================================================


# ------------------------------------------------------------
# 1. MODEL PARAMETERS
# ------------------------------------------------------------

beta = 0.94
alpha = 0.36
delta = 0.08

# Productivity states:
# State 0 = Low productivity
# State 1 = High productivity

z = np.array([0.5, 1.5])

# Markov transition matrix
P = np.array([[0.90, 0.10], [0.10, 0.90]])

# Asset grid
N_A = 400
A_MIN = 0.0
A_MAX = 30.0

a_grid = np.linspace(A_MIN, A_MAX, N_A)

# Numerical tolerances
V_TOL = 1e-7
MU_TOL = 1e-10
MARKET_TOL = 1e-3

MAX_V_ITER = 5000
MAX_MU_ITER = 20000
MAX_GE_ITER = 60


# ------------------------------------------------------------
# 2. STATIONARY PRODUCTIVITY DISTRIBUTION
# ------------------------------------------------------------


def stationary_productivity():

    # Solve pi = pi P and sum(pi) = 1

    A = np.vstack([P.T - np.eye(2), np.ones(2)])

    b = np.array([0.0, 0.0, 1.0])

    pi = np.linalg.lstsq(A, b, rcond=None)[0]

    return pi


pi = stationary_productivity()

print("Stationary productivity distribution:")
print("Low productivity:", pi[0])
print("High productivity:", pi[1])

average_labor = np.sum(pi * z)

print("Average effective labor:", average_labor)


# ------------------------------------------------------------
# 3. FIRM'S FACTOR PRICES
# ------------------------------------------------------------


def factor_prices(K):

    L = 1.0

    # Marginal product of capital minus depreciation
    r = alpha * (K / L) ** (alpha - 1) - delta

    # Marginal product of labor
    w = (1 - alpha) * (K / L) ** alpha

    return r, w


# ------------------------------------------------------------
# 4. HOUSEHOLD PROBLEM: VALUE FUNCTION ITERATION
# ------------------------------------------------------------


def solve_household(r, w, V_initial=None):

    # Initial value function
    if V_initial is None:
        V = np.zeros((2, N_A))
    else:
        V = V_initial.copy()

    policy_index = np.zeros((2, N_A), dtype=int)

    for iteration in range(MAX_V_ITER):

        V_new = np.empty_like(V)

        for iz in range(2):

            # Expected continuation value
            EV = P[iz, :] @ V

            # Available resources for each current asset
            resources = (1 + r) * a_grid + w * z[iz]

            # Consumption matrix:
            # Rows = current assets
            # Columns = next-period assets
            consumption = resources[:, None] - a_grid[None, :]

            # Infeasible choices receive -infinity
            utility = np.full((N_A, N_A), -np.inf)

            feasible = consumption > 0

            utility[feasible] = np.log(consumption[feasible])

            # Bellman objective
            objective = utility + beta * EV[None, :]

            # Optimal next-period asset choice
            policy_index[iz, :] = np.argmax(objective, axis=1)

            V_new[iz, :] = np.max(objective, axis=1)

        # Convergence criterion
        error = np.max(np.abs(V_new - V))

        V = V_new

        if error < V_TOL:
            break

    # Convert optimal indices into asset levels
    policy_assets = a_grid[policy_index]

    # Consumption policy
    consumption_policy = np.empty((2, N_A))

    for iz in range(2):

        consumption_policy[iz, :] = (1 + r) * a_grid + w * z[iz] - policy_assets[iz, :]

    return {
        "V": V,
        "policy_index": policy_index,
        "policy_assets": policy_assets,
        "consumption_policy": consumption_policy,
        "iterations": iteration + 1,
        "error": error,
    }


# ------------------------------------------------------------
# 5. STATIONARY JOINT DISTRIBUTION
# ------------------------------------------------------------


def stationary_distribution(policy_index):

    # mu[iz, ia]:
    # Probability mass at productivity iz and asset ia

    mu = np.zeros((2, N_A))

    # Initial distribution:
    # All households begin with zero assets.
    # Productivity begins at its stationary distribution.

    mu[:, 0] = pi

    for iteration in range(MAX_MU_ITER):

        mu_new = np.zeros_like(mu)

        # Current productivity state
        for iz in range(2):

            # Current distribution over assets
            current_mass = mu[iz, :]

            # Chosen next-period asset indices
            next_indices = policy_index[iz, :]

            # Next productivity state
            for iz_next in range(2):

                # Probability of transition to iz_next
                transition_probability = P[iz, iz_next]

                # Move probability mass to next assets
                moved_mass = np.bincount(
                    next_indices,
                    weights=(current_mass * transition_probability),
                    minlength=N_A,
                )

                mu_new[iz_next, :] += moved_mass

        # Distribution convergence error
        error = np.max(np.abs(mu_new - mu))

        mu = mu_new

        if error < MU_TOL:
            break

    return {"mu": mu, "iterations": iteration + 1, "error": error}


# ------------------------------------------------------------
# 6. AGGREGATE CAPITAL SUPPLY
# ------------------------------------------------------------


def aggregate_capital(mu):

    # Sum over productivity states
    asset_distribution = np.sum(mu, axis=0)

    # Average household assets
    K_supply = np.sum(a_grid * asset_distribution)

    return K_supply


# ------------------------------------------------------------
# 7. EQUILIBRIUM EVALUATION FOR A GIVEN K
# ------------------------------------------------------------


def evaluate_equilibrium(K, V_initial=None):

    # Factor prices
    r, w = factor_prices(K)

    # Solve household problem
    household = solve_household(r, w, V_initial=V_initial)

    # Obtain asset policy
    policy_index = household["policy_index"]

    # Compute stationary distribution
    distribution = stationary_distribution(policy_index)

    mu = distribution["mu"]

    # Aggregate capital supply
    K_supply = aggregate_capital(mu)

    # Excess supply of capital
    excess_capital = K_supply - K

    return {
        "K": K,
        "K_supply": K_supply,
        "r": r,
        "w": w,
        "household": household,
        "distribution": distribution,
        "excess_capital": excess_capital,
        "market_error": abs(excess_capital),
    }


# ------------------------------------------------------------
# 8. GENERAL EQUILIBRIUM: BISECTION METHOD
# ------------------------------------------------------------


def general_equilibrium(K_low=0.5, K_high=10.0):

    print("\nSearching for general equilibrium...")

    # Evaluate lower endpoint
    result_low = evaluate_equilibrium(K_low)

    excess_low = result_low["excess_capital"]

    # Evaluate upper endpoint
    result_high = evaluate_equilibrium(K_high)

    excess_high = result_high["excess_capital"]

    print(f"Lower bound: K={K_low:.4f}, " f"excess supply={excess_low:.6f}")

    print(f"Upper bound: K={K_high:.4f}, " f"excess supply={excess_high:.6f}")

    # Bisection requires opposite signs
    if excess_low * excess_high > 0:

        raise ValueError(
            "The initial capital interval does not bracket "
            "an equilibrium. Change K_low or K_high. "
            "Also check whether A_MAX is sufficiently large."
        )

    V_initial = None
    result = None

    for outer_iteration in range(MAX_GE_ITER):

        # Midpoint of capital interval
        K_guess = 0.5 * (K_low + K_high)

        # Evaluate equilibrium at this capital
        result = evaluate_equilibrium(K_guess, V_initial=V_initial)

        V_initial = result["household"]["V"]

        excess = result["excess_capital"]

        print(
            f"GE iteration {outer_iteration + 1:2d}: "
            f"K={K_guess:.6f}, "
            f"Ks={result['K_supply']:.6f}, "
            f"excess={excess:.6f}"
        )

        # Check market clearing
        if abs(excess) < MARKET_TOL:
            break

        # Update the bracket
        if excess > 0:

            # Capital supply exceeds capital demand
            K_low = K_guess

        else:

            # Capital supply is below capital demand
            K_high = K_guess

    result["outer_iterations"] = outer_iteration + 1

    return result


# ------------------------------------------------------------
# 9. RUN THE GENERAL EQUILIBRIUM MODEL
# ------------------------------------------------------------

result = general_equilibrium()


# ------------------------------------------------------------
# 10. REPORT EQUILIBRIUM RESULTS
# ------------------------------------------------------------

print("\n")
print("=" * 55)
print("GENERAL EQUILIBRIUM RESULTS")
print("=" * 55)

print(f"Equilibrium capital demand: {result['K']:.8f}")

print(f"Equilibrium capital supply: {result['K_supply']:.8f}")

print(f"Equilibrium interest rate:  {result['r']:.8f}")

print(f"Equilibrium wage:           {result['w']:.8f}")

print(f"Market clearing error:      {result['market_error']:.8e}")


# ------------------------------------------------------------
# 11. REPORT CONVERGENCE ERRORS
# ------------------------------------------------------------

household = result["household"]
distribution = result["distribution"]

print("\n")
print("=" * 55)
print("NUMERICAL CONVERGENCE")
print("=" * 55)

print("Value function iterations:", household["iterations"])

print("Value function error:", f"{household['error']:.8e}")

print("Distribution iterations:", distribution["iterations"])

print("Distribution error:", f"{distribution['error']:.8e}")

mu = distribution["mu"]

print("Sum of probability masses:", f"{np.sum(mu):.12f}")

print("Aggregate capital from distribution:", f"{aggregate_capital(mu):.8f}")


# ------------------------------------------------------------
# 12. CHECK THE UPPER ASSET GRID BOUNDARY
# ------------------------------------------------------------

policy_index = household["policy_index"]

if np.max(policy_index) == N_A - 1:

    print("\nWARNING:")
    print("At least one household chooses the maximum " "asset grid point.")

    print("Increase A_MAX and run the model again.")

else:

    print("\nAsset grid boundary check: passed.")


# ------------------------------------------------------------
# 13. PLOT ASSET POLICY FUNCTIONS
# ------------------------------------------------------------

policy = household["policy_assets"]

plt.figure(figsize=(9, 6))

plt.plot(a_grid, policy[0, :], label="Low productivity (z = 0.5)", linewidth=2)

plt.plot(a_grid, policy[1, :], label="High productivity (z = 1.5)", linewidth=2)

plt.plot(a_grid, a_grid, "--", label="45-degree line")

plt.xlabel("Current assets (a)")
plt.ylabel("Next-period assets (a')")

plt.title("Household Asset Policy Functions")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("question5_policy.png", dpi=300)

plt.show()


# ------------------------------------------------------------
# 14. PLOT STATIONARY WEALTH DISTRIBUTION
# ------------------------------------------------------------

wealth_distribution = np.sum(mu, axis=0)

plt.figure(figsize=(9, 6))

plt.plot(a_grid, wealth_distribution, linewidth=1.8)

plt.xlabel("Assets (a)")
plt.ylabel("Probability mass")

plt.title("Stationary Wealth Distribution")

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("question5_wealth_distribution.png", dpi=300)

plt.show()


# ------------------------------------------------------------
# 15. EULER EQUATION CHECK
# ------------------------------------------------------------


def euler_errors(result):

    household = result["household"]

    r = result["r"]

    c_policy = household["consumption_policy"]

    policy_index = household["policy_index"]

    interior_errors = []
    constrained_violations = []

    for iz in range(2):

        for ia in range(N_A):

            # Current marginal utility
            lhs = 1.0 / c_policy[iz, ia]

            # Chosen next-period asset index
            next_ia = policy_index[iz, ia]

            # Expected next-period marginal utility
            expected_mu = 0.0

            for iz_next in range(2):

                c_next = c_policy[iz_next, next_ia]

                expected_mu += P[iz, iz_next] / c_next

            rhs = beta * (1 + r) * expected_mu

            if next_ia == 0:

                # Borrowing constraint is binding
                violation = max(rhs - lhs, 0.0)

                constrained_violations.append(violation)

            else:

                # Interior choice
                interior_errors.append(abs(lhs - rhs))

    max_interior_error = max(interior_errors) if interior_errors else np.nan

    max_constraint_violation = (
        max(constrained_violations) if constrained_violations else 0.0
    )

    return (max_interior_error, max_constraint_violation)


euler_error, constraint_violation = euler_errors(result)

print("\n")
print("=" * 55)
print("EULER EQUATION CHECK")
print("=" * 55)

print("Maximum interior Euler error:", euler_error)

print("Maximum borrowing constraint violation:", constraint_violation)


# ------------------------------------------------------------
# 16. FINAL MESSAGE
# ------------------------------------------------------------

print("\n")
print("=" * 55)
print("QUESTION 5 COMPLETED")
print("=" * 55)

print("Policy graph saved as: question5_policy.png")

print("Wealth distribution graph saved as: " "question5_wealth_distribution.png")
