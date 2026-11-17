# PART 3: VALUE FUNCTION ITERATION (VFI)

import matplotlib.pyplot as plt
import numpy as np

# 1. Model parameters
alpha = 0.36
beta = 0.95

z = np.array([1.10, 0.90])

P = np.array([[0.90, 0.10], [0.20, 0.80]])

# 2. Capital grid
N = 1000
kmin = 0.02
kmax = 0.50

kgrid = np.linspace(kmin, kmax, N)

# 3. Output for each productivity state
Y = np.zeros((2, N))

for i in range(2):
    Y[i, :] = z[i] * kgrid**alpha

# 4. Initial value function
V = np.zeros((2, N))

policy_index = np.zeros((2, N), dtype=int)

tol = 1e-10
max_iter = 10000

# 5. Value function iteration
for iteration in range(max_iter):

    V_new = np.empty_like(V)

    for i in range(2):

        for n in range(N):

            current_output = Y[i, n]

            # Feasible next-period capital
            feasible = kgrid < current_output

            kp = kgrid[feasible]

            # Current consumption
            c = current_output - kp

            # Expected future value
            EV = P[i, 0] * V[0, feasible] + P[i, 1] * V[1, feasible]

            # Bellman objective
            objective = np.log(c) + beta * EV

            # Maximization
            idx = np.argmax(objective)

            V_new[i, n] = objective[idx]

            feasible_indices = np.where(feasible)[0]

            policy_index[i, n] = feasible_indices[idx]

    # Convergence criterion
    error = np.max(np.abs(V_new - V))

    V = V_new.copy()

    if error < tol:
        break

# 6. Optimal policy
kp_policy = kgrid[policy_index]

c_policy = Y - kp_policy

# 7. Analytical policy for comparison
s = alpha * beta

kp_exact = s * Y
c_exact = (1 - s) * Y

# 8. Results
print("Number of iterations:", iteration + 1)
print("Final error:", error)

print("Maximum capital policy error:", np.max(np.abs(kp_policy - kp_exact)))

print("Maximum consumption policy error:", np.max(np.abs(c_policy - c_exact)))

# 9. Plot value functions
plt.figure()

plt.plot(kgrid, V[0, :], label="High productivity")
plt.plot(kgrid, V[1, :], "--", label="Low productivity")

plt.xlabel("Capital k")
plt.ylabel("Value Function V(k,z)")
plt.title("Value Functions")
plt.legend()
plt.grid()
plt.show()

# 10. Plot optimal capital policy
plt.figure()

plt.plot(kgrid, kp_policy[0, :], label="High productivity")

plt.plot(kgrid, kp_policy[1, :], "--", label="Low productivity")

plt.xlabel("Current capital k")
plt.ylabel("Next period capital k'")
plt.title("Optimal Capital Policy")
plt.legend()
plt.grid()
plt.show()
