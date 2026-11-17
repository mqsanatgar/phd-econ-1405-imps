# PART 4: STATIONARY DISTRIBUTION AND SIMULATION

import matplotlib.pyplot as plt
import numpy as np

# 1. Model parameters
alpha = 0.36
beta = 0.95

z = np.array([1.10, 0.90])

P = np.array([[0.90, 0.10], [0.20, 0.80]])

s = alpha * beta

# 2. Stationary distribution
# Solve pi * P = pi and sum(pi) = 1

A = np.vstack([P.T - np.eye(2), np.ones(2)])

b = np.array([0, 0, 1])

pi_stationary = np.linalg.lstsq(A, b, rcond=None)[0]

print("Stationary probability of H:", pi_stationary[0])
print("Stationary probability of L:", pi_stationary[1])

# 3. Theoretical unconditional moments
mean_log_z = np.sum(pi_stationary * np.log(z))

mean_log_k = (np.log(s) + mean_log_z) / (1 - alpha)

mean_log_y = mean_log_z + alpha * mean_log_k

print("\nTheoretical moments:")
print("E[log(z)] =", mean_log_z)
print("E[log(k)] =", mean_log_k)
print("E[log(y)] =", mean_log_y)

# 4. Simulation settings
T = 200000
burn = 1000

rng = np.random.default_rng(42)

# IMPORTANT: state has T+1 elements
k = np.zeros(T + 1)
state = np.zeros(T + 1, dtype=int)

# Output is needed for T periods
y = np.zeros(T)

# Initial conditions
k[0] = 0.20
state[0] = 0  # 0 = High, 1 = Low

# 5. Simulate the economy
for t in range(T):

    # Current productivity
    current_z = z[state[t]]

    # Production
    y[t] = current_z * k[t] ** alpha

    # Optimal next-period capital
    k[t + 1] = s * y[t]

    # Next-period productivity state
    state[t + 1] = rng.choice([0, 1], p=P[state[t], :])

# 6. Remove burn-in observations
idx = slice(burn, T)

sim_log_k = np.log(k[idx])
sim_log_y = np.log(y[idx])
sim_log_z = np.log(z[state[idx]])

# 7. Simulated moments
print("\nSimulated moments:")

print("Mean log capital =", np.mean(sim_log_k))
print("Mean log output =", np.mean(sim_log_y))
print("Mean log productivity =", np.mean(sim_log_z))

fraction_H = np.mean(state[idx] == 0)
fraction_L = np.mean(state[idx] == 1)

print("Fraction of H =", fraction_H)
print("Fraction of L =", fraction_L)

# 8. Plot simulated capital
plt.figure()

plt.plot(k[:-1])

plt.xlabel("Time")
plt.ylabel("Capital")
plt.title("Simulated Capital")
plt.grid()
plt.show()

# 9. Plot simulated productivity
plt.figure()

plt.step(np.arange(T), z[state[:-1]], where="post")

plt.xlabel("Time")
plt.ylabel("Productivity z")
plt.title("Simulated Productivity")
plt.grid()
plt.show()


# Plot productivity: first 300 periods

plt.figure(figsize=(12, 5))

plt.step(np.arange(300), z[state[:300]], where="post", linewidth=1.2)

plt.xlabel("Time")
plt.ylabel("Productivity z")
plt.title("Simulated Productivity (First 300 Periods)")

plt.yticks([0.90, 1.10], ["Low (0.90)", "High (1.10)"])

plt.ylim(0.85, 1.15)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# Long-run frequency of high productivity

window = 1000

n_blocks = T // window

high_frequency = np.zeros(n_blocks)

for i in range(n_blocks):

    start = i * window
    end = start + window

    high_frequency[i] = np.mean(state[start:end] == 0)

plt.figure(figsize=(12, 5))

plt.plot(np.arange(n_blocks) * window, high_frequency, linewidth=1)

plt.axhline(2 / 3, linestyle="--", label="Theoretical stationary probability = 2/3")

plt.xlabel("Time")
plt.ylabel("Fraction of High Productivity")
plt.title("Long-run Frequency of High Productivity")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# Plot capital: first 1000 periods

plt.figure(figsize=(12, 5))

plt.plot(np.arange(1000), k[:1000], linewidth=1.2)

plt.xlabel("Time")
plt.ylabel("Capital k")
plt.title("Simulated Capital (First 1000 Periods)")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
