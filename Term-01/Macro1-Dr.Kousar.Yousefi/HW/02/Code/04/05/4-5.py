# PART 5: CONDITIONAL EXPECTED PATH

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Model parameters
alpha = 0.36
beta = 0.95

zH = 1.10
zL = 0.90

P = np.array([[0.90, 0.10], [0.20, 0.80]])

s = alpha * beta

# 2. Stationary probabilities
piH = 2 / 3
piL = 1 / 3

# 3. Long-run mean values in logs
mean_log_z = piH * np.log(zH) + piL * np.log(zL)

mean_log_k = (np.log(s) + mean_log_z) / (1 - alpha)

mean_log_y = mean_log_z + alpha * mean_log_k

# 4. Initial condition
# Initial productivity is high
z0 = zH

# Initial capital is geometric mean of stationary capital
k0 = np.exp(mean_log_k)

# Initial deviations
zhat0 = np.log(z0) - mean_log_z

xhat0 = 0.0

print("Initial productivity:", z0)
print("Initial capital:", k0)

# 5. Time horizon
T = 10

t = np.arange(T + 1)

# 6. Conditional probability of high productivity
prob_H = piH + (1 - piH) * (0.70**t)

# 7. Expected productivity deviation
E_zhat = zhat0 * (0.70**t)

# 8. Expected capital deviation
E_xhat = np.zeros(T + 1)

for h in range(T + 1):

    if h == 0:
        E_xhat[h] = 0.0

    else:
        E_xhat[h] = zhat0 * ((0.70**h - alpha**h) / (0.70 - alpha))

# 9. Expected output deviation
E_yhat = E_zhat + alpha * E_xhat

# 10. Display results
results = pd.DataFrame(
    {
        "Time": t,
        "Prob_H": prob_H,
        "Expected_z_Deviation": E_zhat,
        "Expected_k_Deviation": E_xhat,
        "Expected_y_Deviation": E_yhat,
    }
)

print("\nConditional expected paths:")
print(results.to_string(index=False))

# 11. Plot conditional probability
plt.figure()

plt.plot(t, prob_H, "o-")

plt.xlabel("Time")
plt.ylabel("Probability of High Productivity")
plt.title("Conditional Probability of High Productivity")
plt.grid()
plt.show()

# 12. Plot expected deviations
plt.figure()

plt.plot(t, E_zhat, "o-", label="Productivity")
plt.plot(t, E_xhat, "s-", label="Capital")
plt.plot(t, E_yhat, "d-", label="Output")

plt.xlabel("Time")
plt.ylabel("Expected Log Deviation")
plt.title("Conditional Expected Paths")
plt.legend()
plt.grid()
plt.show()
