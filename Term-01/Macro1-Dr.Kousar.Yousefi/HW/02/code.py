import numpy as np
import pandas as pd

# ============================================================
# 1. پارامترهای مدل
# ============================================================

alpha = 0.36
beta = 0.95
k0 = 0.22

# محدوده شبکه سرمایه
k_min = 1e-5
k_max = 1.1

# تعداد نقاط شبکه
N = 4001


# ============================================================
# 2. حل تحلیلی با استقرای پسرو
# ============================================================

# b[3] از شرط انتهایی به دست می‌آید
b = np.zeros(4)
a = np.zeros(4)
s_analytical = np.zeros(4)

b[3] = alpha
a[3] = 0.0

# حرکت از دوره 2 به دوره صفر
for t in range(2, -1, -1):

    b[t] = alpha * (1 + beta * b[t + 1])

    s_analytical[t] = beta * b[t + 1] / (1 + beta * b[t + 1])

    a[t] = (
        beta * a[t + 1]
        + np.log(1 - s_analytical[t])
        + beta * b[t + 1] * np.log(s_analytical[t])
    )

# در دوره آخر تمام تولید مصرف می‌شود
s_analytical[3] = 0.0


# ============================================================
# 3. ساخت شبکه سرمایه
# ============================================================

# شبکه یکنواخت
k_grid = np.linspace(k_min, k_max, N)

# اضافه کردن k0 به شبکه، اگر از قبل در آن نباشد
k_grid = np.unique(np.append(k_grid, k0))

n_grid = len(k_grid)


# ============================================================
# 4. حل عددی بلمن با استقرای پسرو
# ============================================================

# V[t, i] = ارزش تابع در دوره t و سرمایه k_grid[i]
V = np.zeros((4, n_grid))

# policy[t, i] = سرمایه بهینه دوره بعد
policy = np.zeros((4, n_grid))

# شرط انتهایی:
# در دوره سوم، تمام تولید مصرف می‌شود.
V[3, :] = alpha * np.log(k_grid)
policy[3, :] = 0.0

# حل دوره‌های 2، 1 و صفر
for t in range(2, -1, -1):

    for i, k in enumerate(k_grid):

        y = k**alpha

        # کنترل‌های مجاز: سرمایه دوره بعد باید کمتر از تولید باشد
        feasible = k_grid[k_grid < y]

        # اگر هیچ کنترل مجازی وجود نداشت، از کوچک‌ترین مقدار شبکه استفاده می‌کنیم
        if len(feasible) == 0:
            feasible = np.array([k_min])

        # مصرف متناظر با هر انتخاب سرمایه
        c = y - feasible

        # تابع هدف بلمن
        objective = np.log(c) + beta * np.interp(feasible, k_grid, V[t + 1, :])

        # انتخاب سرمایه‌ای که تابع هدف را بیشینه می‌کند
        best_index = np.argmax(objective)

        policy[t, i] = feasible[best_index]
        V[t, i] = objective[best_index]


# ============================================================
# 5. شبیه‌سازی مسیر اقتصاد با سیاست عددی
# ============================================================


def simulate_path(policy, k_grid, k0):

    capital = np.zeros(5)
    output = np.zeros(4)
    consumption = np.zeros(4)
    saving_share = np.zeros(4)

    capital[0] = k0

    for t in range(4):

        k = capital[t]
        y = k**alpha

        output[t] = y

        if t == 3:
            # در دوره آخر سرمایه باقی‌مانده ارزشی ندارد
            capital[t + 1] = 0.0
            consumption[t] = y
            saving_share[t] = 0.0

        else:
            # درون‌یابی خطی سیاست روی شبکه
            k_next = np.interp(k, k_grid, policy[t, :])

            capital[t + 1] = k_next
            consumption[t] = y - k_next
            saving_share[t] = k_next / y

    result = pd.DataFrame(
        {
            "t": np.arange(4),
            "k_t": capital[:4],
            "y_t": output,
            "c_t": consumption,
            "k_next": capital[1:],
            "saving_share": saving_share,
        }
    )

    return result


numerical_path = simulate_path(policy, k_grid, k0)


# ============================================================
# 6. محاسبه مسیر تحلیلی
# ============================================================


def analytical_simulation(k0, saving_shares):

    capital = np.zeros(5)
    output = np.zeros(4)
    consumption = np.zeros(4)

    capital[0] = k0

    for t in range(4):

        output[t] = capital[t] ** alpha

        capital[t + 1] = saving_shares[t] * output[t]

        consumption[t] = output[t] - capital[t + 1]

    return pd.DataFrame(
        {
            "t": np.arange(4),
            "k_t": capital[:4],
            "y_t": output,
            "c_t": consumption,
            "k_next": capital[1:],
            "saving_share": saving_shares,
        }
    )


analytical_path = analytical_simulation(k0, s_analytical)


# ============================================================
# 7. نمایش ضرایب و سهم‌های پس‌انداز تحلیلی
# ============================================================

print("\n========== Analytical coefficients ==========")

coefficients = pd.DataFrame(
    {"t": np.arange(4), "a_t": a, "b_t": b, "saving_share": s_analytical}
)

print(coefficients.to_string(index=False))


# ============================================================
# 8. مقایسه مسیر تحلیلی و عددی
# ============================================================

print("\n========== Analytical path ==========")
print(analytical_path.round(8).to_string(index=False))

print("\n========== Numerical path ==========")
print(numerical_path.round(8).to_string(index=False))

comparison = pd.DataFrame(
    {
        "t": np.arange(4),
        "s_analytical": analytical_path["saving_share"],
        "s_numerical": numerical_path["saving_share"],
        "absolute_error": np.abs(
            analytical_path["saving_share"] - numerical_path["saving_share"]
        ),
    }
)

print("\n========== Policy comparison ==========")
print(comparison.round(8).to_string(index=False))

print("\nMaximum absolute policy error:", comparison["absolute_error"].max())


# ============================================================
# 9. بررسی اثر ریزتر شدن شبکه
# ============================================================


def solve_for_grid_size(N):

    grid = np.linspace(k_min, k_max, N)
    grid = np.unique(np.append(grid, k0))

    n = len(grid)

    V_grid = np.zeros((4, n))
    policy_grid = np.zeros((4, n))

    # شرط انتهایی
    V_grid[3, :] = alpha * np.log(grid)
    policy_grid[3, :] = 0.0

    # استقرای پسرو
    for t in range(2, -1, -1):

        for i, k in enumerate(grid):

            y = k**alpha
            feasible = grid[grid < y]

            if len(feasible) == 0:
                feasible = np.array([k_min])

            c = y - feasible

            objective = np.log(c) + beta * np.interp(feasible, grid, V_grid[t + 1, :])

            j = np.argmax(objective)

            policy_grid[t, i] = feasible[j]
            V_grid[t, i] = objective[j]

    # شبیه‌سازی مسیر
    path = simulate_path(policy_grid, grid, k0)

    # خطای سهم پس‌انداز نسبت به جواب تحلیلی
    error = np.max(np.abs(path["saving_share"].to_numpy() - s_analytical))

    return (path["saving_share"].to_numpy(), error)


grid_sizes = [101, 501, 1001, 4001, 10001, 20001, 50001]

grid_results = []

for n in grid_sizes:

    shares, error = solve_for_grid_size(n)

    grid_results.append(
        {
            "N": n,
            "s0": shares[0],
            "s1": shares[1],
            "s2": shares[2],
            "maximum_error": error,
        }
    )

grid_comparison = pd.DataFrame(grid_results)

print("\n========== Grid refinement ==========")
print(grid_comparison.round(8).to_string(index=False))
