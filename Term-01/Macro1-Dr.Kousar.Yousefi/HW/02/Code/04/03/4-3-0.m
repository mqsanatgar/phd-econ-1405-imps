%% PART 3: VALUE FUNCTION ITERATION (VFI)
clear;
clc;
close all;

%% 1. Model parameters
alpha = 0.36;
beta  = 0.95;

z = [1.10; 0.90];

P = [0.90 0.10
     0.20 0.80];

%% 2. Capital grid
N = 1000;
kmin = 0.02;
kmax = 0.50;

kgrid = linspace(kmin, kmax, N)';

%% 3. Output for each productivity state
Y = zeros(2, N);

for i = 1:2
    Y(i, :) = z(i) .* (kgrid'.^alpha);
end

%% 4. Initial value function
V = zeros(2, N);

Vnew = zeros(2, N);

policy_index = zeros(2, N);

tol = 1e-10;
max_iter = 10000;

%% 5. Value function iteration
for iter = 1:max_iter

    for i = 1:2

        for n = 1:N

            current_output = Y(i, n);

            % Feasible next-period capital
            feasible = (kgrid < current_output);

            kp = kgrid(feasible);

            % Current consumption
            c = current_output - kp;

            % Expected future value
            EV = P(i, 1) * V(1, feasible) ...
               + P(i, 2) * V(2, feasible);

            % Bellman objective
            objective = log(c) + beta * EV;

            % Maximization
            [Vnew(i, n), idx] = max(objective);

            feasible_indices = find(feasible);

            policy_index(i, n) = feasible_indices(idx);

        end
    end

    % Convergence criterion
    error = max(abs(Vnew(:) - V(:)));

    V = Vnew;

    if error < tol
        break
    end

end

%% 6. Optimal policy
kp_policy = kgrid(policy_index);

c_policy = Y - kp_policy';

%% 7. Analytical policy for comparison
s = alpha * beta;

kp_exact = s .* Y;
c_exact  = (1 - s) .* Y;

%% 8. Results
fprintf('Number of iterations = %d\n', iter);
fprintf('Final error = %.12e\n', error);

fprintf('Maximum capital policy error = %.8f\n', ...
        max(abs(kp_policy(:) - kp_exact(:))));

fprintf('Maximum consumption policy error = %.8f\n', ...
        max(abs(c_policy(:) - c_exact(:))));

%% 9. Plot value functions
figure;

plot(kgrid, V(1, :), 'b', 'LineWidth', 2);
hold on;
plot(kgrid, V(2, :), 'r--', 'LineWidth', 2);

xlabel('Capital k');
ylabel('Value Function V(k,z)');
legend('High productivity', 'Low productivity');
title('Value Functions');
grid on;

%% 10. Plot optimal capital policy
figure;

plot(kgrid, kp_policy(1, :), 'b', 'LineWidth', 2);
hold on;
plot(kgrid, kp_policy(2, :), 'r--', 'LineWidth', 2);

xlabel('Current capital k');
ylabel('Next period capital k''');
legend('High productivity', 'Low productivity');
title('Optimal Capital Policy');
grid on;
