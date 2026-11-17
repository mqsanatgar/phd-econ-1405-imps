%% PART 4: STATIONARY DISTRIBUTION AND SIMULATION
clear;
clc;
close all;

%% 1. Model parameters
alpha = 0.36;
beta  = 0.95;

z = [1.10; 0.90];

P = [0.90 0.10
     0.20 0.80];

s = alpha * beta;

%% 2. Stationary distribution of Markov chain
% We solve:
% pi * P = pi
% pi_H + pi_L = 1

A = [P' - eye(2)
     1 1];

b = [0; 0; 1];

pi_stationary = A \ b;

fprintf('Stationary probability of H = %.6f\n', ...
        pi_stationary(1));

fprintf('Stationary probability of L = %.6f\n', ...
        pi_stationary(2));

%% 3. Theoretical unconditional moments
mean_log_z = sum(pi_stationary .* log(z));

mean_log_k = (log(s) + mean_log_z) / (1 - alpha);

mean_log_y = mean_log_z + alpha * mean_log_k;

fprintf('\nTheoretical moments:\n');

fprintf('E[log(z)] = %.6f\n', mean_log_z);
fprintf('E[log(k)] = %.6f\n', mean_log_k);
fprintf('E[log(y)] = %.6f\n', mean_log_y);

%% 4. Simulation settings
T = 200000;
burn = 1000;

rng(42);

k = zeros(T + 1, 1);
state = zeros(T, 1);
y = zeros(T + 1, 1);

% Initial conditions
k(1) = 0.20;
state(1) = 1;     % 1 = High, 2 = Low

%% 5. Simulate the economy
for t = 1:T

    current_z = z(state(t));

    % Production
    y(t) = current_z * k(t)^alpha;

    % Optimal saving and next-period capital
    k(t + 1) = s * y(t);

    % Draw next productivity state
    u = rand;

    if u <= P(state(t), 1)
        state(t + 1) = 1;
    else
        state(t + 1) = 2;
    end

end

%% 6. Remove burn-in observations
idx = (burn + 1):T;

sim_log_k = log(k(idx));
sim_log_y = log(y(idx));
sim_log_z = log(z(state(idx)));

%% 7. Simulated moments
fprintf('\nSimulated moments:\n');

fprintf('Mean log capital = %.6f\n', mean(sim_log_k));
fprintf('Mean log output = %.6f\n', mean(sim_log_y));
fprintf('Mean log productivity = %.6f\n', mean(sim_log_z));

fraction_H = mean(state(idx) == 1);
fraction_L = mean(state(idx) == 2);

fprintf('Fraction of H = %.6f\n', fraction_H);
fprintf('Fraction of L = %.6f\n', fraction_L);

%% 8. Plot simulated capital
figure;

plot(1:T, k(1:T), 'LineWidth', 1);

xlabel('Time');
ylabel('Capital');
title('Simulated Capital');
grid on;

%% 9. Plot simulated productivity state
figure;

stairs(1:T, z(state(1:T)), 'LineWidth', 1);

xlabel('Time');
ylabel('Productivity z');
title('Simulated Productivity');
grid on;
