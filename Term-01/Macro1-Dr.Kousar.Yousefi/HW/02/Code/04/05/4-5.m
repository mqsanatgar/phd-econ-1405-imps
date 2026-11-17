%% PART 5: CONDITIONAL EXPECTED PATH
clear;
clc;
close all;

%% 1. Model parameters
alpha = 0.36;
beta  = 0.95;

zH = 1.10;
zL = 0.90;

P = [0.90 0.10
     0.20 0.80];

s = alpha * beta;

%% 2. Stationary probabilities
piH = 2 / 3;
piL = 1 / 3;

%% 3. Long-run mean values in logs
mean_log_z = piH * log(zH) + piL * log(zL);

mean_log_k = (log(s) + mean_log_z) / (1 - alpha);

mean_log_y = mean_log_z + alpha * mean_log_k;

%% 4. Initial condition
% Initial productivity is high
z0 = zH;

% Initial capital is the geometric mean of stationary capital
k0 = exp(mean_log_k);

% Initial deviations
zhat0 = log(z0) - mean_log_z;

xhat0 = 0;

fprintf('Initial productivity = %.4f\n', z0);
fprintf('Initial capital = %.6f\n', k0);

%% 5. Time horizon
T = 10;

t = (0:T)';

%% 6. Conditional probability of high productivity
prob_H = piH + (1 - piH) * (0.70.^t);

%% 7. Expected productivity deviation
E_zhat = zhat0 * (0.70.^t);

%% 8. Expected capital deviation
E_xhat = zeros(T + 1, 1);

for h = 0:T

    if h == 0
        E_xhat(h + 1) = 0;
    else

        E_xhat(h + 1) = zhat0 * ...
            ((0.70^h - alpha^h) / (0.70 - alpha));

    end

end

%% 9. Expected output deviation
E_yhat = E_zhat + alpha * E_xhat;

%% 10. Display results
Results = table( ...
                t, prob_H, E_zhat, E_xhat, E_yhat, ...
                'VariableNames', ...
                {'Time', 'Prob_H', 'Expected_z_Deviation', ...
                 'Expected_k_Deviation', 'Expected_y_Deviation'});

disp(Results);

%% 11. Plot conditional probability
figure;

plot(t, prob_H, 'o-', 'LineWidth', 2);

xlabel('Time');
ylabel('Probability of High Productivity');
title('Conditional Probability of High Productivity');
grid on;

%% 12. Plot expected deviations
figure;

plot(t, E_zhat, 'o-', 'LineWidth', 2);
hold on;
plot(t, E_xhat, 's-', 'LineWidth', 2);
plot(t, E_yhat, 'd-', 'LineWidth', 2);

xlabel('Time');
ylabel('Expected Log Deviation');
legend('Productivity', 'Capital', 'Output');
title('Conditional Expected Paths');
grid on;
