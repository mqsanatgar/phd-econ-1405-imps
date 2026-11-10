

clc;
clear;
close all;

%% =========================================================
%  QUESTION 4: STOCHASTIC GROWTH MODEL
% ==========================================================

%% 1. Parameters

alpha = 0.36;
beta  = 0.95;

zL = 0.90;
zH = 1.10;

z = [zL; zH];

% Markov transition matrix
% Rows: current state
% Columns: next-period state

P = [0.90 0.10;
     0.20 0.80];

%% =========================================================
% 2. Analytical solution
% ==========================================================

% Analytical saving share
s = alpha * beta;

% Coefficient of log capital
b = alpha / (1-alpha*beta);

% Constant term in Bellman equation
q = log(1-s) + beta*b*log(s);

% Solve for aL and aH:
% (I-beta*P)*a = q*ones + (1+beta*b)*log(z)

A_matrix = eye(2)-beta*P;

rhs = q*ones(2,1) + (1+beta*b)*log(z);

a = A_matrix\rhs;

fprintf('\n========== ANALYTICAL SOLUTION ==========\n');

fprintf('Saving share = %.8f\n',s);
fprintf('Consumption share = %.8f\n',1-s);
fprintf('b = %.8f\n',b);

fprintf('aL = %.8f\n',a(1));
fprintf('aH = %.8f\n',a(2));

%% =========================================================
% 3. Analytical policy functions
% ==========================================================

% k_next = s*z*k^alpha
% c = (1-s)*z*k^alpha

fprintf('\nOptimal capital policy:\n');
fprintf('k_next = %.4f*z*k^alpha\n',s);

fprintf('\nOptimal consumption policy:\n');
fprintf('c = %.4f*z*k^alpha\n',1-s);

%% =========================================================
% 4. Numerical solution using Value Function Iteration
% ==========================================================

% Capital grid

N = 1000;

k_min = 0.0001;
k_max = 1.5;

k_grid = linspace(k_min,k_max,N)';

% Initial value functions:
% Column 1: Low productivity
% Column 2: High productivity

V = zeros(N,2);

% Policy functions
policy = zeros(N,2);

% Convergence parameters

tol = 1e-10;
max_iter = 10000;

distance = inf;
iteration = 0;

while distance > tol && iteration < max_iter

    iteration = iteration + 1;

    V_new = zeros(N,2);
    policy_new = zeros(N,2);

    for iz = 1:2

        for ik = 1:N

            k = k_grid(ik);

            % Current output
            y = z(iz)*k^alpha;

            % Feasible next-period capital
            feasible = k_grid(k_grid < y);

            % Consumption
            c = y-feasible;

            % Expected continuation value
            V_L_next = interp1( ...
                k_grid,V(:,1),feasible,'linear');

            V_H_next = interp1( ...
                k_grid,V(:,2),feasible,'linear');

            expected_V = ...
                P(iz,1)*V_L_next + ...
                P(iz,2)*V_H_next;

            % Bellman objective
            objective = log(c)+beta*expected_V;

            % Maximization
            [V_new(ik,iz),j] = max(objective);

            % Store optimal policy
            policy_new(ik,iz) = feasible(j);

        end
    end

    % Convergence criterion
    distance = max(abs(V_new(:)-V(:)));

    V = V_new;
    policy = policy_new;

end

fprintf('\n========== VALUE FUNCTION ITERATION ==========\n');

fprintf('Number of iterations = %d\n',iteration);
fprintf('Final convergence error = %.12f\n',distance);


%% =========================================================
% 5. Compare numerical and analytical policies
% ==========================================================

policy_exact = zeros(N,2);

for iz = 1:2

    policy_exact(:,iz) = ...
        s*z(iz)*k_grid.^alpha;

end

policy_error = abs(policy-policy_exact);

max_error_L = max(policy_error(:,1));
max_error_H = max(policy_error(:,2));

fprintf('\n========== POLICY ERRORS ==========\n');

fprintf('Maximum error in Low state = %.10f\n',max_error_L);
fprintf('Maximum error in High state = %.10f\n',max_error_H);

% Saving shares on the grid

saving_numerical = policy ./ ...
    ([zL*k_grid.^alpha, zH*k_grid.^alpha]);

saving_exact = s*ones(N,2);


%% =========================================================
% 6. Plot numerical and analytical policies
% ==========================================================

figure;

subplot(1,2,1);

plot(k_grid,policy(:,1),'b','LineWidth',2);
hold on;

plot(k_grid,policy_exact(:,1),'r--','LineWidth',2);

xlabel('Capital k');
ylabel('Next period capital k''');

title('Low productivity state');

legend('Numerical','Analytical','Location','best');
grid on;


subplot(1,2,2);

plot(k_grid,policy(:,2),'b','LineWidth',2);
hold on;

plot(k_grid,policy_exact(:,2),'r--','LineWidth',2);

xlabel('Capital k');
ylabel('Next period capital k''');

title('High productivity state');

legend('Numerical','Analytical','Location','best');
grid on;


%% =========================================================
% 7. Stationary distribution of Markov chain
% ==========================================================

A_stationary = P'-eye(2);

% Replace one equation with sum of probabilities = 1
A_stationary(2,:) = [1 1];

rhs_stationary = [0;1];

pi = A_stationary\rhs_stationary;

fprintf('\n========== STATIONARY DISTRIBUTION ==========\n');

fprintf('Probability of Low state = %.8f\n',pi(1));
fprintf('Probability of High state = %.8f\n',pi(2));


%% =========================================================
% 8. Theoretical long-run moments
% ==========================================================

E_log_z = pi(1)*log(zL)+pi(2)*log(zH);

E_log_k = (log(s)+E_log_z)/(1-alpha);

E_log_y = E_log_z+alpha*E_log_k;

geometric_mean_k = exp(E_log_k);
geometric_mean_y = exp(E_log_y);

fprintf('\n========== THEORETICAL LONG-RUN VALUES ==========\n');

fprintf('Mean log productivity = %.8f\n',E_log_z);
fprintf('Mean log capital = %.8f\n',E_log_k);
fprintf('Mean log output = %.8f\n',E_log_y);

fprintf('Capital exp(E[log k]) = %.8f\n', ...
    geometric_mean_k);

fprintf('Output exp(E[log y]) = %.8f\n', ...
    geometric_mean_y);


%% =========================================================
% 9. Simulate the Markov chain and economy
% ==========================================================

rng(12345);

T = 200000;
burn = 10000;

% State 1 = Low
% State 2 = High

state = zeros(T+1,1);

% Initial state
state(1) = 2;

capital = zeros(T+1,1);
output = zeros(T,1);
consumption = zeros(T,1);

% Initial capital
capital(1) = geometric_mean_k;

for t = 1:T

    % Current productivity
    z_current = z(state(t));

    % Current output
    output(t) = z_current*capital(t)^alpha;

    % Numerical policy
    k_next = interp1( ...
        k_grid,policy(:,state(t)), ...
        capital(t),'linear');

    % Keep next capital feasible
    k_next = max(k_min,min(k_next,output(t)*(1-1e-12)));

    % Consumption
    consumption(t) = output(t)-k_next;

    % Next-period capital
    capital(t+1) = k_next;

    % Draw next productivity state
    u = rand;

    if u <= P(state(t),1)
        state(t+1) = 1;
    else
        state(t+1) = 2;
    end

end

% Remove burn-in observations

sample = burn+1:T;

log_k_sim = log(capital(sample));
log_y_sim = log(output(sample));

frequency_H = mean(state(sample)==2);

fprintf('\n========== SIMULATION RESULTS ==========\n');

fprintf('Simulated mean log capital = %.8f\n', ...
    mean(log_k_sim));

fprintf('Simulated mean log output = %.8f\n', ...
    mean(log_y_sim));

fprintf('Simulated frequency of High state = %.8f\n', ...
    frequency_H);


%% =========================================================
% 10. Compare simulated and theoretical moments
% ==========================================================

moment_table = table( ...
    [E_log_k; E_log_y; pi(2)], ...
    [mean(log_k_sim); mean(log_y_sim); frequency_H], ...
    'VariableNames', ...
    {'Theoretical','Simulated'}, ...
    'RowNames', ...
    {'Mean_log_capital', ...
     'Mean_log_output', ...
     'High_state_frequency'});

fprintf('\n========== MOMENT COMPARISON ==========\n');

disp(moment_table);


%% =========================================================
% 11. Conditional expected deviations after H at t=0
% ==========================================================

% Initial productivity is High
% Initial capital is at stationary mean log capital

T_IRF = 20;

time = (0:T_IRF)';

% Conditional expected productivity deviation
delta_z = (2/3)*log(zH/zL)*0.7.^time;

% Conditional expected capital deviation

delta_k = zeros(T_IRF+1,1);

for t = 1:T_IRF

    delta_k(t+1) = ...
        delta_z(t)+alpha*delta_k(t);

end

% Conditional expected output deviation

delta_y = delta_z+alpha*delta_k;

IRF_table = table( ...
    time,delta_z,delta_k,delta_y, ...
    'VariableNames', ...
    {'t','Delta_log_z','Delta_log_k','Delta_log_y'});

fprintf('\n========== CONDITIONAL EXPECTED DEVIATIONS ==========\n');

disp(IRF_table);


%% =========================================================
% 12. Plot conditional expected deviations
% ==========================================================

figure;

plot(time,delta_k,'b-o','LineWidth',2);
hold on;

plot(time,delta_y,'r-s','LineWidth',2);

xlabel('Time');
ylabel('Conditional expected log deviation');

title('Conditional Response to Initial High Productivity');

legend('Capital','Output','Location','best');

grid on;


%% =========================================================
% 13. Plot simulated capital and output
% ==========================================================

figure;

plot(1:1000,capital(1:1000),'LineWidth',1.5);

xlabel('Time');
ylabel('Capital');

title('Simulated Capital Path');

grid on;


figure;

plot(1:1000,output(1:1000),'LineWidth',1.5);

xlabel('Time');
ylabel('Output');

title('Simulated Output Path');

grid on;


%% =========================================================
% 14. Plot simulated productivity states
% ==========================================================

figure;

stairs(0:999,state(1:1000),'LineWidth',1.5);

xlabel('Time');
ylabel('Productivity state');

yticks([1 2]);
yticklabels({'Low','High'});

title('Simulated Markov Productivity');

grid on;
