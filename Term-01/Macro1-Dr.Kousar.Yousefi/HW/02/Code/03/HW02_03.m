
clc;
clear;
close all;

%% =========================================================
%  1. Model parameters
% ==========================================================

alpha = 0.36;
beta  = 0.95;
k0    = 0.22;

k_min = 1e-5;
k_max = 1.1;

N = 4001;


%% =========================================================
%  2. Analytical solution using backward induction
% ==========================================================

a = zeros(4,1);
b = zeros(4,1);
s_analytical = zeros(4,1);

% Terminal condition
b(4) = alpha;
a(4) = 0;

% Backward induction: t = 2, 1, 0
for t = 3:-1:1

    b(t) = alpha * (1 + beta*b(t+1));

    s_analytical(t) = ...
        beta*b(t+1)/(1 + beta*b(t+1));

    a(t) = beta*a(t+1) ...
        + log(1-s_analytical(t)) ...
        + beta*b(t+1)*log(s_analytical(t));

end

% In the final period, all output is consumed
s_analytical(4) = 0;


%% =========================================================
%  3. Construct capital grid
% ==========================================================

k_grid = linspace(k_min,k_max,N)';

% Add initial capital to the grid
k_grid = unique([k_grid; k0]);

n_grid = length(k_grid);


%% =========================================================
%  4. Numerical solution of Bellman equation
% ==========================================================

V = zeros(4,n_grid);
policy = zeros(4,n_grid);

% Terminal value function:
% V_3(k) = alpha*log(k)

V(4,:) = alpha*log(k_grid)';
policy(4,:) = 0;

% Backward induction
for t = 3:-1:1

    for i = 1:n_grid

        k = k_grid(i);
        y = k^alpha;

        % Feasible next-period capital choices
        feasible = k_grid(k_grid < y);

        if isempty(feasible)
            feasible = k_min;
        end

        % Consumption
        c = y - feasible;

        % Interpolation of next-period value function
        V_next = interp1( ...
            k_grid, V(t+1,:)', feasible, 'linear');

        % Bellman objective
        objective = log(c) + beta*V_next;

        % Optimal choice
        [V(t,i), best_index] = max(objective);

        policy(t,i) = feasible(best_index);

    end
end


%% =========================================================
%  5. Simulate numerical optimal path
% ==========================================================

capital_num = zeros(5,1);
output_num = zeros(4,1);
consumption_num = zeros(4,1);
saving_num = zeros(4,1);

capital_num(1) = k0;

for t = 1:4

    k = capital_num(t);
    y = k^alpha;

    output_num(t) = y;

    if t == 4

        % No terminal value of capital
        capital_num(t+1) = 0;
        consumption_num(t) = y;
        saving_num(t) = 0;

    else

        % Interpolate the optimal policy
        k_next = interp1( ...
            k_grid, policy(t,:)', k, 'linear');

        capital_num(t+1) = k_next;
        consumption_num(t) = y-k_next;
        saving_num(t) = k_next/y;

    end
end

numerical_path = table( ...
    (0:3)', ...
    capital_num(1:4), ...
    output_num, ...
    consumption_num, ...
    capital_num(2:5), ...
    saving_num, ...
    'VariableNames', ...
    {'t','k_t','y_t','c_t','k_next','saving_share'});


%% =========================================================
%  6. Analytical optimal path
% ==========================================================

capital_ana = zeros(5,1);
output_ana = zeros(4,1);
consumption_ana = zeros(4,1);

capital_ana(1) = k0;

for t = 1:4

    output_ana(t) = capital_ana(t)^alpha;

    capital_ana(t+1) = ...
        s_analytical(t)*output_ana(t);

    consumption_ana(t) = ...
        output_ana(t)-capital_ana(t+1);

end

analytical_path = table( ...
    (0:3)', ...
    capital_ana(1:4), ...
    output_ana, ...
    consumption_ana, ...
    capital_ana(2:5), ...
    s_analytical, ...
    'VariableNames', ...
    {'t','k_t','y_t','c_t','k_next','saving_share'});


%% =========================================================
%  7. Display analytical coefficients
% ==========================================================

disp(' ');
disp('========== Analytical coefficients ==========');

coefficients = table( ...
    (0:3)', a, b, s_analytical, ...
    'VariableNames', ...
    {'t','a_t','b_t','saving_share'});

disp(coefficients);


%% =========================================================
%  8. Compare analytical and numerical paths
% ==========================================================

disp(' ');
disp('========== Analytical path ==========');
disp(analytical_path);

disp(' ');
disp('========== Numerical path ==========');
disp(numerical_path);

absolute_error = abs( ...
    analytical_path.saving_share ...
    - numerical_path.saving_share);

comparison = table( ...
    (0:3)', ...
    analytical_path.saving_share, ...
    numerical_path.saving_share, ...
    absolute_error, ...
    'VariableNames', ...
    {'t','s_analytical','s_numerical','absolute_error'});

disp(' ');
disp('========== Policy comparison ==========');
disp(comparison);

fprintf('\nMaximum absolute policy error = %.10f\n', ...
    max(absolute_error));


%% =========================================================
%  9. Grid refinement analysis
% ==========================================================

grid_sizes = [101, 501, 1001, 4001, 10001];

grid_results = zeros(length(grid_sizes),5);

for g = 1:length(grid_sizes)

    Ng = grid_sizes(g);

    [shares, error_value] = solve_grid( ...
        Ng, k_min, k_max, k0, alpha, beta, ...
        s_analytical);

    grid_results(g,:) = [ ...
        Ng, shares(1), shares(2), ...
        shares(3), error_value];

end

grid_comparison = array2table( ...
    grid_results, ...
    'VariableNames', ...
    {'N','s0','s1','s2','maximum_error'});

disp(' ');
disp('========== Grid refinement ==========');
disp(grid_comparison);


%% =========================================================
%  10. Plot saving shares
% ==========================================================

figure;

plot(0:3, s_analytical, 'o-', ...
    'LineWidth',2,'MarkerSize',7);

hold on;

plot(0:3, saving_num, 's--', ...
    'LineWidth',2,'MarkerSize',7);

xlabel('Time period');
ylabel('Saving share');
title('Analytical vs Numerical Saving Policy');

legend('Analytical','Numerical','Location','best');
grid on;


%% =========================================================
%  11. Plot grid refinement error
% ==========================================================

figure;

loglog(grid_comparison.N, ...
    grid_comparison.maximum_error, ...
    'o-','LineWidth',2,'MarkerSize',7);

xlabel('Number of grid points');
ylabel('Maximum absolute policy error');
title('Effect of Grid Refinement on Numerical Error');

grid on;


