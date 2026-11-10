
%% =========================================================
%  12. Local function: solve model on a given grid
% ==========================================================

function [shares,error_value] = solve_grid( ...
    N,k_min,k_max,k0,alpha,beta,s_analytical)

    grid = linspace(k_min,k_max,N)';
    grid = unique([grid;k0]);

    n = length(grid);

    V = zeros(4,n);
    policy = zeros(4,n);

    % Terminal condition
    V(4,:) = alpha*log(grid)';
    policy(4,:) = 0;

    % Backward induction
    for t = 3:-1:1

        for i = 1:n

            k = grid(i);
            y = k^alpha;

            feasible = grid(grid < y);

            if isempty(feasible)
                feasible = k_min;
            end

            c = y-feasible;

            V_next = interp1( ...
                grid,V(t+1,:)',feasible,'linear');

            objective = log(c)+beta*V_next;

            [V(t,i),j] = max(objective);

            policy(t,i) = feasible(j);

        end
    end

    % Forward simulation
    capital = zeros(5,1);
    shares = zeros(4,1);

    capital(1) = k0;

    for t = 1:4

        k = capital(t);
        y = k^alpha;

        if t == 4

            capital(t+1) = 0;
            shares(t) = 0;

        else

            k_next = interp1( ...
                grid,policy(t,:)',k,'linear');

            capital(t+1) = k_next;
            shares(t) = k_next/y;

        end
    end

    error_value = max(abs(shares-s_analytical));

end

