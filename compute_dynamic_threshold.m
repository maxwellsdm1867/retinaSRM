function theta_val = compute_dynamic_threshold(t_now, spike_times, centers, tau_ms, theta0, weights)
    % Compute dynamic threshold using exponential basis functions
    theta_val = theta0;
    for j = 1:length(weights)
        delays = t_now - spike_times;
        valid = delays > 0;

        % Exponential basis: exp(-delay / tau)
        B = exp(-delays(valid) * 1000 / tau_ms(j));
        theta_val = theta_val + weights(j) * sum(B);
    end
end
