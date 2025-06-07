function theta_trace = compute_dynamic_threshold_trace(spike_inds, theta0, A, tau, dt, N)
    theta_trace = theta0 * ones(N,1);
    
    for i = 1:length(spike_inds)
        idx = spike_inds(i);
        decay_len = N - idx + 1;
        t_rel = (0:decay_len-1)' * dt;
        contribution = A * exp(-t_rel / tau);
        theta_trace(idx:end) = theta_trace(idx:end) + contribution;
    end
end
