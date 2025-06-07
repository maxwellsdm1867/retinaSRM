function trace = compute_inline_trace(theta0, A, tau)
    trace = theta0 * ones(N,1);
    for i = 1:length(elbow_indices)
        idx = elbow_indices(i);
        decay_len = N - idx + 1;
        t_rel = (0:decay_len-1)' * dt;
        contribution = A * exp(-t_rel / tau);
        trace(idx:end) = trace(idx:end) + contribution;
    end
end