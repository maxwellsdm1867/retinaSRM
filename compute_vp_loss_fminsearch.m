function vp_dist = compute_vp_loss_fminsearch(params, Vm_cleaned, dt, avg_spike, tau_ref_ms, elbow_indices, vp_q)
    theta0 = params(1);
    A = params(2);
    tau = params(3);

    % Reject invalid parameters
    if tau <= 0 || A < 0
        vp_dist = Inf;
        return;
    end

    % Run simulation on Vm_cleaned
    [~, ~, ~, predicted_spike_times] = simulate_dynamic_threshold_spikes( ...
        Vm_cleaned, theta0, A, tau, dt, avg_spike, tau_ref_ms);

    true_spike_times = elbow_indices * dt;

    % Compute Victor–Purpura distance
    vp_dist = spkd_c(predicted_spike_times, true_spike_times, ...
                     length(predicted_spike_times), length(true_spike_times), vp_q);

    % Diagnostic print
    fprintf('🔍 fminsearch: θ0 = %.2f | A = %.2f | τ = %.4f → VP = %.4f | Spikes: %d vs. %d\n', ...
        theta0, A, tau, vp_dist, length(predicted_spike_times), length(true_spike_times));
end
