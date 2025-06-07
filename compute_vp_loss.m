function vp_dist = compute_vp_loss(params, Vm, dt, avg_spike, tau_ref_ms, elbow_indices, vp_q)
    theta0 = params(1);
    A = params(2);
    tau = params(3);

    [~, ~, ~, predicted_spike_times] = simulate_dynamic_threshold_spikes( ...
        Vm, theta0, A, tau, dt, avg_spike, tau_ref_ms);

    true_spike_times = elbow_indices * dt;

    % --- Call C-accelerated Victor–Purpura distance ---
    vp_dist = spkd_c(predicted_spike_times, true_spike_times, ...
                     length(predicted_spike_times), length(true_spike_times), vp_q);

    % --- Diagnostic print
    fprintf('🔧 θ0 = %.2f | A = %.2f | τ = %.4f → VP = %.4f | Spikes: %d vs. %d\n', ...
        theta0, A, tau, vp_dist, length(predicted_spike_times), length(true_spike_times));
end
