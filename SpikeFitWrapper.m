function vp_err = SpikeFitWrapper(threshold)

    % === Assumes global data ===
    global Vm_pred true_spike_times avg_spike dt vp_cost

    % --- Model parameters ---
    tau_ref_ms = 2;  % ms
    [Vm_model, predicted_spike_times] = predict_spike_train_fixed_threshold( ...
        Vm_pred, avg_spike, threshold, tau_ref_ms, dt);

    % --- Call C-accelerated Victor-Purpura distance ---
    vp_err = spkd_c(predicted_spike_times, true_spike_times, ...
        length(predicted_spike_times), length(true_spike_times), vp_cost);

    % Optional diagnostic
    fprintf('✅ Threshold = %.2f → VP distance = %.2f | Spikes: %d vs. %d\n', ...
        threshold, vp_err, length(predicted_spike_times), length(true_spike_times));

end
