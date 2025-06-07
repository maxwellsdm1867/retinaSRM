function D = loss_exp1(Vm, spike_shape, p, dt, true_spikes, tau_ref_ms, cost)
    theta0 = p(1);
    A = p(2);
    tau = p(3);
    [~, spike_pred, ~] = predict_spike_train_exp_single(Vm, spike_shape, theta0, A, tau, tau_ref_ms, dt);
    D = spkd_c(spike_pred, true_spikes, length(spike_pred), length(true_spikes), cost);
end
