function dist = evaluate_spkd_loss(params, Vm, avg_spike, dt, centers, width, true_spikes, cost)
    theta0 = params(1);
    weights = params(2:end);
    [~, pred_spikes, ~] = predict_spike_train_dynamic(Vm, avg_spike, theta0, weights, centers, width, 2, dt);
    dist = spkd_c(pred_spikes, true_spikes, length(pred_spikes), length(true_spikes), cost);
end