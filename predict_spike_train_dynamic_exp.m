function [V_pred_full, predicted_spike_times, theta_trace] = predict_spike_train_dynamic_exp( ...
    Vm_pred, avg_spike, theta0, weights, tau_ms, tau_ref_ms, dt)

    tau_ref_pts = round(tau_ref_ms / 1000 / dt);
    spike_len = length(avg_spike);
    avg_spike_corrected = avg_spike - avg_spike(1);

    V_pred_full = Vm_pred;
    theta_trace = zeros(size(Vm_pred));
    predicted_spike_times = [];

    spike_history = [];

    for t = 2:length(Vm_pred)
        % Compute dynamic threshold from past spike history
        t_now = t * dt;
        theta_t = theta0;
        for k = 1:length(weights)
            tau = tau_ms(k) / 1000;  % convert ms to s
            deltas = t_now - spike_history;
            deltas = deltas(deltas > 0);
            theta_t = theta_t + weights(k) * sum(exp(-deltas / tau));
        end
        theta_trace(t) = theta_t;

        if Vm_pred(t) >= theta_t
            predicted_spike_times(end+1) = t;
            spike_history(end+1) = t_now;

            eta_start = t;
            eta_end = min(t + spike_len - 1, length(V_pred_full));
            eta_len = eta_end - eta_start + 1;

            V_pred_full(eta_start:eta_end) = ...
                V_pred_full(eta_start:eta_end) + avg_spike_corrected(1:eta_len)';

            t = eta_end + tau_ref_pts;
        end
    end

    predicted_spike_times = predicted_spike_times * dt;
end
