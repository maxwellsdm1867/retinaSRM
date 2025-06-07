%% === predict_spike_train_dynamic ===
function [V_pred_full, spike_times, theta_trace] = predict_spike_train_dynamic(Vm, avg_spike, theta0, weights, centers, width, tau_ref_ms, dt)
    tau_ref_pts = round(tau_ref_ms / 1000 / dt);
    spike_len = length(avg_spike);
    avg_spike = avg_spike - avg_spike(1);

    V_pred_full = Vm;
    theta_trace = zeros(size(Vm));
    spike_times = [];
    t_vec = (0:length(Vm)-1) * dt;

    t = 2;
    while t <= length(Vm)
        theta_t = compute_dynamic_threshold(t_vec(t), predicted_spike_times, centers, tau_ms, theta0, weights);
        theta_trace(t) = theta_t;

        if Vm(t) >= theta_t
            spike_times(end+1) = t;
            eta_start = t;
            eta_end = min(t + spike_len - 1, length(Vm));
            eta_len = eta_end - eta_start + 1;
            V_pred_full(eta_start:eta_end) = V_pred_full(eta_start:eta_end) + avg_spike(1:eta_len)';
            t = eta_end + tau_ref_pts;
        else
            t = t + 1;
        end
    end
    spike_times = spike_times * dt;
end