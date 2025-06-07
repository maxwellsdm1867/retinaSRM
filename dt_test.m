%% === Optimization Settings
init = [-50, 2, 0.005, 0.02];  % [theta0, A, T_rise, tau_decay]
vp_q = 4;

loss_fn = @(params) compute_vp_loss_piecewise(params, ...
    Vm_cleaned, dt, avg_spike_short, tau_ref_ms, elbow_indices, vp_q);

options = optimset('Display', 'iter', 'MaxFunEvals', 5000, 'MaxIter', 1000);
[opt_params, vp_loss_val] = fminsearch(loss_fn, init, options);

theta0_opt = opt_params(1);
A_opt = opt_params(2);
T_rise_opt = opt_params(3);
tau_decay_opt = opt_params(4);

fprintf('\n🎯 Final fit:\n');
fprintf('Theta0     = %.3f mV\n', theta0_opt);
fprintf('A          = %.3f mV\n', A_opt);
fprintf('T_rise     = %.4f s\n', T_rise_opt);
fprintf('Tau_decay  = %.4f s\n', tau_decay_opt);

%% === Simulate with optimized parameters
kernel_piecewise = @(t) (t < T_rise_opt) .* (A_opt / T_rise_opt .* t) + ...
                        (t >= T_rise_opt) .* (A_opt * exp(-(t - T_rise_opt) / tau_decay_opt));

[spikes_opt, theta_trace_opt, V_pred_opt, spike_times_opt] = ...
    simulate_dynamic_threshold_generic(Vm_cleaned, theta0_opt, dt, avg_spike_short, tau_ref_ms, kernel_piecewise);

%% === Compare to data
t = (0:length(Vm_all)-1) * dt;
true_spike_times = elbow_indices * dt;
vp_dist_final = spkd_c(spike_times_opt, true_spike_times, ...
                       length(spike_times_opt), length(true_spike_times), vp_q);

zoom_start = 1; zoom_end = 3;
zoom_mask = (t >= zoom_start & t <= zoom_end);

%% === Plotting
figure;

% Subplot 1: Vm_all, V_pred, theta
subplot(3,1,1);
plot(t(zoom_mask), Vm_all(zoom_mask), 'b'); hold on;
plot(t(zoom_mask), V_pred_opt(zoom_mask), 'r');
plot(t(zoom_mask), theta_trace_opt(zoom_mask), 'k--', 'LineWidth', 1.2);
ylabel('Voltage (mV)');
legend('Vm\_all', 'V\_pred', '\theta');
title(sprintf('Zoomed Voltage Trace (%.1f–%.1f s)', zoom_start, zoom_end));
grid on;

% Subplot 2: Threshold comparison
subplot(3,1,2);
plot(t(zoom_mask), threshold_interp(zoom_mask), 'b'); hold on;
plot(t(zoom_mask), theta_trace_opt(zoom_mask), 'r--', 'LineWidth', 1.2);
elbow_times = elbow_indices * dt;
elbow_mask = elbow_times >= zoom_start & elbow_times <= zoom_end;
plot(elbow_times(elbow_mask), threshold_values(elbow_mask), ...
     'go', 'MarkerSize', 6, 'MarkerFaceColor', 'g');
ylabel('Threshold (mV)');
legend('\theta (data interp)', '\theta (model)', 'Spike Initiations');
title('Threshold Comparison');
grid on;

% Subplot 3: ISI histograms
isi_true = diff(elbow_indices) * dt * 1000;
isi_pred = diff(find(spikes_opt)) * dt * 1000;
edges = 0:2:100;

subplot(3,1,3);
histogram(isi_true, edges, 'FaceColor', 'b', 'FaceAlpha', 0.5); hold on;
histogram(isi_pred, edges, 'FaceColor', 'r', 'FaceAlpha', 0.5);
xlabel('ISI (ms)'); ylabel('Count');
title('ISI Distribution: True vs. Predicted');
legend('True ISI', 'Predicted ISI');
grid on;

sgtitle(sprintf('VP-Optimized Piecewise Threshold — VP Distance (q = %.1f): %.3f', vp_q, vp_dist_final));


%% === Local Function
function loss = compute_vp_loss_piecewise(params, Vm, dt, avg_spike, tau_ref_ms, elbow_indices, vp_q)
    theta0 = params(1);
    A = params(2);
    T_rise = params(3);
    tau_decay = params(4);

    kernel = @(t) (t < T_rise) .* (A / T_rise .* t) + ...
                  (t >= T_rise) .* (A * exp(-(t - T_rise) / tau_decay));

    [~, ~, ~, spike_times] = simulate_dynamic_threshold_generic( ...
        Vm, theta0, dt, avg_spike, tau_ref_ms, kernel);

    true_spike_times = elbow_indices * dt;
    loss = spkd_c(spike_times, true_spike_times, ...
                  length(spike_times), length(true_spike_times), vp_q);

    fprintf('θ = %.2f | A = %.2f | T_rise = %.3f | τ_decay = %.3f → VP = %.3f | spikes = %d vs %d\n', ...
        theta0, A, T_rise, tau_decay, loss, length(spike_times), length(true_spike_times));
end


function [spikes, threshold_trace, V_pred_full, predicted_spike_times_sec] = ...
    simulate_dynamic_threshold_generic(Vm, theta0, dt, avg_spike, tau_ref_ms, kernel_fn)

    N = length(Vm);
    tau_ref_pts = round(tau_ref_ms / 1000 / dt);
    spike_len = length(avg_spike);
    avg_spike_corrected = avg_spike - avg_spike(1);

    spikes = zeros(N,1);
    threshold_trace = theta0 * ones(N,1);
    V_pred_full = Vm;
    predicted_spike_times = [];

    t = 2;
    next_allowed_spike_time = 1;

    while t <= N
        if t < next_allowed_spike_time
            t = t + 1;
            continue;
        end

        threshold = threshold_trace(t);
        if V_pred_full(t) >= threshold && V_pred_full(t-1) < threshold
            spikes(t) = 1;
            predicted_spike_times(end+1) = t;

            % Inject spike waveform
            eta_start = t;
            eta_end = min(t + spike_len - 1, N);
            eta_len = eta_end - eta_start + 1;
            V_pred_full(eta_start:eta_end) = ...
                V_pred_full(eta_start:eta_end) + avg_spike_corrected(1:eta_len)';

            % Add kernel to threshold
            decay_len = N - t + 1;
            t_rel = (0:decay_len-1)' * dt;
            kernel = kernel_fn(t_rel);
            threshold_trace(t:end) = threshold_trace(t:end) + kernel;

            next_allowed_spike_time = t + tau_ref_pts;
            t = t + 1;
        else
            t = t + 1;
        end
    end

    predicted_spike_times_sec = predicted_spike_times * dt;
    fprintf('✅ Predicted spike count: %d\n', length(predicted_spike_times));
end
