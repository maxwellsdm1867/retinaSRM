%% === Dynamic Threshold VP Optimization with Piecewise Kernel ===

% --- Initial guess [theta0, A, T_rise (s), tau_decay (s)]
init = [-50, 5, 0.002, 0.03];

% --- VP loss wrapper
loss_fn = @(params) compute_vp_loss_piecewise(params, Vm_cleaned, dt, avg_spike_short, tau_ref_ms, elbow_indices, vp_q);

% --- fminsearch options
options = optimset('Display', 'iter', 'MaxFunEvals', 3000, 'MaxIter', 1000);
[opt_params, vp_loss_val] = fminsearch(loss_fn, init, options);

% --- Extract optimized parameters
theta0_vp = opt_params(1);
A_vp = opt_params(2);
T_rise_vp = opt_params(3);
tau_decay_vp = opt_params(4);

fprintf('\n🎯 Final VP-fit (piecewise model):\n');
fprintf('Theta0 = %.3f mV\n', theta0_vp);
fprintf('A      = %.3f mV\n', A_vp);
fprintf('T_rise = %.4f s\n', T_rise_vp);
fprintf('Tau    = %.4f s\n', tau_decay_vp);

% --- Simulate with optimized params
[spikes_vp, theta_trace_vp, V_pred_vp, predicted_spike_times_vp] = ...
    simulate_dynamic_threshold_piecewise(Vm_cleaned, theta0_vp, A_vp, T_rise_vp, tau_decay_vp, dt, avg_spike_short, tau_ref_ms);

% === Compute ISI and plot
isi_true = diff(elbow_indices) * dt * 1000;
isi_pred = diff(find(spikes_vp)) * dt * 1000;
t = (0:length(Vm_all)-1) * dt;

zoom_start = 1;
zoom_end = 3;
zoom_mask = (t >= zoom_start) & (t <= zoom_end);
elbow_times = elbow_indices * dt;

vp_dist_final = spkd_c(predicted_spike_times_vp, elbow_times, ...
                       length(predicted_spike_times_vp), length(elbow_times), vp_q);

figure;

% --- Subplot 1: Vm + V_pred + threshold
subplot(3,1,1);
plot(t(zoom_mask), Vm_all(zoom_mask), 'b'); hold on;
plot(t(zoom_mask), V_pred_vp(zoom_mask), 'r');
plot(t(zoom_mask), theta_trace_vp(zoom_mask), 'k--', 'LineWidth', 1.2);
ylabel('Voltage (mV)');
legend('Vm\_all', 'V\_pred', '\theta (VP-fit)');
title(sprintf('Zoomed Voltage Trace (%.1f–%.1f s)', zoom_start, zoom_end));
grid on;

% --- Subplot 2: Thresholds + elbow points
subplot(3,1,2);
plot(t(zoom_mask), threshold_interp(zoom_mask), 'b', 'DisplayName', '\theta (data interp)'); hold on;
plot(t(zoom_mask), theta_trace_vp(zoom_mask), 'r--', 'DisplayName', '\theta (VP-fit)');
elbow_mask = elbow_times >= zoom_start & elbow_times <= zoom_end;
plot(elbow_times(elbow_mask), threshold_values(elbow_mask), ...
     'go', 'MarkerSize', 6, 'MarkerFaceColor', 'g', 'DisplayName', 'Spike Initiations');
ylabel('Threshold (mV)');
legend;
title('Threshold Comparison with Real Spike Initiation');
grid on;

% --- Subplot 3: ISI histograms
edges = 0:2:100;
subplot(3,1,3);
histogram(isi_true, edges, 'FaceColor', 'b', 'FaceAlpha', 0.5, 'DisplayName', 'True ISI'); hold on;
histogram(isi_pred, edges, 'FaceColor', 'r', 'FaceAlpha', 0.5, 'DisplayName', 'Predicted ISI');
xlabel('ISI (ms)');
ylabel('Count');
legend;
title('ISI Distribution: True vs. Predicted');
grid on;

sgtitle(sprintf('VP-Optimized Dynamic Threshold — VP Distance (q = %.1f): %.3f', vp_q, vp_dist_final));


%%
function vp_dist = compute_vp_loss_piecewise(params, Vm, dt, avg_spike, tau_ref_ms, elbow_indices, vp_q)
    theta0 = params(1);
    A = params(2);
    T_rise = params(3);
    tau = params(4);

    if tau <= 0 || A < 0 || T_rise <= 0
        vp_dist = Inf;
        return;
    end

    [~, ~, ~, predicted_spike_times] = simulate_dynamic_threshold_piecewise( ...
        Vm, theta0, A, T_rise, tau, dt, avg_spike, tau_ref_ms);

    true_spike_times = elbow_indices * dt;

    vp_dist = spkd_c(predicted_spike_times, true_spike_times, ...
                     length(predicted_spike_times), length(true_spike_times), vp_q);

    fprintf('θ0 = %.2f | A = %.2f | T_rise = %.3f | τ = %.3f → VP = %.2f | Spikes: %d vs. %d\n', ...
        theta0, A, T_rise, tau, vp_dist, length(predicted_spike_times), length(true_spike_times));
end
function [spikes, threshold_trace, V_pred_full, predicted_spike_times_sec] = ...
    simulate_dynamic_threshold_piecewise(Vm, theta0, A, T_rise, tau_decay, dt, avg_spike, tau_ref_ms)

    % === Parameters ===
    N = length(Vm);
    tau_ref_pts = round(tau_ref_ms / 1000 / dt);   % refractory period in samples
    spike_len = length(avg_spike);
    avg_spike_corrected = avg_spike - avg_spike(1);  % baseline at 0

    % === Outputs ===
    spikes = zeros(N,1);
    threshold_trace = theta0 * ones(N,1);
    V_pred_full = Vm;
    predicted_spike_times = [];

    % === Initialize
    t = 2;
    next_allowed_spike_time = 1;

    while t <= N
        % --- Enforce absolute refractory ---
        if t < next_allowed_spike_time
            t = t + 1;
            continue;
        end

        % --- Dynamically evaluate threshold using piecewise shape ---
        threshold = theta0;
        for i = 1:length(predicted_spike_times)
            delta_t = (t - predicted_spike_times(i)) * dt;
            if delta_t < 0
                continue;
            elseif delta_t < T_rise
                threshold = threshold + (A / T_rise) * delta_t;  % linear rise
            else
                threshold = threshold + A * exp(-(delta_t - T_rise) / tau_decay);  % exponential decay
            end
        end

        threshold_trace(t) = threshold;

        % --- Spike condition ---
        if Vm(t) >= threshold && Vm(t-1) < threshold
            % Register spike
            spikes(t) = 1;
            predicted_spike_times(end+1) = t;

            % Add spike waveform
            eta_start = t;
            eta_end = min(t + spike_len - 1, N);
            eta_len = eta_end - eta_start + 1;
            V_pred_full(eta_start:eta_end) = ...
                V_pred_full(eta_start:eta_end) + avg_spike_corrected(1:eta_len)';

            % Update refractory window
            next_allowed_spike_time = t + tau_ref_pts;
            t = t + 1;
        else
            t = t + 1;
        end
    end

    % === Final output ===
    predicted_spike_times_sec = predicted_spike_times * dt;
    fprintf('✅ Predicted spike count: %d\n', length(predicted_spike_times));
end
