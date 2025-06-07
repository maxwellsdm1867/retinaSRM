% === Setup ===
N = length(Vm_cleaned);
t = (0:N-1) * dt;
elbow_times = elbow_indices * dt;

% Interpolate threshold from elbow points
threshold_interp = interp1(elbow_times, threshold_values, t, 'linear', 'extrap');

% === Initial guess and bounds
init = [-50, 5, 0.01];        % [theta0, A, tau]
lb = [-60, 0, 0.001];
ub = [-40, 20, 0.05];

% === Run optimization using a plain function handle that calls a standalone function
opts = optimoptions('fmincon', 'Display', 'iter', 'MaxFunctionEvaluations', 1e4);

% Store everything needed in a struct (so we don't rely on workspace scope)
data_struct.N = N;
data_struct.dt = dt;
data_struct.spike_inds = elbow_indices;
data_struct.target = threshold_interp;

% Use anonymous handle that passes struct
objective = @(params) compute_loss_pure(params, data_struct);

[opt_params, loss_val] = fmincon(objective, init, [], [], [], [], lb, ub, [], opts);

% === Extract fitted parameters
theta0_fit = opt_params(1);
A_fit = opt_params(2);
tau_fit = opt_params(3);

% === Reconstruct model threshold trace using fitted parameters
theta_trace = theta0_fit * ones(N,1);
for i = 1:length(elbow_indices)
    idx = elbow_indices(i);
    if idx < N
        decay_len = N - idx + 1;
        t_rel = (0:decay_len-1)' * dt;
        theta_trace(idx:end) = theta_trace(idx:end) + A_fit * exp(-t_rel / tau_fit);
    end
end

% === Plot
figure;
plot(t, threshold_interp, 'b', 'LineWidth', 1.2); hold on;
plot(t, theta_trace, 'r--', 'LineWidth', 1.2);
legend('Interpolated Threshold', 'Fitted Model');
xlabel('Time (s)');
ylabel('Threshold (mV)');
title(sprintf('Fitted: \\theta_0 = %.2f, A = %.2f, \\tau = %.3f s', ...
    theta0_fit, A_fit, tau_fit));
grid on;

% === Print fitted parameters
fprintf('\nFitted Parameters:\n');
fprintf('Theta0 = %.3f mV\n', theta0_fit);
fprintf('A      = %.3f mV\n', A_fit);
fprintf('Tau    = %.4f s\n', tau_fit);


% === Use fitted parameters
theta0 = theta0_fit;
A = A_fit;
tau = tau_fit;


% === Compute Victor–Purpura distance
vp_q = 4;
vp_dist = spkd_c(pred_spike_times_sec, true_spike_times_sec, ...
                 length(pred_spike_times_sec), length(true_spike_times_sec), vp_q);

[spikes, threshold_trace, V_pred_full, predicted_spike_times_sec] = ...
    simulate_dynamic_threshold_spikes(Vm, theta0, A, tau, dt, avg_spike, tau_ref_ms);

% === Time vector
t = (0:length(Vm_all)-1) * dt;

% === Compute ISI (in ms)
isi = diff(pred_spike_times_sec) * 1000;

% === Zoom window (adjust as needed)
zoom_start = 1;         % in seconds
zoom_end   = 2;         % in seconds
zoom_mask = (t >= zoom_start) & (t <= zoom_end);

% === Plot
figure;

% --- Subplot 1: Zoomed Vm_all, V_pred_full, theta_trace
subplot(3,1,1);
plot(t(zoom_mask), Vm_all(zoom_mask), 'b', 'DisplayName', 'Vm\_all'); hold on;
plot(t(zoom_mask), V_pred_full(zoom_mask), 'r', 'DisplayName', 'V\_pred (with spikes)');
plot(t(zoom_mask), theta_trace(zoom_mask), 'k--', 'LineWidth', 1.2, 'DisplayName', '\theta (fitted)');
ylabel('Voltage (mV)');
legend;
title(sprintf('Zoomed Voltage Trace (%.1f–%.1f s)', zoom_start, zoom_end));
grid on;
ylim([-80 10])

% --- Subplot 2: Zoomed threshold comparison
subplot(3,1,2);
plot(t(zoom_mask), threshold_interp(zoom_mask), 'b', 'DisplayName', '\theta (data interp)'); hold on;
plot(t(zoom_mask), theta_trace(zoom_mask), 'r--', 'LineWidth', 1.2, 'DisplayName', '\theta (fitted)');
ylabel('Threshold (mV)');
legend;
title('Zoomed Threshold Comparison');
grid on;

% --- Subplot 3: Histogram of ISI
subplot(3,1,3);
histogram(isi, 30, 'FaceColor', [0.2 0.2 0.2]);
xlabel('ISI (ms)');
ylabel('Count');
title('ISI Distribution (Predicted Spikes)');
grid on;

% --- Global title
sgtitle(sprintf('Dynamic Threshold Model — VP Distance (q = %.1f): %.3f', vp_q, vp_dist));

%%

% === Initial guess from threshold fit
init = [-50, 1, 1];  % [theta0, A, tau]
lb = [-60, 0, 0.001];
ub = [-40, 20, 0.1];

% === Fixed parameters
vp_q = 4;
tau_ref_ms = 2;  % example refractory window

% === Loss function: VP distance from predicted spikes
loss_fn = @(params) compute_vp_loss(params, Vm_cleaned, dt, avg_spike, tau_ref_ms, elbow_indices, vp_q);

% === Optimization
opts = optimoptions('fmincon', 'Display', 'iter', 'MaxFunctionEvaluations', 1e4);
[opt_vp_params, vp_loss_val] = fminsearch(loss_fn, init, [], [], [], [], lb, ub, [], opts);

% === Extract new best-fit parameters
theta0_vp = opt_vp_params(1);
A_vp = opt_vp_params(2);
tau_vp = opt_vp_params(3);

fprintf('\n🧠 Optimized for Spike Timing (VP):\n');
fprintf('Theta0 = %.3f mV\n', theta0_vp);
fprintf('A      = %.3f mV\n', A_vp);
fprintf('Tau    = %.4f s\n', tau_vp);

% === Simulate using VP-optimized parameters
[spikes_vp, theta_trace_vp, V_pred_vp, predicted_spike_times_vp] = ...
    simulate_dynamic_threshold_spikes(Vm_cleaned, theta0_vp, A_vp, tau_vp, dt, avg_spike, tau_ref_ms);

% === Compute ISI
isi_vp = diff(predicted_spike_times_vp) * 1000;  % in ms
t = (0:length(Vm_all)-1) * dt;

% === Zoom window
zoom_start = 1;
zoom_end = 3;
zoom_mask = (t >= zoom_start) & (t <= zoom_end);

% === Compute final VP distance
true_spike_times_sec = elbow_indices * dt;
vp_dist_final = spkd_c(predicted_spike_times_vp, true_spike_times_sec, ...
                       length(predicted_spike_times_vp), length(true_spike_times_sec), vp_q);

% === Plot
figure;

% --- Subplot 1: Zoomed Vm_all, V_pred_vp, theta_trace_vp
subplot(3,1,1);
plot(t(zoom_mask), Vm_all(zoom_mask), 'b', 'DisplayName', 'Vm\_all'); hold on;
plot(t(zoom_mask), V_pred_vp(zoom_mask), 'r', 'DisplayName', 'V\_pred (with spikes)');
plot(t(zoom_mask), theta_trace_vp(zoom_mask), 'k--', 'LineWidth', 1.2, 'DisplayName', '\theta (VP-fit)');
ylabel('Voltage (mV)');
legend;
title(sprintf('Zoomed Voltage Trace (%.1f–%.1f s)', zoom_start, zoom_end));
grid on;

% --- Subplot 2: Threshold comparison (data vs. VP-fit)
subplot(3,1,2);
plot(t(zoom_mask), threshold_interp(zoom_mask), 'b', 'DisplayName', '\theta (data interp)'); hold on;
plot(t(zoom_mask), theta_trace_vp(zoom_mask), 'r--', 'LineWidth', 1.2, 'DisplayName', '\theta (VP-fit)');
ylabel('Threshold (mV)');
legend;
title('Threshold Comparison');
grid on;

% --- Subplot 3: ISI histogram
subplot(3,1,3);
histogram(isi_vp, 30, 'FaceColor', [0.2 0.2 0.2]);
xlabel('ISI (ms)');
ylabel('Count');
title('ISI Distribution (Predicted Spikes)');
grid on;

% --- Supertitle
sgtitle(sprintf('VP-Optimized Dynamic Threshold — VP Distance (q = %.1f): %.3f', vp_q, vp_dist_final));



%%
% === Initial guess
init = [theta0_fit, A_fit, tau_fit];

% === Loss function (calls your simulation function)
loss_fn = @(params) compute_vp_loss_fminsearch(params, Vm_cleaned, dt, avg_spike_short, tau_ref_ms, elbow_indices, vp_q);

% === Optimization
options = optimset('Display', 'iter', 'MaxFunEvals', 3000, 'MaxIter', 1000);
[opt_vp_params, vp_loss_val] = fminsearch(loss_fn, init, options);

% === Extract fitted params
theta0_vp = opt_vp_params(1);
A_vp = opt_vp_params(2);
tau_vp = opt_vp_params(3);

fprintf('\n🎯 Final fminsearch result:\n');
fprintf('Theta0 = %.3f mV\n', theta0_vp);
fprintf('A      = %.3f mV\n', A_vp);
fprintf('Tau    = %.4f s\n', tau_vp);


% === Simulate using VP-optimized parameters
[spikes_vp, theta_trace_vp, V_pred_vp, predicted_spike_times_vp] = ...
    simulate_dynamic_threshold_spikes(Vm_cleaned, theta0_vp, A_vp, tau_vp, dt, avg_spike, tau_ref_ms);

% === Compute ISI
isi_vp = diff(predicted_spike_times_vp) * 1000;  % in ms
t = (0:length(Vm_all)-1) * dt;

% === Zoom window
zoom_start = 1;
zoom_end = 3;
zoom_mask = (t >= zoom_start) & (t <= zoom_end);

% === Compute final VP distance
true_spike_times_sec = elbow_indices * dt;
vp_dist_final = spkd_c(predicted_spike_times_vp, true_spike_times_sec, ...
                       length(predicted_spike_times_vp), length(true_spike_times_sec), vp_q);

% === Plot
figure;

% --- Subplot 1: Zoomed Vm_all, V_pred_vp, theta_trace_vp
subplot(3,1,1);
plot(t(zoom_mask), Vm_all(zoom_mask), 'b', 'DisplayName', 'Vm\_all'); hold on;
plot(t(zoom_mask), V_pred_vp(zoom_mask), 'r', 'DisplayName', 'V\_pred (with spikes)');
plot(t(zoom_mask), theta_trace_vp(zoom_mask), 'k--', 'LineWidth', 1.2, 'DisplayName', '\theta (VP-fit)');
ylabel('Voltage (mV)');
legend;
title(sprintf('Zoomed Voltage Trace (%.1f–%.1f s)', zoom_start, zoom_end));
grid on;

% --- Subplot 2: Threshold comparison (data vs. VP-fit)
subplot(3,1,2);
plot(t(zoom_mask), threshold_interp(zoom_mask), 'b', 'DisplayName', '\theta (data interp)'); hold on;
plot(t(zoom_mask), theta_trace_vp(zoom_mask), 'r--', 'LineWidth', 1.2, 'DisplayName', '\theta (VP-fit)');

% Add green markers at elbow points
elbow_times = elbow_indices * dt;
elbow_mask = elbow_times >= zoom_start & elbow_times <= zoom_end;
plot(elbow_times(elbow_mask), threshold_values(elbow_mask), 'go', 'MarkerSize', 6, 'MarkerFaceColor', 'g', 'DisplayName', 'Spike Initiations');


ylabel('Threshold (mV)');
legend;
title('Threshold Comparison with Real Spike Initiation');
grid on;
% === Compute ISIs
isi_true = diff(elbow_indices) * dt * 1000;         % true ISIs (ms)
isi_pred = diff(find(spikes_vp)) * dt * 1000;       % predicted ISIs (ms)

% === Histogram bins
edges = 0:2:100;  % 2 ms bins up to 100 ms

% --- Subplot 3: ISI histograms
subplot(3,1,3);
histogram(isi_true, edges, 'FaceColor', 'b', 'FaceAlpha', 0.5, 'DisplayName', 'True ISI'); hold on;
histogram(isi_pred, edges, 'FaceColor', 'r', 'FaceAlpha', 0.5, 'DisplayName', 'Predicted ISI');
xlabel('ISI (ms)');
ylabel('Count');
title('ISI Distribution: True vs. Predicted');
legend;
grid on;

% --- Supertitle
sgtitle(sprintf('VP-Optimized Dynamic Threshold — VP Distance (q = %.1f): %.3f', vp_q, vp_dist_final));












% === Helper Function (separate definition at bottom of script)
function loss = compute_loss_pure(params, data)
    theta0 = params(1);
    A = params(2);
    tau = params(3);
    N = data.N;
    dt = data.dt;
    spike_inds = data.spike_inds;
    target = data.target;

    theta = theta0 * ones(N,1);
    for i = 1:length(spike_inds)
        idx = spike_inds(i);
        if idx < N
            len = N - idx + 1;
            t_rel = (0:len-1)' * dt;
            theta(idx:end) = theta(idx:end) + A * exp(-t_rel / tau);
        end
    end

    % <<< SAFEST WAY TO COMPUTE LOSS >>>
    loss = 0;
    for k = 1:N
        d = theta(k) - target(k);
        if ~isnan(d)
            loss = loss + d^2;
        end
    end
end


