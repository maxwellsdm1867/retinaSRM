%% THRESHOLD RECOVERY ANALYSIS  
spike_idx = round(spike_times(1) / dt);
recovery_indices = spike_idx + round([0, 5, 10, 20, 50] / 1000 / dt);
recovery_indices = recovery_indices(recovery_indices <= length(threshold_trace));

fprintf('Threshold recovery after first spike:\n');
for i = 1:length(recovery_indices)
    time_offset = (recovery_indices(i) - spike_idx) * dt * 1000;
    thresh_val = threshold_trace(recovery_indices(i));
    thresh_increase = thresh_val - theta0;
    fprintf('  +%2.0fms: θ=%.1fmV (+%.1fmV)\n', time_offset, thresh_val, thresh_increase);
end%% Advanced Simulation Analysis and Parameter Sensitivity
% This script analyzes the simulation results and tests parameter sensitivity

clear; close all; clc;

%% REPRODUCE BASIC SETUP (from previous test)
dt = 0.0001; duration = 0.5; t = 0:dt:duration-dt; N = length(t);
Vm_base = -65;
Vm = Vm_base + 5*sin(2*pi*10*t) + 2*randn(size(t));

step_times = [0.1, 0.2, 0.35];
for i = 1:length(step_times)
    step_idx = round(step_times(i)/dt);
    if step_idx <= N
        Vm(step_idx:min(step_idx+500, N)) = Vm(step_idx:min(step_idx+500, N)) + 8;
    end
end

Vm_recorded = Vm; % Simplified for analysis
spike_len = round(0.003/dt);
t_spike = (0:spike_len-1)*dt*1000;
avg_spike = 30 * exp(-t_spike/0.5) .* sin(2*pi*t_spike/0.8);
avg_spike = avg_spike(:);

tau_ref_ms = 2;
elbow_indices = round(step_times/dt);
threshold_values = -55 * ones(size(elbow_indices));

model = SpikeResponseModel(Vm, Vm_recorded, dt, avg_spike, tau_ref_ms, ...
                          elbow_indices, threshold_values, 'TestCell', 'Analysis');

%% PARAMETER SENSITIVITY TEST
fprintf('=== PARAMETER SENSITIVITY ===\n');

% Test different kernel parameters
theta0 = -55;
kernel_params = [
    10, 0.020;     % Original: strong, fast decay
    10, 0.050;     % Slower decay
    15, 0.020;     % Stronger amplitude
    5,  0.020;     % Weaker amplitude
    10, 0.010;     % Faster decay
];

figure('Position', [100, 100, 1500, 1000]);
colors = {'r', 'g', 'b', 'm', 'c'};

for i = 1:size(kernel_params, 1)
    A = kernel_params(i, 1);
    tau = kernel_params(i, 2);
    kernel_fn = @(t) A * exp(-t / tau);
    
    [spikes, V_pred, threshold_trace, spike_times] = model.simulate(theta0, kernel_fn);
    
    fprintf('A=%02.0f τ=%03.0fms: %2d spikes | ISI=[%s] | Times=[%s]\n', ...
            A, tau*1000, length(spike_times), ...
            sprintf('%2.0f ', diff(spike_times)*1000), ...
            sprintf('%3.0f ', spike_times*1000));
    
    % Plot threshold evolution
    subplot(2, 3, i);
    plot(t*1000, Vm, 'k:', 'LineWidth', 0.5); hold on;
    plot(t*1000, V_pred, colors{i}, 'LineWidth', 1);
    plot(t*1000, threshold_trace, [colors{i} '--'], 'LineWidth', 2);
    
    spike_idx = find(spikes);
    plot(t(spike_idx)*1000, V_pred(spike_idx), [colors{i} 'o'], 'MarkerSize', 6, 'MarkerFaceColor', colors{i});
    
    xlim([90, 140]); ylim([-70, -40]);
    title(sprintf('A=%.1f, \\tau=%.3fs (%d spikes)', A, tau, length(spike_times)));
    xlabel('Time (ms)'); ylabel('Voltage (mV)');
    legend('Vm', 'V_{pred}', 'Threshold', 'Spikes', 'Location', 'best');
    grid on;
end

% Summary plot: kernel comparison
subplot(2, 3, 6);
t_kernel = 0:0.001:0.1; % 0 to 100 ms
for i = 1:size(kernel_params, 1)
    A = kernel_params(i, 1);
    tau = kernel_params(i, 2);
    kernel_vals = A * exp(-t_kernel / tau);
    plot(t_kernel*1000, kernel_vals, colors{i}, 'LineWidth', 2); hold on;
end
xlabel('Time (ms)'); ylabel('Threshold Increase (mV)');
title('Kernel Comparison');
legend('Orig', 'Slow', 'Strong', 'Weak', 'Fast', 'Location', 'northeast');
grid on;

sgtitle('Parameter Sensitivity Analysis');

%% TIMING VALIDATION
fprintf('\n=== TIMING VALIDATION ===\n');

% Use original parameters
A_orig = 10; tau_orig = 0.020;
kernel_fn = @(t) A_orig * exp(-t / tau_orig);
[spikes, V_pred, threshold_trace, spike_times] = model.simulate(theta0, kernel_fn);

% Analyze inter-spike intervals
if length(spike_times) > 1
    ISIs = diff(spike_times) * 1000;
    fprintf('ISI stats: mean=%.1fms, min=%.1fms, max=%.1fms\n', mean(ISIs), min(ISIs), max(ISIs));
    
    violations = sum(ISIs < tau_ref_ms);
    if violations > 0
        fprintf('⚠ WARNING: %d refractory violations detected!\n', violations);
    else
        fprintf('✓ All ISIs respect %.0fms refractory period\n', tau_ref_ms);
    end
end

% Check spike timing accuracy vs expected
expected_times = step_times * 1000; % Convert to ms
fprintf('Expected stimulus times: [%s]ms\n', sprintf('%3.0f ', expected_times));
spike_times_ms = spike_times * 1000;

for i = 1:length(expected_times)
    nearby_spikes = spike_times_ms(abs(spike_times_ms - expected_times(i)) < 30);
    fprintf('Stimulus %d (%.0fms): %d spikes within 30ms [%s]\n', ...
            i, expected_times(i), length(nearby_spikes), sprintf('%.0f ', nearby_spikes));
end

%% CRITICAL METRICS FOR VALIDATION
fprintf('\n=== CRITICAL VALIDATION METRICS ===\n');

% Metric 1: Spike count per stimulus
stim_windows = [0.09 0.14; 0.19 0.24; 0.34 0.39];
for i = 1:size(stim_windows, 1)
    spikes_in_window = sum(spike_times >= stim_windows(i,1) & spike_times <= stim_windows(i,2));
    fprintf('Stimulus %d: %d spikes in [%.0f-%.0f]ms window\n', ...
            i, spikes_in_window, stim_windows(i,:)*1000);
end

% Metric 2: Threshold dynamics validation
max_threshold = max(threshold_trace);
min_threshold = min(threshold_trace);
fprintf('Threshold range: [%.1f, %.1f]mV (span=%.1fmV)\n', ...
        min_threshold, max_threshold, max_threshold - min_threshold);

% Metric 3: Voltage prediction accuracy
correlation = corr(Vm(:), V_pred(:));
fprintf('Vm vs V_pred correlation: r=%.3f\n', correlation);

% Metric 4: Spike injection validation
spike_indices = find(spikes);
if ~isempty(spike_indices)
    pre_spike_V = V_pred(spike_indices);
    post_spike_V = V_pred(min(spike_indices + 1, length(V_pred)));
    avg_spike_jump = mean(post_spike_V - pre_spike_V);
    fprintf('Average spike injection amplitude: %.1fmV\n', avg_spike_jump);
end

% Focus on first spike sequence (around t=0.1s)
focus_start = 0.09; focus_end = 0.14;
focus_idx = (t >= focus_start) & (t <= focus_end);
t_focus = t(focus_idx);
Vm_focus = Vm(focus_idx);
V_pred_focus = V_pred(focus_idx);
thresh_focus = threshold_trace(focus_idx);

% Find spikes in this window
spike_mask = (spike_times >= focus_start) & (spike_times <= focus_end);
spikes_in_window = spike_times(spike_mask);

figure('Position', [100, 100, 1200, 800]);

% Subplot 1: Voltage traces
subplot(3,1,1);
plot(t_focus*1000, Vm_focus, 'b-', 'LineWidth', 1.5); hold on;
plot(t_focus*1000, V_pred_focus, 'r-', 'LineWidth', 1.5);
plot(t_focus*1000, thresh_focus, 'k--', 'LineWidth', 2);

% Mark spike times
for i = 1:length(spikes_in_window)
    spike_t = spikes_in_window(i) * 1000;
    spike_idx_local = find(t_focus <= spikes_in_window(i), 1, 'last');
    if ~isempty(spike_idx_local)
        plot(spike_t, V_pred_focus(spike_idx_local), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
        % Add vertical line
        ylims = ylim;
        plot([spike_t, spike_t], ylims, 'r:', 'LineWidth', 1);
    end
end

xlabel('Time (ms)'); ylabel('Voltage (mV)');
title('Voltage Dynamics During Spike Sequence');
legend('Vm (input)', 'V_{pred}', 'Threshold', 'Spikes', 'Location', 'best');
grid on;

% Subplot 2: Threshold evolution
subplot(3,1,2);
plot(t_focus*1000, thresh_focus, 'k-', 'LineWidth', 2); hold on;
plot(t_focus*1000, theta0*ones(size(t_focus)), 'k:', 'LineWidth', 1);

% Show kernel contributions
baseline_thresh = theta0 * ones(size(t_focus));
kernel_contribution = thresh_focus - baseline_thresh;
plot(t_focus*1000, kernel_contribution, 'g-', 'LineWidth', 1.5);

for i = 1:length(spikes_in_window)
    spike_t = spikes_in_window(i) * 1000;
    plot(spike_t, theta0, 'ro', 'MarkerSize', 6, 'MarkerFaceColor', 'r');
end

xlabel('Time (ms)'); ylabel('Threshold (mV)');
title('Threshold Evolution');
legend('Dynamic Threshold', 'Baseline', 'Kernel Contribution', 'Spike Times', 'Location', 'best');
grid on;

% Subplot 3: Margin analysis
subplot(3,1,3);
margin = V_pred_focus - thresh_focus;
plot(t_focus*1000, margin, 'Color', [0.5 0 0.5], 'LineWidth', 2); hold on;
plot(t_focus*1000, zeros(size(t_focus)), 'k--', 'LineWidth', 1);

% Shade positive regions (spiking regions)
positive_idx = margin > 0;
if any(positive_idx)
    area(t_focus(positive_idx)*1000, margin(positive_idx), 'FaceColor', 'red', 'FaceAlpha', 0.3);
end

xlabel('Time (ms)'); ylabel('V_{pred} - Threshold (mV)');
title('Spiking Margin (Positive = Spike Condition Met)');
grid on;

sgtitle('Detailed Spike Train Analysis');

%% OPTIMIZED PARAMETER TEST
fprintf('\n=== OPTIMIZED PARAMETERS TEST ===\n');

A_opt = 15; tau_opt = 0.050;
kernel_opt = @(t) A_opt * exp(-t / tau_opt);
[spikes_opt, ~, ~, spike_times_opt] = model.simulate(theta0, kernel_opt);

fprintf('Optimized (A=%d, τ=%dms): %d spikes | Times=[%s]ms\n', ...
        A_opt, tau_opt*1000, length(spike_times_opt), sprintf('%3.0f ', spike_times_opt*1000));

% Calculate success metric
success_count = 0;
for i = 1:length(step_times)
    nearby_spikes = sum(abs(spike_times_opt - step_times(i)) < 0.02);
    success_count = success_count + min(nearby_spikes, 1);
end
fprintf('Success rate: %d/%d stimuli triggered spikes (%.0f%%)\n', ...
        success_count, length(step_times), 100*success_count/length(step_times));