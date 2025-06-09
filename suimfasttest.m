%% Fast Spike Response Model Testing using SpikeResponseModel.simulateFast()
% Simple demonstration using existing simulate method with known good parameters

clear; clc; close all;

%% === SIMULATION PARAMETERS ===
% Time parameters
dt = 0.0001;                % Time step (100 μs)
t_total = 5;                % Total simulation time (s)
t_vec = 0:dt:t_total-dt;    % Time vector
n_samples = length(t_vec);

% Victor-Purpura cost parameter
vp_q = 4;

% Refractory period
tau_ref_ms = 2;             % Refractory period (ms)

%% === GENERATE REALISTIC DATA ===
fprintf('Generating realistic test data...\n');

% Create realistic voltage trace
Vm_base = -65;              % Baseline voltage (mV)
noise_std = 3;              % Noise standard deviation (mV)

% Generate voltage with realistic fluctuations
Vm_cleaned = Vm_base + noise_std * randn(1, n_samples);

% Add some slow trends and larger excursions
for i = 1:20
    event_time = rand() * t_total;
    event_idx = round(event_time / dt);
    if event_idx > 100 && event_idx < n_samples - 200
        % Add depolarizing event
        event_amplitude = 5 + 10*rand();  % 5-15 mV depolarization
        event_duration = 50 + 100*rand(); % 50-150 samples
        event_profile = event_amplitude * exp(-(0:round(event_duration)) / (event_duration/3));
        end_idx = min(event_idx + length(event_profile) - 1, n_samples);
        actual_profile = event_profile(1:(end_idx - event_idx + 1));
        Vm_cleaned(event_idx:end_idx) = Vm_cleaned(event_idx:end_idx) + actual_profile;
    end
end

% Create Vm_recorded (will be modified by simulate to include spikes)
Vm_all = Vm_cleaned;

% Create spike template
avg_spike_short = 20 * exp(-(0:50) / 10);  % 20 mV spike

% Generate some fake elbow indices and thresholds for the model
% (These won't affect the actual simulation, just needed for model construction)
fake_spike_times = [1.0, 2.0, 3.0, 4.0];  % Some fake spike times
fake_elbow_indices = round(fake_spike_times / dt);
fake_threshold_values = -50 + 2*randn(1, length(fake_elbow_indices));

fprintf('Vm range: %.1f to %.1f mV\n', min(Vm_cleaned), max(Vm_cleaned));

%% === CREATE SPIKERESPONSEMODEL OBJECT ===
fprintf('Creating SpikeResponseModel object...\n');

model = SpikeResponseModel( ...
    Vm_cleaned, ...              % Subthreshold voltage trace
    Vm_all, ...                  % Recorded full Vm
    dt, ...                      % Time step
    avg_spike_short, ...         % Spike waveform to inject
    tau_ref_ms, ...              % Refractory period in ms
    fake_elbow_indices, ...      % Fake elbow indices
    fake_threshold_values, ...   % Fake threshold values
    "Cell-Test", ...             % Cell ID
    "Fast-Test" ...              % Cell type
);

%% === TEST DIFFERENT PARAMETER SETS ===
fprintf('\n=== TESTING DIFFERENT PARAMETER COMBINATIONS ===\n');

% Define parameter sets to test (theta0, A, tau)
param_sets = [
    -52, 3.0, 0.01;    % Set 1: Moderate threshold, moderate adaptation
    -48, 5.0, 0.02;    % Set 2: Higher threshold, stronger adaptation
    -50, 2.0, 0.015;   % Set 3: Medium threshold, weak adaptation
    -46, 4.0, 0.025;   % Set 4: Low threshold, strong long adaptation
    -54, 1.5, 0.005;   % Set 5: High threshold, weak fast adaptation
];

param_names = {'Moderate', 'Strong Adapt', 'Weak Adapt', 'Long Adapt', 'Fast Adapt'};

spike_results = cell(size(param_sets, 1), 1);

for i = 1:size(param_sets, 1)
    theta0 = param_sets(i, 1);
    A = param_sets(i, 2);
    tau = param_sets(i, 3);
    
    fprintf('\nTesting %s: θ₀=%.1f, A=%.1f, τ=%.3f\n', param_names{i}, theta0, A, tau);
    
    % Define kernel
    kernel_fn = @(t) A * exp(-t / tau);
    
    % Run simulation
    [spikes, V_pred, threshold_trace, spike_times_sec, spike_V_values] = ...
        model.simulate(theta0, kernel_fn);
    
    fprintf('  Generated %d spikes (%.1f Hz)\n', length(spike_times_sec), length(spike_times_sec)/t_total);
    
    % Store results
    spike_results{i} = struct(...
        'params', param_sets(i, :), ...
        'param_name', param_names{i}, ...
        'spikes', spikes, ...
        'V_pred', V_pred, ...
        'threshold_trace', threshold_trace, ...
        'spike_times_sec', spike_times_sec, ...
        'spike_V_values', spike_V_values);
end

%% === SELECT BEST PARAMETER SET FOR DETAILED ANALYSIS ===
% Choose the set that generates a reasonable number of spikes (5-15 spikes)
spike_counts = cellfun(@(x) length(x.spike_times_sec), spike_results);
target_count_range = [5, 15];
good_sets = find(spike_counts >= target_count_range(1) & spike_counts <= target_count_range(2));

if ~isempty(good_sets)
    best_idx = good_sets(1);  % Use first good set
else
    [~, best_idx] = min(abs(spike_counts - 10));  % Closest to 10 spikes
end

best_result = spike_results{best_idx};
fprintf('\n=== SELECTED PARAMETER SET: %s ===\n', best_result.param_name);
fprintf('Parameters: θ₀=%.1f, A=%.1f, τ=%.3f\n', best_result.params);
fprintf('Generated %d spikes\n', length(best_result.spike_times_sec));

%% === PERFORMANCE COMPARISON: simulate vs simulateFast ===
fprintf('\n=== PERFORMANCE TESTING ===\n');

% Use the best parameter set
theta0_best = best_result.params(1);
A_best = best_result.params(2);
tau_best = best_result.params(3);
kernel_best = @(t) A_best * exp(-t / tau_best);

% Test standard simulate method
n_tests_standard = 10;
fprintf('Testing standard simulate method with %d simulations...\n', n_tests_standard);
tic;
for i = 1:n_tests_standard
    [~, ~, ~, ~, ~] = model.simulate(theta0_best, kernel_best);
end
standard_sim_time = toc;

% Test simulateFast method  
n_tests_fast = 50;
fprintf('Testing simulateFast method with %d simulations...\n', n_tests_fast);
tic;
for i = 1:n_tests_fast
    [~, ~, ~, ~, ~] = model.simulateFast(theta0_best, kernel_best, 'profile', false);
end
fast_sim_time = toc;

% Calculate performance metrics
avg_standard_time = 1000 * standard_sim_time / n_tests_standard;
avg_fast_time = 1000 * fast_sim_time / n_tests_fast;
speedup_factor = avg_standard_time / avg_fast_time;

fprintf('Standard simulate: %.2f ms per simulation\n', avg_standard_time);
fprintf('simulateFast: %.2f ms per simulation\n', avg_fast_time);
fprintf('Speedup factor: %.1fx\n', speedup_factor);

%% === DETAILED SIMULATEFAST ANALYSIS ===
fprintf('\n=== DETAILED SIMULATEFAST ANALYSIS ===\n');

% Run simulateFast with profiling
[spikes_fast, V_pred_fast, threshold_fast, spike_times_fast, spike_V_fast] = ...
    model.simulateFast(theta0_best, kernel_best, 'method', 'vectorized', 'profile', true);

% Compare results between simulate and simulateFast
fprintf('\nComparison between simulate and simulateFast:\n');
fprintf('  Standard simulate: %d spikes\n', length(best_result.spike_times_sec));
fprintf('  simulateFast:      %d spikes\n', length(spike_times_fast));

% Check if results are similar
if length(spike_times_fast) > 0 && length(best_result.spike_times_sec) > 0
    time_diff = abs(length(spike_times_fast) - length(best_result.spike_times_sec));
    fprintf('  Spike count difference: %d\n', time_diff);
    if time_diff <= 2
        fprintf('  ✅ Results are consistent\n');
    else
        fprintf('  ⚠️  Significant difference in spike counts\n');
    end
else
    fprintf('  ⚠️  One method produced no spikes\n');
end

%% === VISUALIZATION ===
fprintf('\nCreating diagnostic plots...\n');

figure('Position', [50, 50, 1400, 900]);

% Plot all parameter set results
subplot(3,3,1);
bar(spike_counts);
set(gca, 'XTickLabel', param_names, 'XTickLabelRotation', 45);
ylabel('Spike Count');
title('Spike Counts for Different Parameters');
grid on;
hold on;
bar(best_idx, spike_counts(best_idx), 'r');  % Highlight selected set

% Plot voltage trace and threshold for best result
subplot(3,3,[2 3]);
t_plot = t_vec;
plot(t_plot, model.Vm, 'k-', 'LineWidth', 0.8); hold on;
plot(t_plot, best_result.V_pred, 'b-', 'LineWidth', 1.2);
plot(t_plot, best_result.threshold_trace, 'r--', 'LineWidth', 1.5);
if ~isempty(best_result.spike_times_sec)
    scatter(best_result.spike_times_sec, best_result.spike_V_values, 60, 'ro', 'filled');
end
xlabel('Time (s)');
ylabel('Voltage (mV)');
title(sprintf('Voltage Trace - %s Parameters', best_result.param_name));
legend('Vm cleaned', 'V pred', 'Threshold', 'Spikes', 'Location', 'best');
xlim([1, 3]);  % Zoom to 1-3 seconds
grid on;

% Plot zoomed view
subplot(3,3,[4 5]);
zoom_range = t_plot >= 1.5 & t_plot <= 2.5;
plot(t_plot(zoom_range), model.Vm(zoom_range), 'k-', 'LineWidth', 1); hold on;
plot(t_plot(zoom_range), best_result.V_pred(zoom_range), 'b-', 'LineWidth', 1.5);
plot(t_plot(zoom_range), best_result.threshold_trace(zoom_range), 'r--', 'LineWidth', 2);
zoom_spikes = best_result.spike_times_sec >= 1.5 & best_result.spike_times_sec <= 2.5;
if any(zoom_spikes)
    scatter(best_result.spike_times_sec(zoom_spikes), ...
        best_result.spike_V_values(zoom_spikes), 80, 'ro', 'filled');
end
xlabel('Time (s)');
ylabel('Voltage (mV)');
title('Zoomed View (1.5-2.5s)');
legend('Vm cleaned', 'V pred', 'Threshold', 'Spikes', 'Location', 'best');
grid on;

% Performance comparison
subplot(3,3,6);
perf_data = [avg_standard_time, avg_fast_time];
bar_labels = {'simulate', 'simulateFast'};
bar_colors = [0.7 0.7 0.7; 0.2 0.6 0.8];
b = bar(perf_data);
b.FaceColor = 'flat';
b.CData = bar_colors;
set(gca, 'XTickLabel', bar_labels);
ylabel('Time per simulation (ms)');
title('Performance Comparison');
text(1.5, max(perf_data) * 0.7, sprintf('%.1fx\nspeedup', speedup_factor), ...
    'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

% Adaptation kernel
subplot(3,3,7);
t_kernel = 0:dt:0.1;
kernel_vals = A_best * exp(-t_kernel / tau_best);
plot(t_kernel*1000, kernel_vals, 'b-', 'LineWidth', 2);
xlabel('Time after spike (ms)');
ylabel('Threshold increase (mV)');
title(sprintf('Adaptation Kernel\nA=%.1f, τ=%.1fms', A_best, tau_best*1000));
grid on;

% Spike time raster for all parameter sets
subplot(3,3,8);
for i = 1:length(spike_results)
    if ~isempty(spike_results{i}.spike_times_sec)
        y_pos = i * ones(size(spike_results{i}.spike_times_sec));
        plot(spike_results{i}.spike_times_sec, y_pos, 'ko', 'MarkerSize', 4); hold on;
    end
end
set(gca, 'YTick', 1:length(param_names), 'YTickLabel', param_names);
xlabel('Time (s)');
ylabel('Parameter Set');
title('Spike Rasters');
ylim([0.5, length(param_names) + 0.5]);
grid on;

% ISI distribution for best result
subplot(3,3,9);
if length(best_result.spike_times_sec) > 1
    isis = diff(best_result.spike_times_sec) * 1000;  % Convert to ms
    histogram(isis, 'BinWidth', 5, 'FaceColor', 'b', 'FaceAlpha', 0.7);
    xlabel('ISI (ms)');
    ylabel('Count');
    title('Inter-Spike Intervals');
    grid on;
else
    text(0.5, 0.5, 'Not enough spikes\nfor ISI analysis', ...
        'HorizontalAlignment', 'center', 'Units', 'normalized');
    title('Inter-Spike Intervals');
end

sgtitle('SpikeResponseModel simulateFast Analysis', 'FontSize', 16, 'FontWeight', 'bold');

%% === PARAMETER OPTIMIZATION DEMO ===
fprintf('\n=== PARAMETER OPTIMIZATION DEMO ===\n');

% Use the generated spikes as "ground truth" for optimization demo
true_spike_times = best_result.spike_times_sec;
fprintf('Using %d spikes as ground truth for optimization demo\n', length(true_spike_times));

% Update model with these spike times as elbow indices
true_elbow_indices = round(true_spike_times / dt);
true_threshold_values = best_result.threshold_trace(true_elbow_indices);

% Create new model object with true spike data
model_opt = SpikeResponseModel( ...
    Vm_cleaned, Vm_all, dt, avg_spike_short, tau_ref_ms, ...
    true_elbow_indices, true_threshold_values, "Cell-Opt", "Optimization-Test");

% Define optimization objective
loss_fn = @(params) model_opt.vp_loss_exponential(params, vp_q);

% Start with slightly perturbed parameters
init_params = best_result.params + [2, 0.5, 0.005];  % Add some error
fprintf('Initial guess: θ₀=%.1f, A=%.1f, τ=%.3f\n', init_params);

% Run quick optimization
options = optimset('Display', 'off', 'MaxFunEvals', 50, 'MaxIter', 25);
tic;
[opt_params, final_vp] = fminsearch(loss_fn, init_params, options);
opt_time = toc;

fprintf('Optimized:     θ₀=%.1f, A=%.1f, τ=%.3f\n', opt_params);
fprintf('True values:   θ₀=%.1f, A=%.1f, τ=%.3f\n', best_result.params);
fprintf('Optimization time: %.2f seconds\n', opt_time);
fprintf('Final VP distance: %.3f\n', final_vp);

%% === SUMMARY ===
fprintf('\n=== FINAL SUMMARY ===\n');
fprintf('✅ Successfully demonstrated simulateFast with SpikeResponseModel\n');
fprintf('📊 Best parameter set: %s\n', best_result.param_name);
fprintf('⚡ Performance: %.1fx speedup over standard simulate\n', speedup_factor);
fprintf('🎯 Generated %d spikes with realistic parameters\n', length(best_result.spike_times_sec));
fprintf('🔧 Optimization demo completed in %.2f seconds\n', opt_time);
fprintf('\nReady for production use with your actual experimental data!\n');