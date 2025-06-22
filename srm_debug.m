% SRM Debugging Script
clear all;
close all;

%% Load or generate test data
% You can either load your data or use the synthetic data from srm_simulation.m
% For this example, we'll use synthetic data
dt = 0.0001;          % Sampling interval (100 kHz)
T = 2;               % Shorter time window for debugging
N = round(T/dt);     % Number of time points
tau_ref_ms = 2.0;    % Refractory period

% Generate synthetic voltage trace
t = (0:N-1)*dt;
f = 2;               % Frequency of input (Hz)
Vm = -65 + 10*sin(2*pi*f*t) + 2*randn(size(t));  % Base voltage with noise

% Generate synthetic spike shape
spike_duration = 0.002;  % 2 ms spike
spike_points = round(spike_duration/dt);
t_spike = (0:spike_points-1)*dt;
spike_shape = 100*exp(-t_spike/0.0005).*sin(2*pi*1000*t_spike);  % Synthetic spike shape

% Add spikes to voltage trace
spike_times = [0.5, 0.8, 1.2, 1.5];  % Fewer spikes for debugging
spike_indices = round(spike_times/dt);
Vm_with_spikes = Vm;
for i = 1:length(spike_indices)
    idx = spike_indices(i);
    if idx + spike_points <= N
        Vm_with_spikes(idx:idx+spike_points-1) = Vm_with_spikes(idx:idx+spike_points-1) + spike_shape;
    end
end

% Detect spikes using simple threshold
threshold = -50;  % mV
elbow_indices = find(diff(Vm_with_spikes > threshold) == 1) + 1;
threshold_values = Vm_with_spikes(elbow_indices);

%% Create SRM and run debug analysis
cell_id = 'debug_cell';
cell_type = 'synthetic';
srm = SpikeResponseModel(Vm, Vm_with_spikes, dt, spike_shape, tau_ref_ms, ...
    elbow_indices, threshold_values, cell_id, cell_type);

% Test parameters
theta0_test = -50;  % Test threshold
A_test = 10;        % Test amplitude
tau_test = 0.01;    % Test time constant

% Define kernel function
kernel_fn = @(t) A_test * exp(-t / tau_test);

% Run simulation with debug points
[spikes, V_pred, threshold_trace, spike_times, spike_V] = srm.simulate2(theta0_test, kernel_fn);

%% Debug Analysis and Visualization
figure('Position', [100, 100, 1400, 1000]);

% 1. Voltage and Threshold Analysis
subplot(4,2,1);
plot(t, Vm_with_spikes, 'b-', 'LineWidth', 1);
hold on;
plot(t, V_pred, 'r-', 'LineWidth', 1);
% Robust spike plotting
if islogical(spikes)
    spike_idx = find(spikes);
elseif isnumeric(spikes) && ~isempty(spikes) && all(spikes > 0) && all(mod(spikes,1)==0)
    spike_idx = spikes;
else
    spike_idx = [];
end
if ~isempty(spike_idx)
    plot(t(spike_idx), V_pred(spike_idx), 'k.', 'MarkerSize', 10);
end
xlabel('Time (s)');
ylabel('Voltage (mV)');
title('Voltage Traces');
legend('Original', 'Predicted', 'Spikes');

% 2. Threshold Dynamics
subplot(4,2,2);
plot(t, threshold_trace, 'k-', 'LineWidth', 1);
hold on;
plot(t(elbow_indices), threshold_values, 'ro', 'MarkerSize', 6);
xlabel('Time (s)');
ylabel('Threshold (mV)');
title('Dynamic Threshold');

% 3. Spike Shape Analysis
subplot(4,2,3);
plot(t_spike*1000, spike_shape, 'b-', 'LineWidth', 2);
xlabel('Time (ms)');
ylabel('Amplitude');
title('Spike Shape');

% 4. Kernel Analysis
subplot(4,2,4);
t_kernel = 0:0.001:0.1;  % 100ms window
kernel = kernel_fn(t_kernel);
plot(t_kernel*1000, kernel, 'k-', 'LineWidth', 2);
xlabel('Time (ms)');
ylabel('Kernel Amplitude');
title('Spike Response Kernel');

% 5. ISI Analysis
subplot(4,2,5);
isi_true = diff(elbow_indices) * dt * 1000;  % Convert to ms
isi_pred = diff(find(spikes)) * dt * 1000;
histogram(isi_true, 'FaceColor', 'b', 'FaceAlpha', 0.5);
hold on;
histogram(isi_pred, 'FaceColor', 'r', 'FaceAlpha', 0.5);
xlabel('ISI (ms)');
ylabel('Count');
title('Inter-Spike Interval Distribution');
legend('True', 'Predicted');

% 6. Threshold vs Voltage
subplot(4,2,6);
scatter(Vm_with_spikes, threshold_trace, 10, 'filled');
hold on;
plot(xlim, xlim, 'k--');
xlabel('Voltage (mV)');
ylabel('Threshold (mV)');
title('Threshold vs Voltage');
axis square;

% 7. Spike Timing Error
subplot(4,2,7);
spike_timing_error = zeros(length(elbow_indices), 1);
for i = 1:length(elbow_indices)
    if ~isempty(spike_idx)
        [~, idx] = min(abs(spike_idx - elbow_indices(i)));
        spike_timing_error(i) = (spike_idx(idx) - elbow_indices(i)) * dt * 1000;  % Convert to ms
    else
        spike_timing_error(i) = NaN;
    end
end
histogram(spike_timing_error, 'FaceColor', 'g');
xlabel('Timing Error (ms)');
ylabel('Count');
title('Spike Timing Error');

% 8. Performance Metrics
subplot(4,2,8);
metrics = zeros(4,1);
metrics(1) = length(elbow_indices);  % True spikes
metrics(2) = sum(spikes);            % Predicted spikes
metrics(3) = sum(abs(spike_timing_error) < 1);  % Spikes within 1ms
metrics(4) = mean(abs(spike_timing_error));     % Mean timing error

bar(metrics);
set(gca, 'XTickLabel', {'True Spikes', 'Predicted', 'Within 1ms', 'Mean Error (ms)'});
title('Performance Metrics');
xtickangle(45);

% Print detailed debug information
fprintf('\nSRM Debug Information:\n');
fprintf('=====================\n');
fprintf('Time window: %.1f s\n', T);
fprintf('Sampling rate: %.1f kHz\n', 1/dt/1000);
fprintf('Number of true spikes: %d\n', length(elbow_indices));
fprintf('Number of predicted spikes: %d\n', sum(spikes));
fprintf('Mean ISI (true): %.2f ms\n', mean(isi_true));
fprintf('Mean ISI (predicted): %.2f ms\n', mean(isi_pred));
fprintf('Mean timing error: %.2f ms\n', mean(abs(spike_timing_error)));
fprintf('Spikes within 1ms: %d (%.1f%%)\n', sum(abs(spike_timing_error) < 1), ...
    100*sum(abs(spike_timing_error) < 1)/length(elbow_indices));

% Save debug figure
saveas(gcf, 'srm_debug_results.png');

%% Generate synthetic trace using simulateFast and recover parameters
% Define ground truth parameters
A_true = 30;         % Ground truth kernel amplitude
tau_true = 0.015;    % Ground truth kernel time constant (s)
theta0_true = -50;   % Ground truth threshold (mV)

% Define kernel function for ground truth
kernel_fn_true = @(t) A_true * exp(-t / tau_true);

% Test kernel function output
t_test = 0:0.001:0.1;  % 100ms window
kernel_output = kernel_fn_true(t_test);

% Verify kernel output
fprintf('\nKernel Function Test:\n');
fprintf('====================\n');
fprintf('Input time vector length: %d\n', length(t_test));
fprintf('Output kernel vector length: %d\n', length(kernel_output));
fprintf('Initial value (t=0): %.2f mV\n', kernel_output(1));
fprintf('Value at t=tau: %.2f mV\n', kernel_output(round(tau_true/0.001) + 1));
fprintf('Final value (t=100ms): %.2f mV\n', kernel_output(end));

% Check if all values are positive
if all(kernel_output > 0)
    fprintf('All values positive: Yes\n');
else
    fprintf('All values positive: No\n');
end

% Check if values are monotonically decreasing
if all(diff(kernel_output) <= 0)
    fprintf('Monotonically decreasing: Yes\n');
else
    fprintf('Monotonically decreasing: No\n');
end

% Generate synthetic trace using simulateFast and recover parameters
[spikes_true, V_pred_true, ~, ~, ~] = srm.simulateFast(theta0_true, kernel_fn_true);

% Define loss function for parameter recovery
loss_fn = @(params) srm.loss_fn_exp_holdout(params(1), params(2), params(3));

% Initial guess for parameters
params_init = [theta0_true, A_true, tau_true];

% Use fminsearch to recover parameters
options = optimset('Display', 'iter', 'MaxIter', 100);
params_recovered = fminsearch(loss_fn, params_init, options);

% Print recovered parameters
fprintf('\nParameter Recovery Results:\n');
fprintf('==========================\n');
fprintf('Ground Truth: theta0 = %.2f, A = %.2f, tau = %.4f\n', theta0_true, A_true, tau_true);
fprintf('Recovered:    theta0 = %.2f, A = %.2f, tau = %.4f\n', params_recovered(1), params_recovered(2), params_recovered(3));

% Simulate with recovered parameters
kernel_fn_recovered = @(t) params_recovered(2) * exp(-t / params_recovered(3));
[spikes_recovered, V_pred_recovered, ~, ~, ~] = srm.simulateFast(params_recovered(1), kernel_fn_recovered);

% Plot ground truth vs recovered
figure('Position', [100, 100, 1200, 400]);
subplot(1,2,1);
plot(t, V_pred_true, 'b-', 'LineWidth', 1);
hold on;
plot(t(spikes_true), V_pred_true(spikes_true), 'k.', 'MarkerSize', 10);
xlabel('Time (s)');
ylabel('Voltage (mV)');
title('Ground Truth Trace');
legend('Voltage', 'Spikes');

subplot(1,2,2);
plot(t, V_pred_recovered, 'r-', 'LineWidth', 1);
hold on;
plot(t(spikes_recovered), V_pred_recovered(spikes_recovered), 'k.', 'MarkerSize', 10);
xlabel('Time (s)');
ylabel('Voltage (mV)');
title('Recovered Trace');
legend('Voltage', 'Spikes');

% Save recovery figure
saveas(gcf, 'srm_parameter_recovery.png'); 