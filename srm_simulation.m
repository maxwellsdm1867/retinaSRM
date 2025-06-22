% SRM Simulation Script
clear all;
close all;

%% Generate synthetic data
% Parameters
dt = 0.0001;          % Sampling interval (100 kHz)
T = 10;              % Total simulation time (seconds)
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
spike_times = [1.5, 2.3, 3.1, 4.2, 5.0, 6.1, 7.2, 8.3, 9.1];  % Example spike times
spike_indices = round(spike_times/dt);
Vm_with_spikes = Vm;
for i = 1:length(spike_indices)
    idx = spike_indices(i);
    if idx + spike_points <= N
        Vm_with_spikes(idx:idx+spike_points-1) = Vm_with_spikes(idx:idx+spike_points-1) + spike_shape';
    end
end

% Detect spikes using simple threshold
threshold = -20;  % mV
elbow_indices = find(diff(Vm_with_spikes > threshold) == 1) + 1;
threshold_values = Vm_with_spikes(elbow_indices);

%% Create and fit SRM
% Initialize SRM
cell_id = 'synthetic_cell';
cell_type = 'synthetic';
srm = SpikeResponseModel(Vm, Vm_with_spikes, dt, spike_shape, tau_ref_ms, ...
    elbow_indices, threshold_values, cell_id, cell_type);

% Define initial parameters
theta0_init = -20;  % Initial threshold
A_init = 10;        % Initial amplitude
tau_init = 0.01;    % Initial time constant

% Optimization options
options = optimset('Display', 'iter', 'MaxIter', 100);

% Fit exponential kernel
params_exp = [theta0_init, A_init, tau_init];
[params_opt, fval] = fminsearch(@(p) srm.vp_loss_exponential(p, 4), params_exp, options);

% Extract optimized parameters
theta0_opt = params_opt(1);
A_opt = params_opt(2);
tau_opt = params_opt(3);

% Define kernel function with optimized parameters
kernel_fn = @(t) A_opt * exp(-t / tau_opt);

% Simulate with optimized parameters
[spikes, V_pred, threshold_trace, spike_times, spike_V] = srm.simulate2(theta0_opt, kernel_fn);

%% Visualization
figure('Position', [100, 100, 1200, 800]);

% Plot voltage traces
subplot(3,1,1);
plot(t, Vm_with_spikes, 'b-', 'LineWidth', 1);
hold on;
plot(t, V_pred, 'r-', 'LineWidth', 1);
plot(t(spikes), V_pred(spikes), 'k.', 'MarkerSize', 10);
xlabel('Time (s)');
ylabel('Voltage (mV)');
title('Voltage Traces');
legend('Original', 'Predicted', 'Spikes');

% Plot threshold
subplot(3,1,2);
plot(t, threshold_trace, 'k-', 'LineWidth', 1);
hold on;
plot(t(elbow_indices), threshold_values, 'ro', 'MarkerSize', 6);
xlabel('Time (s)');
ylabel('Threshold (mV)');
title('Dynamic Threshold');

% Plot kernel
subplot(3,1,3);
t_kernel = 0:0.001:0.1;  % 100ms window
kernel = kernel_fn(t_kernel);
plot(t_kernel*1000, kernel, 'k-', 'LineWidth', 2);
xlabel('Time (ms)');
ylabel('Kernel Amplitude');
title('Spike Response Kernel');

% Print results
fprintf('\nSRM Fitting Results:\n');
fprintf('Victor-Purpura distance: %.4f\n', fval);
fprintf('\nParameters:\n');
fprintf('  theta0 = %.2f mV\n', theta0_opt);
fprintf('  A = %.2f mV\n', A_opt);
fprintf('  tau = %.4f s\n', tau_opt);
fprintf('\nSpike Statistics:\n');
fprintf('  True spikes: %d\n', length(elbow_indices));
fprintf('  Predicted spikes: %d\n', sum(spikes)); 