%% Real Experimental Data Analysis with Dynamic Exponential Threshold - simulateFast ONLY
% Complete workflow for analyzing real patch-clamp data using SpikeResponseModel
% Modified to use ONLY simulateFast for all simulations
% Assumes: Vm_all (raw data), Vm_cleaned (spike-removed data) are loaded

fprintf('=== REAL EXPERIMENTAL DATA ANALYSIS (simulateFast ONLY) ===\n');

%% === CHECK DATA AVAILABILITY ===
if ~exist('Vm_all', 'var') || ~exist('Vm_cleaned', 'var')
    error(['Please load your experimental data first:\n' ...
           '  - Vm_all: Raw voltage trace with spikes\n' ...
           '  - Vm_cleaned: Subthreshold voltage (spikes removed)\n' ...
           'Example: load(''your_data.mat'');\n' ...
           'Note: dt will be set to 1/10000 (10 kHz sampling)']);
end

% Check for your spike detection function
if ~exist('detect_spike_initiation_elbow', 'file')
    error(['Please ensure detect_spike_initiation_elbow.m is in your MATLAB path.\n' ...
           'This function is required for spike detection.']);
end

fprintf('=== REAL EXPERIMENTAL DATA ANALYSIS ===\n');
fprintf('Data loaded successfully!\n');
fprintf('Raw data length: %d samples\n', length(Vm_all));
fprintf('Cleaned data length: %d samples\n', length(Vm_cleaned));

%% === EXTRACT EXPERIMENTAL PARAMETERS ===
% Set sampling parameters for 10 kHz data
dt = 1/10000;  % 10 kHz sampling = 100 μs intervals
sampling_rate = 1/dt;

% Calculate time parameters
t_total = (length(Vm_all) - 1) * dt;
t_vec = (0:length(Vm_all)-1) * dt;

fprintf('\nExperimental parameters:\n');
fprintf('  Sampling rate: %.1f kHz\n', sampling_rate/1000);
fprintf('  Time step (dt): %.0f μs\n', dt*1e6);
fprintf('  Recording duration: %.2f s\n', t_total);
fprintf('  Data points: %d samples\n', length(Vm_all));
fprintf('  Voltage range (raw): %.1f to %.1f mV\n', min(Vm_all), max(Vm_all));
fprintf('  Voltage range (cleaned): %.1f to %.1f mV\n', min(Vm_cleaned), max(Vm_cleaned));

%% === SPIKE DETECTION USING YOUR ELBOW FUNCTION ===
fprintf('\n=== SPIKE DETECTION ===\n');

% Parameters for your elbow detection function
vm_thresh = -20;        % Voltage threshold for peak detection (mV)
d2v_thresh = 10;        % Second derivative threshold for elbow detection
search_back_ms = 2;     % Search back window in ms
plot_flag = true;       % Show detection plots

fprintf('Elbow detection parameters:\n');
fprintf('  Voltage threshold: %.1f mV\n', vm_thresh);
fprintf('  d²V/dt² threshold: %.1f mV/ms²\n', d2v_thresh);
fprintf('  Search back window: %.1f ms\n', search_back_ms);

% Run your spike detection function
[elbow_indices, spike_peaks, isi, avg_spike] = detect_spike_initiation_elbow(...
    Vm_all, dt, vm_thresh, d2v_thresh, search_back_ms, plot_flag);

% Calculate threshold values at spike initiation points
if isempty(elbow_indices)
    error('No spikes detected! Try adjusting detection parameters:\n - Lower vm_thresh\n - Lower d2v_thresh\n - Increase search_back_ms');
end

threshold_values = Vm_all(elbow_indices);  % Voltage at spike initiation

fprintf('\nSpike detection results:\n');
fprintf('  Detected elbow points: %d\n', length(elbow_indices));
fprintf('  Detected spike peaks: %d\n', length(spike_peaks));
fprintf('  Average firing rate: %.2f Hz\n', length(elbow_indices)/t_total);
fprintf('  Threshold range: %.1f to %.1f mV\n', min(threshold_values), max(threshold_values));
fprintf('  Mean threshold: %.1f ± %.1f mV\n', mean(threshold_values), std(threshold_values));

if ~isempty(isi)
    fprintf('  Mean ISI: %.1f ± %.1f ms\n', mean(isi), std(isi));
    fprintf('  ISI range: %.1f to %.1f ms\n', min(isi), max(isi));
end

%% === SPIKE SHAPE ANALYSIS ===
fprintf('\n=== SPIKE SHAPE ANALYSIS ===\n');

% The detection function provides avg_spike - use it directly
fprintf('Spike shape analysis (from detect_spike_initiation_elbow):\n');
fprintf('  Average spike calculated from %d spikes\n', length(elbow_indices));

% Validate that avg_spike was properly returned
if isempty(avg_spike) || length(avg_spike) < 2
    fprintf('  Warning: avg_spike not properly returned from detection function\n');
    % Create a simple spike template as fallback
    spike_duration_ms = 2;  % 2 ms spike
    spike_samples = round(spike_duration_ms / 1000 / dt);
    t_spike = (0:spike_samples-1) * dt * 1000;
    avg_spike = 20 * exp(-t_spike / 0.5);  % Simple exponential decay
    fprintf('  Using fallback spike template: %d samples\n', length(avg_spike));
else
    % Use the avg_spike returned by the detection function
    fprintf('  Spike waveform length: %d samples (%.1f ms)\n', length(avg_spike), length(avg_spike)*dt*1000);
    
    spike_amplitude = max(avg_spike) - min(avg_spike);
    fprintf('  Average spike amplitude: %.1f mV\n', spike_amplitude);
    
    % Find peak and half-width
    [peak_val, peak_idx] = max(avg_spike);
    trough_val = min(avg_spike);
    half_max = trough_val + (peak_val - trough_val) / 2;
    
    % Find half-width points
    pre_peak = avg_spike(1:peak_idx);
    post_peak = avg_spike(peak_idx:end);
    
    pre_half_idx = find(pre_peak <= half_max, 1, 'last');
    post_half_idx = find(post_peak <= half_max, 1, 'first');
    
    if ~isempty(pre_half_idx) && ~isempty(post_half_idx)
        half_width_samples = (post_half_idx + peak_idx - 1) - pre_half_idx;
        half_width_ms = half_width_samples * dt * 1000;
        fprintf('  Spike half-width: %.2f ms\n', half_width_ms);
    end
    
    fprintf('  Peak amplitude: %.1f mV\n', peak_val);
    fprintf('  Trough amplitude: %.1f mV\n', trough_val);
end

%% === CREATE SPIKERESPONSEMODEL OBJECT ===
fprintf('\n=== CREATING SPIKERESPONSEMODEL ===\n');

% Set refractory period
tau_ref_ms = 2.0;  % 2 ms absolute refractory period

% Create cell identifiers
if exist('cell_id', 'var')
    cell_name = cell_id;
else
    cell_name = 'Experimental-Cell';
end

if exist('cell_type', 'var')
    cell_type_name = cell_type;
else
    cell_type_name = 'Unknown';
end

% Create the model
model = SpikeResponseModel( ...
    Vm_cleaned, ...          % Subthreshold voltage trace
    Vm_all, ...              % Raw voltage trace
    dt, ...                  % Time step
    avg_spike, ...           % Average spike waveform
    tau_ref_ms, ...          % Refractory period
    elbow_indices, ...       % Spike initiation indices
    threshold_values, ...    % Threshold values at spikes
    cell_name, ...           % Cell identifier
    cell_type_name ...       % Cell type
);

fprintf('SpikeResponseModel created successfully:\n');
fprintf('  Cell ID: %s\n', cell_name);
fprintf('  Cell type: %s\n', cell_type_name);
fprintf('  Spikes: %d\n', length(elbow_indices));

%% === PARAMETER OPTIMIZATION - EXPONENTIAL KERNEL ===
fprintf('\n=== EXPONENTIAL THRESHOLD OPTIMIZATION WITH SIMULATEFAST ===\n');

% Define parameter search ranges based on literature (Jolivet et al., 2006)
% theta0: baseline threshold (typically -62 to -38 mV)
% A: adaptation amplitude (typically 2 to 12 mV) 
% tau: adaptation time constant (typically 15 to 71 ms)

% Initial parameter estimates for exponential kernel
theta0_range = [min(threshold_values)-10, max(threshold_values)+5];
theta0_init = mean(threshold_values) - 2;  % Start slightly below mean threshold

A_range = [1, 15];  % mV
A_init = 5;         % Start with moderate adaptation

tau_range = [0.005, 0.1];  % 5-100 ms
tau_init = 0.03;           % Start with 30 ms

init_params_exp = [theta0_init, A_init, tau_init];

fprintf('Exponential kernel optimization setup:\n');
fprintf('  Initial θ₀: %.1f mV (range: %.1f to %.1f mV)\n', theta0_init, theta0_range);
fprintf('  Initial A:  %.1f mV (range: %.1f to %.1f mV)\n', A_init, A_range);
fprintf('  Initial τ:  %.1f ms (range: %.1f to %.1f ms)\n', tau_init*1000, tau_range*1000);

% Victor-Purpura distance parameter
vp_q = 4;  % Standard value for spike timing precision

% Define FAST optimization objective using simulateFast ONLY
fprintf('\nOptimizing EXPONENTIAL kernel using simulateFast...\n');

% Fast loss function for exponential kernel
fast_loss_exp_fn = @(params) compute_fast_vp_loss_exponential(params, model, vp_q);

% Set optimization options for fminsearch
options = optimset('Display', 'iter', 'MaxFunEvals', 1000, 'MaxIter', 500, ...
                   'TolX', 1e-6, 'TolFun', 1e-4);

fprintf('Using fminsearch with simulateFast for exponential kernel...\n');

% Run optimization with fminsearch - EXPONENTIAL
tic;
[opt_params_exp, final_vp_exp] = fminsearch(fast_loss_exp_fn, init_params_exp, options);
opt_time_exp = toc;

% Extract optimized parameters - exponential
theta0_opt_exp = opt_params_exp(1);
A_opt_exp = opt_params_exp(2);
tau_opt_exp = opt_params_exp(3);

fprintf('\n=== EXPONENTIAL KERNEL RESULTS ===\n');
fprintf('Optimization completed in %.1f seconds\n', opt_time_exp);
fprintf('Initial params: θ₀=%.1f, A=%.1f, τ=%.1f ms\n', init_params_exp(1), init_params_exp(2), init_params_exp(3)*1000);
fprintf('Optimized params: θ₀=%.1f, A=%.1f, τ=%.1f ms\n', theta0_opt_exp, A_opt_exp, tau_opt_exp*1000);
fprintf('Final VP distance: %.3f\n', final_vp_exp);

%% === PARAMETER OPTIMIZATION - LINEAR RISE + EXPONENTIAL DECAY KERNEL ===
fprintf('\n=== LINEAR RISE + EXPONENTIAL DECAY OPTIMIZATION ===\n');

% Initial parameter estimates for linear rise + exp decay kernel
% theta0: baseline threshold (same as before)
% A: adaptation amplitude (same range)
% T_rise: rise time constant (typically 1-10 ms)
% tau_decay: decay time constant (typically 15-71 ms)

T_rise_range = [0.001, 0.01];  % 1-10 ms
T_rise_init = 0.003;           % Start with 3 ms

tau_decay_range = [0.01, 0.1]; % 10-100 ms
tau_decay_init = 0.04;         % Start with 40 ms

init_params_linexp = [theta0_init, A_init, T_rise_init, tau_decay_init];

fprintf('Linear rise + exponential decay kernel optimization setup:\n');
fprintf('  Initial θ₀: %.1f mV\n', theta0_init);
fprintf('  Initial A:  %.1f mV\n', A_init);
fprintf('  Initial T_rise: %.1f ms (range: %.1f to %.1f ms)\n', T_rise_init*1000, T_rise_range*1000);
fprintf('  Initial τ_decay: %.1f ms (range: %.1f to %.1f ms)\n', tau_decay_init*1000, tau_decay_range*1000);

% Define optimization objective for linear rise + exp decay
fast_loss_linexp_fn = @(params) compute_fast_vp_loss_linexp(params, model, vp_q);

fprintf('\nOptimizing LINEAR RISE + EXP DECAY kernel using simulateFast...\n');

% Run optimization - LINEAR RISE + EXP DECAY
tic;
[opt_params_linexp, final_vp_linexp] = fminsearch(fast_loss_linexp_fn, init_params_linexp, options);
opt_time_linexp = toc;

% Extract optimized parameters - linear rise + exp decay
theta0_opt_linexp = opt_params_linexp(1);
A_opt_linexp = opt_params_linexp(2);
T_rise_opt_linexp = opt_params_linexp(3);
tau_decay_opt_linexp = opt_params_linexp(4);

fprintf('\n=== LINEAR RISE + EXP DECAY RESULTS ===\n');
fprintf('Optimization completed in %.1f seconds\n', opt_time_linexp);
fprintf('Initial params: θ₀=%.1f, A=%.1f, T_rise=%.1f ms, τ_decay=%.1f ms\n', ...
    init_params_linexp(1), init_params_linexp(2), init_params_linexp(3)*1000, init_params_linexp(4)*1000);
fprintf('Optimized params: θ₀=%.1f, A=%.1f, T_rise=%.1f ms, τ_decay=%.1f ms\n', ...
    theta0_opt_linexp, A_opt_linexp, T_rise_opt_linexp*1000, tau_decay_opt_linexp*1000);
fprintf('Final VP distance: %.3f\n', final_vp_linexp);

%% === COMPARE KERNELS AND SELECT BEST ===
fprintf('\n=== KERNEL COMPARISON ===\n');
fprintf('Exponential kernel VP distance: %.3f\n', final_vp_exp);
fprintf('Linear rise + exp decay VP distance: %.3f\n', final_vp_linexp);

if final_vp_exp < final_vp_linexp
    fprintf('🏆 EXPONENTIAL kernel performs better (lower VP distance)\n');
    best_kernel = 'exponential';
    theta0_opt = theta0_opt_exp;
    A_opt = A_opt_exp;
    tau_opt = tau_opt_exp;
    final_vp = final_vp_exp;
    opt_time = opt_time_exp;
    opt_params = opt_params_exp;
    kernel_opt = @(t) A_opt * exp(-t / tau_opt);
    kernel_params = [A_opt, tau_opt];
else
    fprintf('🏆 LINEAR RISE + EXP DECAY kernel performs better (lower VP distance)\n');
    best_kernel = 'linear_rise_exp_decay';
    theta0_opt = theta0_opt_linexp;
    A_opt = A_opt_linexp;
    T_rise_opt = T_rise_opt_linexp;
    tau_decay_opt = tau_decay_opt_linexp;
    final_vp = final_vp_linexp;
    opt_time = opt_time_linexp;
    opt_params = opt_params_linexp;
    % Correct piecewise kernel function handle
    kernel_opt = @(t) (t < T_rise_opt) .* (A_opt / T_rise_opt .* t) + ...
                      (t >= T_rise_opt) .* (A_opt * exp(-(t - T_rise_opt) / tau_decay_opt));
    kernel_params = [A_opt, T_rise_opt, tau_decay_opt];
end

improvement_pct = abs(final_vp_exp - final_vp_linexp) / max(final_vp_exp, final_vp_linexp) * 100;
fprintf('Performance improvement: %.1f%%\n', improvement_pct);

%% === VALIDATION AND SIMULATION FOR BOTH KERNELS ===
fprintf('\n=== VALIDATION FOR BOTH KERNELS ===\n');

% Run simulation with EXPONENTIAL kernel using simulateFast
fprintf('Running validation with EXPONENTIAL kernel using simulateFast...\n');
kernel_exp = @(t) A_opt_exp * exp(-t / tau_opt_exp);
[spikes_exp, V_pred_exp, threshold_trace_exp, spike_times_exp, spike_V_exp] = ...
    model.simulateFast(theta0_opt_exp, kernel_exp, 'profile', false);

% Run simulation with LINEAR RISE + EXP DECAY kernel using simulateFast  
fprintf('Running validation with LINEAR RISE + EXP DECAY kernel using simulateFast...\n');
kernel_linexp = @(t) (t < T_rise_opt_linexp) .* (A_opt_linexp / T_rise_opt_linexp .* t) + ...
                     (t >= T_rise_opt_linexp) .* (A_opt_linexp * exp(-(t - T_rise_opt_linexp) / tau_decay_opt_linexp));
[spikes_linexp, V_pred_linexp, threshold_trace_linexp, spike_times_linexp, spike_V_linexp] = ...
    model.simulateFast(theta0_opt_linexp, kernel_linexp, 'profile', false);

% Calculate performance metrics for both kernels
n_true_spikes = length(elbow_indices);
true_rate = n_true_spikes / t_total;

% Exponential kernel metrics
n_pred_spikes_exp = length(spike_times_exp);
pred_rate_exp = n_pred_spikes_exp / t_total;
rate_accuracy_exp = 100 * min(pred_rate_exp, true_rate) / max(pred_rate_exp, true_rate);

% Linear rise + exp decay kernel metrics
n_pred_spikes_linexp = length(spike_times_linexp);
pred_rate_linexp = n_pred_spikes_linexp / t_total;
rate_accuracy_linexp = 100 * min(pred_rate_linexp, true_rate) / max(pred_rate_linexp, true_rate);

fprintf('\nValidation results:\n');
fprintf('  True spikes: %d (%.2f Hz)\n', n_true_spikes, true_rate);
fprintf('  EXPONENTIAL: %d spikes (%.2f Hz, %.1f%% accuracy, VP=%.3f)\n', ...
    n_pred_spikes_exp, pred_rate_exp, rate_accuracy_exp, final_vp_exp);
fprintf('  LINEAR+EXP: %d spikes (%.2f Hz, %.1f%% accuracy, VP=%.3f)\n', ...
    n_pred_spikes_linexp, pred_rate_linexp, rate_accuracy_linexp, final_vp_linexp);

% Set which kernel results to use for the final performance testing
if strcmp(best_kernel, 'exponential')
    spikes_opt = spikes_exp;
    V_pred_opt = V_pred_exp;
    threshold_trace_opt = threshold_trace_exp;
    spike_times_opt = spike_times_exp;
    spike_V_opt = spike_V_exp;
    n_pred_spikes = n_pred_spikes_exp;
    pred_rate = pred_rate_exp;
else
    spikes_opt = spikes_linexp;
    V_pred_opt = V_pred_linexp;
    threshold_trace_opt = threshold_trace_linexp;
    spike_times_opt = spike_times_linexp;
    spike_V_opt = spike_V_linexp;
    n_pred_spikes = n_pred_spikes_linexp;
    pred_rate = pred_rate_linexp;
end

%% === PERFORMANCE TESTING - SIMULATEFAST ONLY ===
fprintf('\n=== PERFORMANCE TESTING ===\n');

% Test simulateFast performance only
n_fast_tests = 50;
fprintf('Testing simulateFast with %d simulations...\n', n_fast_tests);

tic;
for i = 1:n_fast_tests
    [~, ~, ~, ~, ~] = model.simulateFast(theta0_opt, kernel_opt, 'profile', false);
end
fast_time = toc;

% Calculate average time
avg_fast_time = 1000 * fast_time / n_fast_tests;

fprintf('Performance results:\n');
fprintf('  simulateFast: %.2f ms/simulation\n', avg_fast_time);
fprintf('  Total time for %d simulations: %.2f seconds\n', n_fast_tests, fast_time);

%% === COMPREHENSIVE VISUALIZATION FOR BOTH KERNELS ===
fprintf('\n=== CREATING DIAGNOSTIC PLOTS FOR BOTH KERNELS ===\n');

% Define zoom window for detailed view - focus on middle section with good activity
zoom_start = t_total * 0.3;  % Start at 30% of recording
zoom_duration = min(2.0, t_total * 0.4);  % Show 2 seconds or 40% of recording, whichever is smaller
zoom_xlim = [zoom_start, zoom_start + zoom_duration];

% Ensure zoom window contains some spikes for meaningful visualization
spike_times_sec = elbow_indices * dt;
spikes_in_window = spike_times_sec >= zoom_xlim(1) & spike_times_sec <= zoom_xlim(2);
n_spikes_in_zoom = sum(spikes_in_window);

% If few spikes in default window, find a better window
if n_spikes_in_zoom < 3 && length(spike_times_sec) > 5
    % Find a 2-second window with the most spikes
    best_spike_count = 0;
    best_start = zoom_xlim(1);
    
    for start_time = 0:0.5:(t_total-zoom_duration)
        test_window = [start_time, start_time + zoom_duration];
        spikes_in_test = sum(spike_times_sec >= test_window(1) & spike_times_sec <= test_window(2));
        if spikes_in_test > best_spike_count
            best_spike_count = spikes_in_test;
            best_start = start_time;
        end
    end
    zoom_xlim = [best_start, best_start + zoom_duration];
    fprintf('Adjusted zoom window to [%.2f, %.2f] s with %d spikes\n', ...
        zoom_xlim(1), zoom_xlim(2), best_spike_count);
else
    fprintf('Using zoom window [%.2f, %.2f] s with %d spikes\n', ...
        zoom_xlim(1), zoom_xlim(2), n_spikes_in_zoom);
end

%% === DIAGNOSTIC PLOTS FOR EXPONENTIAL KERNEL ===
fprintf('\n--- Creating diagnostic plots for EXPONENTIAL kernel ---\n');
kernel_params_exp = [A_opt_exp, tau_opt_exp];

model.diagnostics(V_pred_exp, threshold_trace_exp, spikes_exp, final_vp_exp, ...
    zoom_xlim, '', kernel_params_exp, spike_times_exp, spike_V_exp);

% Add kernel type to title
sgtitle(sprintf('EXPONENTIAL Kernel Diagnostics: %s (%s) | VP=%.3f', ...
    cell_name, cell_type_name, final_vp_exp), 'FontSize', 14, 'FontWeight', 'bold');

%% === DIAGNOSTIC PLOTS FOR LINEAR RISE + EXP DECAY KERNEL ===
fprintf('--- Creating diagnostic plots for LINEAR RISE + EXP DECAY kernel ---\n');
kernel_params_linexp = [A_opt_linexp, T_rise_opt_linexp, tau_decay_opt_linexp];

model.diagnostics(V_pred_linexp, threshold_trace_linexp, spikes_linexp, final_vp_linexp, ...
    zoom_xlim, '', kernel_params_linexp, spike_times_linexp, spike_V_linexp);

% Add kernel type to title  
sgtitle(sprintf('LINEAR RISE + EXP DECAY Kernel Diagnostics: %s (%s) | VP=%.3f', ...
    cell_name, cell_type_name, final_vp_linexp), 'FontSize', 14, 'FontWeight', 'bold');

%% === KERNEL COMPARISON PLOT ===
fprintf('--- Creating kernel comparison plot ---\n');
figure('Position', [200, 200, 1400, 600]);

% Plot both kernels side by side
t_kernel = 0:dt:0.15;  % 150 ms

subplot(1,3,1);
kernel_exp_response = A_opt_exp * exp(-t_kernel / tau_opt_exp);
plot(t_kernel*1000, kernel_exp_response, 'b-', 'LineWidth', 3);
xlabel('Time after spike (ms)');
ylabel('Threshold increase (mV)');
title(sprintf('Exponential Kernel\nA=%.1f mV, τ=%.1f ms\nVP=%.3f', ...
      A_opt_exp, tau_opt_exp*1000, final_vp_exp));
grid on;

subplot(1,3,2);
kernel_linexp_response = (t_kernel < T_rise_opt_linexp) .* (A_opt_linexp / T_rise_opt_linexp .* t_kernel) + ...
                        (t_kernel >= T_rise_opt_linexp) .* (A_opt_linexp * exp(-(t_kernel - T_rise_opt_linexp) / tau_decay_opt_linexp));
plot(t_kernel*1000, kernel_linexp_response, 'r-', 'LineWidth', 3);
xlabel('Time after spike (ms)');
ylabel('Threshold increase (mV)');
title(sprintf('Linear Rise + Exp Decay\nA=%.1f mV, T_{rise}=%.1f ms, τ_{decay}=%.1f ms\nVP=%.3f', ...
      A_opt_linexp, T_rise_opt_linexp*1000, tau_decay_opt_linexp*1000, final_vp_linexp));
grid on;

subplot(1,3,3);
plot(t_kernel*1000, kernel_exp_response, 'b-', 'LineWidth', 3); hold on;
plot(t_kernel*1000, kernel_linexp_response, 'r-', 'LineWidth', 3);
xlabel('Time after spike (ms)');
ylabel('Threshold increase (mV)');
title('Kernel Comparison');
legend('Exponential', 'Linear + Exp', 'Location', 'best');
grid on;

% Add winner annotation
if final_vp_exp < final_vp_linexp
    winner_text = sprintf('Winner: Exponential\n(%.1f%% better)', improvement_pct);
    text(0.05, 0.95, winner_text, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
         'BackgroundColor', [0.8 0.9 1], 'FontSize', 12, 'FontWeight', 'bold');
else
    winner_text = sprintf('Winner: Linear + Exp\n(%.1f%% better)', improvement_pct);
    text(0.05, 0.95, winner_text, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
         'BackgroundColor', [1 0.8 0.8], 'FontSize', 12, 'FontWeight', 'bold');
end

sgtitle(sprintf('Kernel Comparison: %s (%s)', cell_name, cell_type_name), ...
    'FontSize', 14, 'FontWeight', 'bold');

% Additional analysis plots
figure('Position', [150, 150, 1200, 800]);

% Subplot 1: Parameter sensitivity analysis
subplot(2,3,1);
param_sweep_ranges = {
    linspace(theta0_opt-5, theta0_opt+5, 11),
    linspace(A_opt*0.5, A_opt*1.5, 11),
    linspace(tau_opt*0.5, tau_opt*1.5, 11)
};
param_names = {'θ₀ (mV)', 'A (mV)', 'τ (s)'};

% Test sensitivity to theta0 using simulateFast
vp_sweep = zeros(size(param_sweep_ranges{1}));
for i = 1:length(param_sweep_ranges{1})
    test_params = [param_sweep_ranges{1}(i), A_opt, tau_opt];
    vp_sweep(i) = compute_fast_vp_loss(test_params, model, vp_q);
end

plot(param_sweep_ranges{1}, vp_sweep, 'o-', 'LineWidth', 2); hold on;
plot(theta0_opt, final_vp, 'ro', 'MarkerSize', 10, 'LineWidth', 3);
xlabel(param_names{1});
ylabel('VP Distance');
title('Sensitivity to θ₀');
grid on;

% Subplot 2: Threshold adaptation kernel
subplot(2,3,2);
t_kernel = 0:dt:0.2;  % 200 ms
if strcmp(best_kernel, 'exponential')
    kernel_response = A_opt_exp * exp(-t_kernel / tau_opt_exp);
    plot(t_kernel*1000, kernel_response, 'b-', 'LineWidth', 3);
    title(sprintf('Adaptation Kernel (Exponential)\nA=%.1f mV, τ=%.1f ms', A_opt_exp, tau_opt_exp*1000));
else
    kernel_response = (t_kernel < T_rise_opt_linexp) .* (A_opt_linexp / T_rise_opt_linexp .* t_kernel) + ...
                     (t_kernel >= T_rise_opt_linexp) .* (A_opt_linexp * exp(-(t_kernel - T_rise_opt_linexp) / tau_decay_opt_linexp));
    plot(t_kernel*1000, kernel_response, 'b-', 'LineWidth', 3);
    title(sprintf('Adaptation Kernel (LinExp)\nA=%.1f mV, T_{rise}=%.1f ms, τ_{decay}=%.1f ms', ...
          A_opt_linexp, T_rise_opt_linexp*1000, tau_decay_opt_linexp*1000));
end
xlabel('Time after spike (ms)');
ylabel('Threshold increase (mV)');
grid on;

% Subplot 3: Firing rate vs threshold relationship
subplot(2,3,3);
if length(elbow_indices) > 3
    % Calculate local firing rates around each spike
    window_size = round(0.5 / dt);  % 500 ms windows
    local_rates = [];
    local_thresholds = [];
    
    for i = 1:length(elbow_indices)
        spike_idx = elbow_indices(i);
        win_start = max(1, spike_idx - window_size);
        win_end = min(length(Vm_all), spike_idx + window_size);
        
        spikes_in_window = sum(elbow_indices >= win_start & elbow_indices <= win_end);
        rate = spikes_in_window / (2 * window_size * dt);  % Hz
        
        local_rates = [local_rates, rate];
        local_thresholds = [local_thresholds, threshold_values(i)];
    end
    
    scatter(local_rates, local_thresholds, 60, 'filled');
    xlabel('Local firing rate (Hz)');
    ylabel('Threshold (mV)');
    title('Rate-Threshold Relationship');
    
    % Fit linear relationship
    if length(local_rates) > 2
        p = polyfit(local_rates, local_thresholds, 1);
        rate_range = linspace(min(local_rates), max(local_rates), 100);
        hold on;
        plot(rate_range, polyval(p, rate_range), 'r--', 'LineWidth', 2);
        text(0.05, 0.95, sprintf('Slope: %.2f mV/Hz', p(1)), ...
            'Units', 'normalized', 'VerticalAlignment', 'top');
    end
    grid on;
end

% Subplot 4: ISI distribution comparison
subplot(2,3,4);
if length(elbow_indices) > 1 && length(spike_times_opt) > 1
    true_isis = diff(elbow_indices) * dt * 1000;  % ms
    pred_isis = diff(find(spikes_opt)) * dt * 1000;  % ms
    
    edges = 0:2:50;  % 0-50 ms in 2 ms bins
    histogram(true_isis, edges, 'FaceColor', 'b', 'FaceAlpha', 0.5); hold on;
    histogram(pred_isis, edges, 'FaceColor', 'r', 'FaceAlpha', 0.5);
    xlabel('ISI (ms)');
    ylabel('Count');
    title('ISI Distribution');
    legend('True', 'Predicted', 'Location', 'best');
    grid on;
end

% Subplot 5: Voltage histogram at spike times
subplot(2,3,5);
histogram(threshold_values, 15, 'FaceColor', 'g', 'FaceAlpha', 0.7);
xlabel('Threshold voltage (mV)');
ylabel('Count');
title('Threshold Distribution');
grid on;

% Subplot 6: Performance summary
subplot(2,3,6);
perf_metrics = [true_rate, pred_rate; n_true_spikes, n_pred_spikes; final_vp, avg_fast_time];
bar_data = perf_metrics';
bar(bar_data);
set(gca, 'XTickLabel', {'Rate (Hz)', 'Count', 'VP/Time(ms)'});
ylabel('Value');
title('Performance Summary');
legend('True/VP', 'Pred/Fast', 'Location', 'best');
grid on;

sgtitle(sprintf('Experimental Data Analysis (simulateFast): %s (%s)', cell_name, cell_type_name), ...
    'FontSize', 14, 'FontWeight', 'bold');

%% === SAVE RESULTS ===
fprintf('\n=== SAVING RESULTS ===\n');

% Save optimization results - both kernels
results = struct();
results.cell_info = struct('id', cell_name, 'type', cell_type_name);
results.experimental = struct('dt', dt, 'duration', t_total, 'n_spikes', n_true_spikes);
results.detection = struct('method', 'elbow_d2v', 'vm_thresh', vm_thresh, 'd2v_thresh', d2v_thresh, ...
                          'search_back_ms', search_back_ms, 'elbow_indices', elbow_indices, ...
                          'threshold_values', threshold_values, 'isi', isi);

% Store results for both kernels
results.optimization_exponential = struct('initial_params', init_params_exp, 'optimized_params', opt_params_exp, ...
                                         'vp_distance', final_vp_exp, 'optimization_time', opt_time_exp, ...
                                         'method', 'simulateFast_exponential');
                                         
results.optimization_linexp = struct('initial_params', init_params_linexp, 'optimized_params', opt_params_linexp, ...
                                    'vp_distance', final_vp_linexp, 'optimization_time', opt_time_linexp, ...
                                    'method', 'simulateFast_linear_rise_exp_decay');

% Best kernel results
results.best_kernel = struct('type', best_kernel, 'vp_distance', final_vp, ...
                            'improvement_percent', improvement_pct, 'kernel_params', kernel_params);

results.validation = struct('predicted_spikes', n_pred_spikes, 'rate_accuracy', ...
                           100 * min(pred_rate, true_rate) / max(pred_rate, true_rate));
results.performance = struct('simulateFast_time_ms', avg_fast_time, 'total_tests', n_fast_tests);
results.model = model;  % Save the complete model object

% Save to file
save_filename = sprintf('spike_analysis_fast_%s_%s.mat', cell_name, datestr(now, 'yyyymmdd_HHMMSS'));
save(save_filename, 'results', 'model', 'Vm_all', 'Vm_cleaned');

fprintf('Results saved to: %s\n', save_filename);

%% === COMPACT SUMMARY LOG ===
fprintf('\n' + string(repmat('=', 1, 50)) + '\n');
fprintf('SPIKE ANALYSIS SUMMARY LOG (simulateFast)\n');
fprintf(string(repmat('=', 1, 50)) + '\n');

% Create compact summary for easy copying - include both kernels
summary_log = sprintf([...
    'CELL: %s (%s) | DUR: %.1fs | RATE: %.1fHz\n' ...
    'SPIKES: %d detected | ISI: %.1f±%.1fms\n' ...
    'DETECTION: vm_thresh=%.1f, d2v_thresh=%.1f, search_back=%.1fms\n' ...
    'THRESHOLD_STATS: μ=%.1f±%.1fmV, range=[%.1f,%.1f]mV\n' ...
    'EXPONENTIAL: θ₀=%.1f, A=%.1f, τ=%.1fms | VP=%.3f\n' ...
    'LINEXP: θ₀=%.1f, A=%.1f, T_rise=%.1f, τ_decay=%.1fms | VP=%.3f\n' ...
    'BEST_KERNEL: %s | IMPROVEMENT: %.1f%%\n' ...
    'PERFORMANCE: simulateFast=%.1fms | Rate_acc=%.1f%% | Pred_spikes=%d\n' ...
    'METHOD: simulateFast_BOTH_KERNELS | Tests=%d\n' ...
    'SAVED: %s'], ...
    cell_name, cell_type_name, t_total, true_rate, ...
    n_true_spikes, mean(isi), std(isi), ...
    vm_thresh, d2v_thresh, search_back_ms, ...
    mean(threshold_values), std(threshold_values), min(threshold_values), max(threshold_values), ...
    theta0_opt_exp, A_opt_exp, tau_opt_exp*1000, final_vp_exp, ...
    theta0_opt_linexp, A_opt_linexp, T_rise_opt_linexp*1000, tau_decay_opt_linexp*1000, final_vp_linexp, ...
    best_kernel, improvement_pct, ...
    avg_fast_time, 100 * min(pred_rate, true_rate) / max(pred_rate, true_rate), n_pred_spikes, ...
    n_fast_tests, save_filename);

fprintf('%s\n', summary_log);
fprintf(string(repmat('=', 1, 50)) + '\n');

%% === FINAL SUMMARY ===
fprintf('\n=== ANALYSIS COMPLETE (simulateFast + KERNEL COMPARISON) ===\n');
fprintf('✅ Successfully analyzed experimental data for %s\n', cell_name);
fprintf('📊 Detected %d spikes (%.2f Hz)\n', n_true_spikes, true_rate);
fprintf('\n🔬 KERNEL COMPARISON RESULTS:\n');
fprintf('   Exponential kernel VP: %.3f\n', final_vp_exp);
fprintf('   Linear rise + exp decay VP: %.3f\n', final_vp_linexp);
fprintf('   🏆 Best kernel: %s (%.1f%% improvement)\n', best_kernel, improvement_pct);

if strcmp(best_kernel, 'exponential')
    fprintf('\n🎯 Best model (Exponential):\n');
    fprintf('   • Baseline threshold (θ₀): %.1f mV\n', theta0_opt_exp);
    fprintf('   • Adaptation amplitude (A): %.1f mV\n', A_opt_exp);
    fprintf('   • Adaptation time constant (τ): %.1f ms\n', tau_opt_exp*1000);
else
    fprintf('\n🎯 Best model (Linear rise + exp decay):\n');
    fprintf('   • Baseline threshold (θ₀): %.1f mV\n', theta0_opt_linexp);
    fprintf('   • Adaptation amplitude (A): %.1f mV\n', A_opt_linexp);
    fprintf('   • Rise time constant (T_rise): %.1f ms\n', T_rise_opt_linexp*1000);
    fprintf('   • Decay time constant (τ_decay): %.1f ms\n', tau_decay_opt_linexp*1000);
end

fprintf('⚡ Performance: simulateFast = %.1f ms/simulation\n', avg_fast_time);
fprintf('🔬 Model quality: VP distance = %.3f\n', final_vp);
fprintf('💾 Results saved to: %s\n', save_filename);
fprintf('🚀 Used simulateFast ONLY for all simulations\n');
fprintf('🔄 Compared both exponential and linear rise + exp decay kernels\n');

% Copy to clipboard
clipboard('copy', summary_log);
fprintf('📋 Summary copied to clipboard!\n');
fprintf('🔄 Paste this summary to continue analysis discussions.\n');


%% === FAST LOSS FUNCTIONS FOR BOTH KERNELS ===
function loss = compute_fast_vp_loss(params, model, q)
    % Fast Victor-Purpura loss using simulateFast ONLY - GENERIC (backward compatibility)
    % This is for cases where we need a simple exponential kernel function
    
    theta0 = params(1);
    A = params(2);
    tau = params(3);
    
    % Define exponential kernel
    kernel_fn = @(t) A * exp(-t / tau);
    
    % Use simulateFast ONLY for speed
    try
        [~, ~, ~, spike_times] = model.simulateFast(theta0, kernel_fn, 'profile', false);
        
        % Get true spike times
        true_spike_times = model.elbow_indices * model.dt;
        
        % Calculate Victor-Purpura distance
        loss = spkd_c(spike_times, true_spike_times, ...
            length(spike_times), length(true_spike_times), q);
        
        % Optional: Print progress (comment out for silent optimization)
        fprintf('GENERIC: θ₀=%.1f, A=%.1f, τ=%.1f → VP=%.3f | spikes=%d vs %d\n', ...
            theta0, A, tau*1000, loss, length(spike_times), length(true_spike_times));
            
    catch ME
        % Penalize failed simulations heavily
        loss = 1000;
        fprintf('GENERIC failed: θ₀=%.1f, A=%.1f, τ=%.1f → VP=1000 (penalty)\n', ...
            theta0, A, tau*1000);
    end
end

function loss = compute_fast_vp_loss_exponential(params, model, q)
    % Fast Victor-Purpura loss using simulateFast ONLY - EXPONENTIAL kernel
    
    theta0 = params(1);
    A = params(2);
    tau = params(3);
    
    % Define exponential kernel
    kernel_fn = @(t) A * exp(-t / tau);
    
    % Use simulateFast ONLY for speed
    try
        [~, ~, ~, spike_times] = model.simulateFast(theta0, kernel_fn, 'profile', false);
        
        % Get true spike times
        true_spike_times = model.elbow_indices * model.dt;
        
        % Calculate Victor-Purpura distance
        loss = spkd_c(spike_times, true_spike_times, ...
            length(spike_times), length(true_spike_times), q);
        
        % Optional: Print progress (comment out for silent optimization)
        fprintf('EXP: θ₀=%.1f, A=%.1f, τ=%.1f → VP=%.3f | spikes=%d vs %d\n', ...
            theta0, A, tau*1000, loss, length(spike_times), length(true_spike_times));
            
    catch ME
        % Penalize failed simulations heavily
        loss = 1000;
        fprintf('EXP failed: θ₀=%.1f, A=%.1f, τ=%.1f → VP=1000 (penalty)\n', ...
            theta0, A, tau*1000);
    end
end

function loss = compute_fast_vp_loss_linexp(params, model, q)
    % Fast Victor-Purpura loss using simulateFast ONLY - LINEAR RISE + EXP DECAY kernel
    
    theta0 = params(1);
    A = params(2);
    T_rise = params(3);
    tau_decay = params(4);
    
    % Define linear rise + exponential decay kernel - CORRECTED piecewise function
    kernel_fn = @(t) (t < T_rise) .* (A / T_rise .* t) + ...
                     (t >= T_rise) .* (A * exp(-(t - T_rise) / tau_decay));
    
    % Use simulateFast ONLY for speed
    try
        [~, ~, ~, spike_times] = model.simulateFast(theta0, kernel_fn, 'profile', false);
        
        % Get true spike times
        true_spike_times = model.elbow_indices * model.dt;
        
        % Calculate Victor-Purpura distance
        loss = spkd_c(spike_times, true_spike_times, ...
            length(spike_times), length(true_spike_times), q);
        
        % Optional: Print progress (comment out for silent optimization)
        fprintf('LINEXP: θ₀=%.1f, A=%.1f, T_rise=%.1f, τ_decay=%.1f → VP=%.3f | spikes=%d vs %d\n', ...
            theta0, A, T_rise*1000, tau_decay*1000, loss, length(spike_times), length(true_spike_times));
            
    catch ME
        % Penalize failed simulations heavily
        loss = 1000;
        fprintf('LINEXP failed: θ₀=%.1f, A=%.1f, T_rise=%.1f, τ_decay=%.1f → VP=1000 (penalty)\n', ...
            theta0, A, T_rise*1000, tau_decay*1000);
    end
end