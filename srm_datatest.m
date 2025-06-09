%% Real Experimental Data Analysis with Dynamic Exponential Threshold
% Complete workflow for analyzing real patch-clamp data using SpikeResponseModel
% Assumes: Vm_all (raw data), Vm_cleaned (spike-removed data) are loaded

fprintf('=== REAL EXPERIMENTAL DATA ANALYSIS ===\n');

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
d2v_thresh = 0.5;       % Second derivative threshold for elbow detection
search_back_ms = 2;     % Search back window in ms
visualize = true;       % Show detection plots

fprintf('Elbow detection parameters:\n');
fprintf('  Voltage threshold: %.1f mV\n', vm_thresh);
fprintf('  d²V/dt² threshold: %.1f mV/ms²\n', d2v_thresh);
fprintf('  Search back window: %.1f ms\n', search_back_ms);

% Run your spike detection function
[elbow_indices, spike_peaks, isi, avg_spike] = detect_spike_initiation_elbow(...
    Vm_all, dt, vm_thresh, d2v_thresh, search_back_ms, visualize);

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

% Your detection function already provides avg_spike and analysis
fprintf('Spike shape analysis (from detection function):\n');
fprintf('  Average spike calculated from %d spikes\n', length(elbow_indices));
fprintf('  Spike waveform length: %d samples (%.1f ms)\n', length(avg_spike), length(avg_spike)*dt*1000);

if length(avg_spike) > 1
    spike_amplitude = max(avg_spike) - min(avg_spike);
    fprintf('  Average spike amplitude: %.1f mV\n', spike_amplitude);
    
    % Find peak and half-width
    [~, peak_idx] = max(avg_spike);
    half_max = (max(avg_spike) + min(avg_spike)) / 2;
    
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
else
    fprintf('  Warning: Average spike shape not properly calculated\n');
    % Create a simple spike template as fallback
    spike_duration_ms = 2;  % 2 ms spike
    spike_samples = round(spike_duration_ms / 1000 / dt);
    t_spike = (0:spike_samples-1) * dt * 1000;
    avg_spike = 20 * exp(-t_spike / 0.5);  % Simple exponential decay
    fprintf('  Using fallback spike template: %d samples\n', length(avg_spike));
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

%% === PARAMETER OPTIMIZATION USING SIMULATEFAST ===
fprintf('\n=== EXPONENTIAL THRESHOLD OPTIMIZATION WITH SIMULATEFAST ===\n');

% Define parameter search ranges based on literature (Jolivet et al., 2006)
% theta0: baseline threshold (typically -62 to -38 mV)
% A: adaptation amplitude (typically 2 to 12 mV) 
% tau: adaptation time constant (typically 15 to 71 ms)

% Initial parameter estimates
theta0_range = [min(threshold_values)-10, max(threshold_values)+5];
theta0_init = mean(threshold_values) - 2;  % Start slightly below mean threshold

A_range = [1, 15];  % mV
A_init = 5;         % Start with moderate adaptation

tau_range = [0.005, 0.1];  % 5-100 ms
tau_init = 0.03;           % Start with 30 ms

init_params = [theta0_init, A_init, tau_init];

fprintf('Parameter optimization setup:\n');
fprintf('  Initial θ₀: %.1f mV (range: %.1f to %.1f mV)\n', theta0_init, theta0_range);
fprintf('  Initial A:  %.1f mV (range: %.1f to %.1f mV)\n', A_init, A_range);
fprintf('  Initial τ:  %.1f ms (range: %.1f to %.1f ms)\n', tau_init*1000, tau_range*1000);

% Victor-Purpura distance parameter
vp_q = 4;  % Standard value for spike timing precision

% Define FAST optimization objective using simulateFast
fprintf('\nCreating fast optimization objective using simulateFast...\n');

% Fast loss function using simulateFast instead of simulate
fast_loss_fn = @(params) compute_fast_vp_loss(params, model, vp_q);

% Set optimization options for fminsearch
options = optimset('Display', 'iter', 'MaxFunEvals', 1000, 'MaxIter', 500, ...
                   'TolX', 1e-6, 'TolFun', 1e-4);

fprintf('Using fminsearch with simulateFast for optimization...\n');
fprintf('Max function evaluations: 1000\n');
fprintf('Max iterations: 500\n');

% Run optimization with fminsearch
tic;
[opt_params, final_vp] = fminsearch(fast_loss_fn, init_params, options);
opt_time = toc;

% Extract optimized parameters
theta0_opt = opt_params(1);
A_opt = opt_params(2);
tau_opt = opt_params(3);

fprintf('\n=== OPTIMIZATION RESULTS ===\n');
fprintf('Optimization completed in %.1f seconds\n', opt_time);
fprintf('Initial parameters: θ₀=%.1f, A=%.1f, τ=%.1f ms\n', init_params(1), init_params(2), init_params(3)*1000);
fprintf('Optimized parameters: θ₀=%.1f, A=%.1f, τ=%.1f ms\n', theta0_opt, A_opt, tau_opt*1000);
fprintf('Final VP distance: %.3f\n', final_vp);

%% === VALIDATION AND SIMULATION ===
fprintf('\n=== VALIDATION ===\n');

% Define optimized kernel
kernel_opt = @(t) A_opt * exp(-t / tau_opt);

% Run simulation with optimized parameters using simulateFast
fprintf('Running final validation with simulateFast...\n');
[spikes_opt, V_pred_opt, threshold_trace_opt, spike_times_opt, spike_V_opt] = ...
    model.simulateFast(theta0_opt, kernel_opt, 'profile', false);

% Calculate performance metrics
n_true_spikes = length(elbow_indices);
n_pred_spikes = length(spike_times_opt);
true_rate = n_true_spikes / t_total;
pred_rate = n_pred_spikes / t_total;

fprintf('Validation results:\n');
fprintf('  True spikes: %d (%.2f Hz)\n', n_true_spikes, true_rate);
fprintf('  Predicted spikes: %d (%.2f Hz)\n', n_pred_spikes, pred_rate);
fprintf('  Rate accuracy: %.1f%%\n', 100 * min(pred_rate, true_rate) / max(pred_rate, true_rate));
fprintf('  VP distance: %.3f\n', final_vp);

%% === PERFORMANCE TESTING ===
fprintf('\n=== PERFORMANCE TESTING ===\n');

% Test simulateFast performance
n_fast_tests = 20;
fprintf('Testing simulateFast with %d simulations...\n', n_fast_tests);

tic;
for i = 1:n_fast_tests
    [~, ~, ~, ~, ~] = model.simulateFast(theta0_opt, kernel_opt, 'profile', false);
end
fast_time = toc;

% Test standard simulate performance (fewer tests due to speed)
n_std_tests = 5;
fprintf('Testing standard simulate with %d simulations...\n', n_std_tests);

tic;
for i = 1:n_std_tests
    [~, ~, ~, ~, ~] = model.simulate(theta0_opt, kernel_opt);
end
std_time = toc;

% Calculate speedup
avg_fast_time = 1000 * fast_time / n_fast_tests;
avg_std_time = 1000 * std_time / n_std_tests;
speedup = avg_std_time / avg_fast_time;

fprintf('Performance results:\n');
fprintf('  Standard simulate: %.2f ms/simulation\n', avg_std_time);
fprintf('  simulateFast: %.2f ms/simulation\n', avg_fast_time);
fprintf('  Speedup: %.1fx\n', speedup);

%% === COMPREHENSIVE VISUALIZATION ===
fprintf('\n=== CREATING DIAGNOSTIC PLOTS ===\n');

% Use model's built-in diagnostics
zoom_xlim = [t_total*0.2, t_total*0.8];  % Middle portion of recording
kernel_params = [A_opt, tau_opt];

model.diagnostics(V_pred_opt, threshold_trace_opt, spikes_opt, final_vp, ...
    zoom_xlim, '', kernel_params, spike_times_opt, spike_V_opt);

% Additional analysis plots
figure('Position', [150, 150, 1200, 800]);

% Subplot 1: Parameter sensitivity
subplot(2,3,1);
param_sweep_ranges = {
    linspace(theta0_opt-5, theta0_opt+5, 11),
    linspace(A_opt*0.5, A_opt*1.5, 11),
    linspace(tau_opt*0.5, tau_opt*1.5, 11)
};
param_names = {'θ₀ (mV)', 'A (mV)', 'τ (s)'};

% Test sensitivity to theta0
vp_sweep = zeros(size(param_sweep_ranges{1}));
for i = 1:length(param_sweep_ranges{1})
    test_params = [param_sweep_ranges{1}(i), A_opt, tau_opt];
    vp_sweep(i) = model.vp_loss_exponential(test_params, vp_q);
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
kernel_response = A_opt * exp(-t_kernel / tau_opt);
plot(t_kernel*1000, kernel_response, 'b-', 'LineWidth', 3);
xlabel('Time after spike (ms)');
ylabel('Threshold increase (mV)');
title(sprintf('Adaptation Kernel\nA=%.1f mV, τ=%.1f ms', A_opt, tau_opt*1000));
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
perf_metrics = [true_rate, pred_rate; n_true_spikes, n_pred_spikes; final_vp, speedup];
bar_data = perf_metrics';
bar(bar_data);
set(gca, 'XTickLabel', {'Rate (Hz)', 'Count', 'VP/Speedup'});
ylabel('Value');
title('Performance Summary');
legend('True/Standard', 'Pred/Fast', 'Location', 'best');
grid on;

sgtitle(sprintf('Experimental Data Analysis: %s (%s)', cell_name, cell_type_name), ...
    'FontSize', 14, 'FontWeight', 'bold');

%% === SAVE RESULTS ===
fprintf('\n=== SAVING RESULTS ===\n');

% Save optimization results
results = struct();
results.cell_info = struct('id', cell_name, 'type', cell_type_name);
results.experimental = struct('dt', dt, 'duration', t_total, 'n_spikes', n_true_spikes);
results.detection = struct('method', 'elbow_d2v', 'vm_thresh', vm_thresh, 'd2v_thresh', d2v_thresh, ...
                          'search_back_ms', search_back_ms, 'elbow_indices', elbow_indices, ...
                          'threshold_values', threshold_values, 'isi', isi);
results.optimization = struct('initial_params', init_params, 'optimized_params', opt_params, ...
                             'vp_distance', final_vp, 'optimization_time', opt_time);
results.validation = struct('predicted_spikes', n_pred_spikes, 'rate_accuracy', ...
                           100 * min(pred_rate, true_rate) / max(pred_rate, true_rate));
results.performance = struct('simulate_time_ms', avg_std_time, 'simulateFast_time_ms', avg_fast_time, ...
                            'speedup_factor', speedup);
results.model = model;  % Save the complete model object

% Save to file
save_filename = sprintf('spike_analysis_%s_%s.mat', cell_name, datestr(now, 'yyyymmdd_HHMMSS'));
save(save_filename, 'results', 'model', 'Vm_all', 'Vm_cleaned');

fprintf('Results saved to: %s\n', save_filename);

%% === FINAL SUMMARY ===
fprintf('\n=== ANALYSIS COMPLETE ===\n');
fprintf('✅ Successfully analyzed experimental data for %s\n', cell_name);
fprintf('📊 Detected %d spikes (%.2f Hz)\n', n_true_spikes, true_rate);
fprintf('🎯 Optimized exponential threshold model:\n');
fprintf('   • Baseline threshold (θ₀): %.1f mV\n', theta0_opt);
fprintf('   • Adaptation amplitude (A): %.1f mV\n', A_opt);
fprintf('   • Adaptation time constant (τ): %.1f ms\n', tau_opt*1000);
fprintf('⚡ Performance: %.1fx speedup with simulateFast\n', speedup);
fprintf('🔬 Model quality: VP distance = %.3f\n', final_vp);
fprintf('💾 Results saved to: %s\n', save_filename);
% Copy to clipboard
clipboard('copy', summary_log);
fprintf('📋 Summary copied to clipboard!\n');
fprintf('🔄 Paste this summary to continue analysis discussions.\n');


%% === COMPACT SUMMARY LOG ===
fprintf('\n' + string(repmat('=', 1, 50)) + '\n');
fprintf('SPIKE ANALYSIS SUMMARY LOG\n');
fprintf(string(repmat('=', 1, 50)) + '\n');

% Create compact summary for easy copying
summary_log = sprintf([...
    'CELL: %s (%s) | DUR: %.1fs | RATE: %.1fHz\n' ...
    'SPIKES: %d detected | ISI: %.1f±%.1fms\n' ...
    'DETECTION: vm_thresh=%.1f, d2v_thresh=%.1f, search_back=%.1fms\n' ...
    'THRESHOLD_STATS: μ=%.1f±%.1fmV, range=[%.1f,%.1f]mV\n' ...
    'OPTIMIZED_MODEL: θ₀=%.1f, A=%.1f, τ=%.1fms | VP=%.3f\n' ...
    'PERFORMANCE: %.1fx speedup | Rate_acc=%.1f%% | Pred_spikes=%d\n' ...
    'SAVED: %s'], ...
    cell_name, cell_type_name, t_total, true_rate, ...
    n_true_spikes, mean(isi), std(isi), ...
    vm_thresh, d2v_thresh, search_back_ms, ...
    mean(threshold_values), std(threshold_values), min(threshold_values), max(threshold_values), ...
    theta0_opt, A_opt, tau_opt*1000, final_vp, ...
    speedup, 100 * min(pred_rate, true_rate) / max(pred_rate, true_rate), n_pred_spikes, ...
    save_filename);

fprintf('%s\n', summary_log);
fprintf(string(repmat('=', 1, 50)) + '\n');

% Copy to clipboard
clipboard('copy', summary_log);
fprintf('📋 Summary copied to clipboard!\n');
fprintf('🔄 Paste this summary to continue analysis discussions.\n');




%% === FAST LOSS FUNCTION FOR OPTIMIZATION ===
function loss = compute_fast_vp_loss(params, model, q)
    % Fast Victor-Purpura loss using simulateFast
    % This replaces the built-in vp_loss_exponential to use simulateFast
    
    theta0 = params(1);
    A = params(2);
    tau = params(3);
    
    % Define exponential kernel
    kernel_fn = @(t) A * exp(-t / tau);
    
    % Use simulateFast for speed
    try
        [~, ~, ~, spike_times] = model.simulateFast(theta0, kernel_fn, 'profile', false);
        
        % Get true spike times
        true_spike_times = model.elbow_indices * model.dt;
        
        % Calculate Victor-Purpura distance
        loss = spkd_c(spike_times, true_spike_times, ...
            length(spike_times), length(true_spike_times), q);
        
        % Optional: Print progress (comment out for silent optimization)
        fprintf('θ₀=%.1f, A=%.1f, τ=%.1f → VP=%.3f | spikes=%d vs %d\n', ...
            theta0, A, tau*1000, loss, length(spike_times), length(true_spike_times));
            
    catch ME
        % Penalize failed simulations heavily
        loss = 1000;
        fprintf('Simulation failed: θ₀=%.1f, A=%.1f, τ=%.1f → VP=1000 (penalty)\n', ...
            theta0, A, tau*1000);
    end
end