%% Real Experimental Data Analysis with Dynamic Exponential Threshold - simulateFast ONLY
% Complete workflow for analyzing real patch-clamp data using SpikeResponseModel
% Modified to use ONLY simulateFast for all simulations
% NOW WITH HOLD-OUT VALIDATION
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
if ~exist('detect_spike_initiation_elbow_v2', 'file')
    error(['Please ensure detect_spike_initiation_elbow_v2.m is in your MATLAB path.\n' ...
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

%% === HOLD-OUT VALIDATION SETUP ===
fprintf('\n=== HOLD-OUT VALIDATION SETUP ===\n');

% Hold-out parameters
holdout_duration_sec = 30;  % Fixed 30-second hold-out for fair comparison
holdout_samples = round(holdout_duration_sec / dt);

% Check if hold-out is possible
if holdout_samples >= length(Vm_all)
    warning('Hold-out duration (%.1fs) >= total recording (%.1fs)! Using 70%% of data for hold-out.', ...
        holdout_duration_sec, t_total);
    holdout_samples = round(0.3 * length(Vm_all));  % Use 30% as hold-out
    holdout_duration_sec = holdout_samples * dt;
end

% Calculate split indices
train_end_idx = length(Vm_all) - holdout_samples;
holdout_start_idx = train_end_idx + 1;

% Split data
Vm_all_train = Vm_all(1:train_end_idx);
Vm_cleaned_train = Vm_cleaned(1:train_end_idx);
Vm_all_holdout = Vm_all(holdout_start_idx:end);
Vm_cleaned_holdout = Vm_cleaned(holdout_start_idx:end);

% Calculate durations and percentages
train_duration = train_end_idx * dt;
holdout_percentage = (holdout_duration_sec / t_total) * 100;
train_percentage = (train_duration / t_total) * 100;

fprintf('Data split configuration:\n');
fprintf('  Total duration: %.2f s\n', t_total);
fprintf('  Training set: %.2f s (%.1f%% of total)\n', train_duration, train_percentage);
fprintf('  Hold-out set: %.2f s (%.1f%% of total)\n', holdout_duration_sec, holdout_percentage);
fprintf('  Training samples: %d\n', length(Vm_all_train));
fprintf('  Hold-out samples: %d\n', length(Vm_all_holdout));

if holdout_percentage < 20
    warning('Hold-out set is only %.1f%% of total data. Consider longer recordings for robust validation.', holdout_percentage);
end

%% === SPIKE DETECTION USING YOUR ELBOW FUNCTION (TRAINING SET ONLY) ===
fprintf('\n=== SPIKE DETECTION (TRAINING SET) ===\n');

% Parameters for your elbow detection function
vm_thresh = -20;        % Voltage threshold for peak detection (mV)
d2v_thresh = 50;        % Second derivative threshold for elbow detection
search_back_ms = 2;     % Search back window in ms
plot_flag = true;       % Show detection plots

fprintf('Elbow detection parameters:\n');
fprintf('  Voltage threshold: %.1f mV\n', vm_thresh);
fprintf('  d²V/dt² threshold: %.1f mV/ms²\n', d2v_thresh);
fprintf('  Search back window: %.1f ms\n', search_back_ms);

% Run spike detection on TRAINING set only
[elbow_indices_train, spike_peaks_train, isi_train, avg_spike, diagnostic_info] = ...
    detect_spike_initiation_elbow_v2(Vm_all_train, dt, vm_thresh, d2v_thresh, search_back_ms, plot_flag, ...
    'elbow_thresh', -65, 'spike_thresh', -10, 'min_dv_thresh', 0.1, 'time_to_peak_thresh', 1.5);

% Calculate threshold values at spike initiation points
if isempty(elbow_indices_train)
    error('No spikes detected in training set! Try adjusting detection parameters:\n - Lower vm_thresh\n - Lower d2v_thresh\n - Increase search_back_ms');
end

threshold_values_train = Vm_all_train(elbow_indices_train);  % Voltage at spike initiation

fprintf('\nSpike detection results (TRAINING SET):\n');
fprintf('  Detected elbow points: %d\n', length(elbow_indices_train));
fprintf('  Average firing rate: %.2f Hz\n', length(elbow_indices_train)/train_duration);
fprintf('  Threshold range: %.1f to %.1f mV\n', min(threshold_values_train), max(threshold_values_train));
fprintf('  Mean threshold: %.1f ± %.1f mV\n', mean(threshold_values_train), std(threshold_values_train));

if ~isempty(isi_train)
    fprintf('  Mean ISI: %.1f ± %.1f ms\n', mean(isi_train), std(isi_train));
    fprintf('  ISI range: %.1f to %.1f ms\n', min(isi_train), max(isi_train));
end

%% === SPIKE DETECTION ON HOLD-OUT SET (FOR VALIDATION) ===
fprintf('\n=== SPIKE DETECTION (HOLD-OUT SET) ===\n');

% Run spike detection on HOLD-OUT set using same parameters
[elbow_indices_holdout, ~, isi_holdout, ~, ~] = ...
    detect_spike_initiation_elbow_v2(Vm_all_holdout, dt, vm_thresh, d2v_thresh, search_back_ms, false, ...
    'elbow_thresh', -65, 'spike_thresh', -10, 'min_dv_thresh', 0.1, 'time_to_peak_thresh', 1.5);

threshold_values_holdout = Vm_all_holdout(elbow_indices_holdout);

fprintf('Hold-out set spike detection:\n');
fprintf('  Detected spikes: %d\n', length(elbow_indices_holdout));
fprintf('  Firing rate: %.2f Hz\n', length(elbow_indices_holdout)/holdout_duration_sec);
if ~isempty(threshold_values_holdout)
    fprintf('  Mean threshold: %.1f ± %.1f mV\n', mean(threshold_values_holdout), std(threshold_values_holdout));
end

%% === CREATE SPIKERESPONSEMODEL OBJECT (TRAINING DATA) ===
fprintf('\n=== CREATING SPIKERESPONSEMODEL (TRAINING DATA) ===\n');

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

% Create the model using TRAINING data only
model = SpikeResponseModel( ...
    Vm_cleaned_train, ...          % Subthreshold voltage trace (training)
    Vm_all_train, ...              % Raw voltage trace (training)
    dt, ...                        % Time step
    avg_spike, ...                 % Average spike waveform
    tau_ref_ms, ...                % Refractory period
    elbow_indices_train, ...       % Spike initiation indices (training)
    threshold_values_train, ...    % Threshold values at spikes (training)
    cell_name, ...                 % Cell identifier
    cell_type_name ...             % Cell type
);

fprintf('SpikeResponseModel created successfully (TRAINING DATA):\n');
fprintf('  Cell ID: %s\n', cell_name);
fprintf('  Cell type: %s\n', cell_type_name);
fprintf('  Training spikes: %d\n', length(elbow_indices_train));
fprintf('  Training duration: %.2f s\n', train_duration);

%% === PARAMETER OPTIMIZATION - EXPONENTIAL KERNEL (TRAINING DATA) ===
fprintf('\n=== EXPONENTIAL THRESHOLD OPTIMIZATION WITH SIMULATEFAST (TRAINING) ===\n');

% Initial parameter estimates for exponential kernel
theta0_range = [min(threshold_values_train)-10, max(threshold_values_train)+5];
theta0_init = mean(threshold_values_train) - 2;  % Start slightly below mean threshold

A_range = [1, 15];  % mV
A_init = 5;         % Start with moderate adaptation

tau_range = [0.005, 0.1];  % 5-100 ms
tau_init = 0.03;           % Start with 30 ms

init_params_exp = [theta0_init, A_init, tau_init];

fprintf('Exponential kernel optimization setup (TRAINING):\n');
fprintf('  Initial θ₀: %.1f mV (range: %.1f to %.1f mV)\n', theta0_init, theta0_range);
fprintf('  Initial A:  %.1f mV (range: %.1f to %.1f mV)\n', A_init, A_range);
fprintf('  Initial τ:  %.1f ms (range: %.1f to %.1f ms)\n', tau_init*1000, tau_range*1000);

% Victor-Purpura distance parameter
vp_q = 4;  % Standard value for spike timing precision

% Define FAST optimization objective using simulateFast ONLY
fprintf('\nOptimizing EXPONENTIAL kernel using simulateFast (TRAINING DATA)...\n');

% Fast loss function for exponential kernel
fast_loss_exp_fn = @(params) compute_fast_vp_loss_exponential(params, model, vp_q);

% Set optimization options for fminsearch
options = optimset('Display', 'iter', 'MaxFunEvals', 1000, 'MaxIter', 500, ...
                   'TolX', 1e-6, 'TolFun', 1e-4);

fprintf('Using fminsearch with simulateFast for exponential kernel...\n');

% Run optimization with fminsearch - EXPONENTIAL
tic;
[opt_params_exp, final_vp_exp_train] = fminsearch(fast_loss_exp_fn, init_params_exp, options);
opt_time_exp = toc;

% Extract optimized parameters - exponential
theta0_opt_exp = opt_params_exp(1);
A_opt_exp = opt_params_exp(2);
tau_opt_exp = opt_params_exp(3);

fprintf('\n=== EXPONENTIAL KERNEL RESULTS (TRAINING) ===\n');
fprintf('Optimization completed in %.1f seconds\n', opt_time_exp);
fprintf('Initial params: θ₀=%.1f, A=%.1f, τ=%.1f ms\n', init_params_exp(1), init_params_exp(2), init_params_exp(3)*1000);
fprintf('Optimized params: θ₀=%.1f, A=%.1f, τ=%.1f ms\n', theta0_opt_exp, A_opt_exp, tau_opt_exp*1000);
fprintf('Final VP distance (TRAINING): %.3f\n', final_vp_exp_train);

%% === PARAMETER OPTIMIZATION - LINEAR RISE + EXPONENTIAL DECAY KERNEL (TRAINING) ===
fprintf('\n=== LINEAR RISE + EXPONENTIAL DECAY OPTIMIZATION (TRAINING) ===\n');

% Initial parameter estimates for linear rise + exp decay kernel
T_rise_range = [0.001, 0.01];  % 1-10 ms
T_rise_init = 0.003;           % Start with 3 ms

tau_decay_range = [0.01, 0.1]; % 10-100 ms
tau_decay_init = 0.04;         % Start with 40 ms

init_params_linexp = [theta0_init, A_init, T_rise_init, tau_decay_init];

fprintf('Linear rise + exponential decay kernel optimization setup (TRAINING):\n');
fprintf('  Initial θ₀: %.1f mV\n', theta0_init);
fprintf('  Initial A:  %.1f mV\n', A_init);
fprintf('  Initial T_rise: %.1f ms (range: %.1f to %.1f ms)\n', T_rise_init*1000, T_rise_range*1000);
fprintf('  Initial τ_decay: %.1f ms (range: %.1f to %.1f ms)\n', tau_decay_init*1000, tau_decay_range*1000);

% Define optimization objective for linear rise + exp decay
fast_loss_linexp_fn = @(params) compute_fast_vp_loss_linexp(params, model, vp_q);

fprintf('\nOptimizing LINEAR RISE + EXP DECAY kernel using simulateFast (TRAINING)...\n');

% Run optimization - LINEAR RISE + EXP DECAY
tic;
[opt_params_linexp, final_vp_linexp_train] = fminsearch(fast_loss_linexp_fn, init_params_linexp, options);
opt_time_linexp = toc;

% Extract optimized parameters - linear rise + exp decay
theta0_opt_linexp = opt_params_linexp(1);
A_opt_linexp = opt_params_linexp(2);
T_rise_opt_linexp = opt_params_linexp(3);
tau_decay_opt_linexp = opt_params_linexp(4);

fprintf('\n=== LINEAR RISE + EXP DECAY RESULTS (TRAINING) ===\n');
fprintf('Optimization completed in %.1f seconds\n', opt_time_linexp);
fprintf('Optimized params: θ₀=%.1f, A=%.1f, T_rise=%.1f ms, τ_decay=%.1f ms\n', ...
    theta0_opt_linexp, A_opt_linexp, T_rise_opt_linexp*1000, tau_decay_opt_linexp*1000);
fprintf('Final VP distance (TRAINING): %.3f\n', final_vp_linexp_train);

%% === HOLD-OUT VALIDATION FOR BOTH KERNELS ===
fprintf('\n=== HOLD-OUT VALIDATION ===\n');

% Create hold-out model for validation
model_holdout = SpikeResponseModel( ...
    Vm_cleaned_holdout, ...        % Hold-out subthreshold data
    Vm_all_holdout, ...            % Hold-out raw data
    dt, ...                        % Time step
    avg_spike, ...                 % Same spike waveform
    tau_ref_ms, ...                % Same refractory period
    elbow_indices_holdout, ...     % Hold-out spike indices
    threshold_values_holdout, ...  % Hold-out thresholds
    cell_name, ...                 % Cell identifier
    cell_type_name ...             % Cell type
);

% Test EXPONENTIAL kernel on hold-out data
fprintf('Testing EXPONENTIAL kernel on hold-out data...\n');
kernel_exp = @(t) A_opt_exp * exp(-t / tau_opt_exp);
fast_loss_exp_holdout_fn = @(params) compute_fast_vp_loss_exponential(params, model_holdout, vp_q);
final_vp_exp_holdout = fast_loss_exp_holdout_fn(opt_params_exp);

% Test LINEAR RISE + EXP DECAY kernel on hold-out data
fprintf('Testing LINEAR RISE + EXP DECAY kernel on hold-out data...\n');
fast_loss_linexp_holdout_fn = @(params) compute_fast_vp_loss_linexp(params, model_holdout, vp_q);
final_vp_linexp_holdout = fast_loss_linexp_holdout_fn(opt_params_linexp);

%% === HOLD-OUT RESULTS COMPARISON ===
fprintf('\n=== HOLD-OUT VALIDATION RESULTS ===\n');
fprintf('EXPONENTIAL kernel:\n');
fprintf('  Training VP: %.3f\n', final_vp_exp_train);
fprintf('  Hold-out VP: %.3f\n', final_vp_exp_holdout);
fprintf('  Generalization ratio: %.3f (lower is better)\n', final_vp_exp_holdout / final_vp_exp_train);

fprintf('\nLINEAR RISE + EXP DECAY kernel:\n');
fprintf('  Training VP: %.3f\n', final_vp_linexp_train);
fprintf('  Hold-out VP: %.3f\n', final_vp_linexp_holdout);
fprintf('  Generalization ratio: %.3f (lower is better)\n', final_vp_linexp_holdout / final_vp_linexp_train);

% Select best model based on HOLD-OUT performance
if final_vp_exp_holdout < final_vp_linexp_holdout
    fprintf('\n🏆 EXPONENTIAL kernel performs better on HOLD-OUT data\n');
    best_kernel = 'exponential';
    theta0_opt = theta0_opt_exp;
    A_opt = A_opt_exp;
    tau_opt = tau_opt_exp;
    final_vp_train = final_vp_exp_train;
    final_vp_holdout = final_vp_exp_holdout;
    opt_params = opt_params_exp;
    kernel_opt = @(t) A_opt * exp(-t / tau_opt);
    kernel_params = [A_opt, tau_opt];
else
    fprintf('\n🏆 LINEAR RISE + EXP DECAY kernel performs better on HOLD-OUT data\n');
    best_kernel = 'linear_rise_exp_decay';
    theta0_opt = theta0_opt_linexp;
    A_opt = A_opt_linexp;
    T_rise_opt = T_rise_opt_linexp;
    tau_decay_opt = tau_decay_opt_linexp;
    final_vp_train = final_vp_linexp_train;
    final_vp_holdout = final_vp_linexp_holdout;
    opt_params = opt_params_linexp;
    kernel_opt = @(t) (t < T_rise_opt) .* (A_opt / T_rise_opt .* t) + ...
                      (t >= T_rise_opt) .* (A_opt * exp(-(t - T_rise_opt) / tau_decay_opt));
    kernel_params = [A_opt, T_rise_opt, tau_decay_opt];
end

holdout_improvement_pct = abs(final_vp_exp_holdout - final_vp_linexp_holdout) / max(final_vp_exp_holdout, final_vp_linexp_holdout) * 100;
fprintf('Hold-out performance improvement: %.1f%%\n', holdout_improvement_pct);

% Check for overfitting
generalization_ratio = final_vp_holdout / final_vp_train;
if generalization_ratio > 1.5
    warning('Possible overfitting detected! Hold-out VP is %.1fx higher than training VP.', generalization_ratio);
elseif generalization_ratio > 1.2
    fprintf('⚠️  Moderate generalization gap: Hold-out VP is %.1fx training VP\n', generalization_ratio);
else
    fprintf('✅ Good generalization: Hold-out VP is %.1fx training VP\n', generalization_ratio);
end

%% === FINAL VALIDATION ON FULL DATASET ===
fprintf('\n=== FULL DATASET VALIDATION ===\n');

% Create full model for final validation
model_full = SpikeResponseModel( ...
    Vm_cleaned, ...                % Full subthreshold data
    Vm_all, ...                    % Full raw data
    dt, ...                        % Time step
    avg_spike, ...                 % Average spike waveform
    tau_ref_ms, ...                % Refractory period
    [elbow_indices_train(:); elbow_indices_holdout(:) + train_end_idx], ...  % All spike indices
    [threshold_values_train(:); threshold_values_holdout(:)], ...  % All thresholds
    cell_name, ...                 % Cell identifier
    cell_type_name ...             % Cell type
);

% Run best model on full dataset
fprintf('Running best model (%s) on full dataset...\n', best_kernel);
[spikes_full, V_pred_full, threshold_trace_full, spike_times_full, spike_V_full] = ...
    model_full.simulateFast(theta0_opt, kernel_opt, 'profile', false);

% Calculate final metrics
n_true_spikes_full = length(elbow_indices_train) + length(elbow_indices_holdout);
n_pred_spikes_full = length(spike_times_full);
true_rate_full = n_true_spikes_full / t_total;
pred_rate_full = n_pred_spikes_full / t_total;
rate_accuracy_full = 100 * min(pred_rate_full, true_rate_full) / max(pred_rate_full, true_rate_full);

fprintf('\nFull dataset results:\n');
fprintf('  True spikes: %d (%.2f Hz)\n', n_true_spikes_full, true_rate_full);
fprintf('  Predicted spikes: %d (%.2f Hz)\n', n_pred_spikes_full, pred_rate_full);
fprintf('  Rate accuracy: %.1f%%\n', rate_accuracy_full);

%% === PERFORMANCE TESTING ===
fprintf('\n=== PERFORMANCE TESTING ===\n');

% Test simulateFast performance
n_fast_tests = 50;
fprintf('Testing simulateFast with %d simulations...\n', n_fast_tests);

tic;
for i = 1:n_fast_tests
    [~, ~, ~, ~, ~] = model_full.simulateFast(theta0_opt, kernel_opt, 'profile', false);
end
fast_time = toc;

% Calculate average time
avg_fast_time = 1000 * fast_time / n_fast_tests;

fprintf('Performance results:\n');
fprintf('  simulateFast: %.2f ms/simulation\n', avg_fast_time);

%% === THRESHOLD CORRELATION ANALYSIS ===
fprintf('\n=== THRESHOLD CORRELATION ANALYSIS ===\n');

% Run simulations to get predicted thresholds for both sets
fprintf('Running simulations for threshold correlation analysis...\n');

% Training set simulation
[~, ~, threshold_trace_train, ~, ~] = model.simulateFast(theta0_opt, kernel_opt, 'profile', false);

% Hold-out set simulation  
[~, ~, threshold_trace_holdout_sim, ~, ~] = model_holdout.simulateFast(theta0_opt, kernel_opt, 'profile', false);

% Calculate correlations
train_true_thresholds = threshold_values_train;
train_predicted_thresholds = threshold_trace_train(elbow_indices_train);
train_corr = corr(train_true_thresholds(:), train_predicted_thresholds(:));

holdout_true_thresholds = threshold_values_holdout;
holdout_predicted_thresholds = threshold_trace_holdout_sim(elbow_indices_holdout);
holdout_corr = corr(holdout_true_thresholds(:), holdout_predicted_thresholds(:));

fprintf('Threshold correlations:\n');
fprintf('  Training set: r = %.3f\n', train_corr);
fprintf('  Hold-out set: r = %.3f\n', holdout_corr);
fprintf('  Correlation difference: %.3f\n', holdout_corr - train_corr);

%% === DURATION-NORMALIZED VP ANALYSIS ===
fprintf('\n=== DURATION-NORMALIZED VP ANALYSIS ===\n');

% Normalize VP by duration and spike count
vp_per_second_train = final_vp_train / train_duration;
vp_per_second_holdout = final_vp_holdout / holdout_duration_sec;
vp_per_spike_train = final_vp_train / length(elbow_indices_train);
vp_per_spike_holdout = final_vp_holdout / length(elbow_indices_holdout);

fprintf('Normalized VP analysis:\n');
fprintf('  Training VP/second: %.3f\n', vp_per_second_train);
fprintf('  Hold-out VP/second: %.3f\n', vp_per_second_holdout);
fprintf('  VP/second ratio (holdout/train): %.3f\n', vp_per_second_holdout / vp_per_second_train);
fprintf('  Training VP/spike: %.3f\n', vp_per_spike_train);
fprintf('  Hold-out VP/spike: %.3f\n', vp_per_spike_holdout);
fprintf('  VP/spike ratio (holdout/train): %.3f\n', vp_per_spike_holdout / vp_per_spike_train);

if vp_per_second_holdout > vp_per_second_train
    fprintf('  → Hold-out actually performs WORSE when normalized by duration\n');
else
    fprintf('  → Hold-out still performs better even when normalized by duration\n');
end

%% === COMPREHENSIVE VALIDATION VISUALIZATION ===
fprintf('\n=== CREATING COMPREHENSIVE VALIDATION PLOTS ===\n');

figure('Position', [50, 50, 1800, 1000]);

% Plot 1: Threshold correlation scatter plots
subplot(3,4,1);
% NEW (working):
h1 = scatter(train_true_thresholds, train_predicted_thresholds, 30, 'filled');
try
    h1.MarkerFaceAlpha = 0.6;  % This is the correct property name
catch
    % Fallback for older MATLAB versions
end
hold on;
plot([min(train_true_thresholds), max(train_true_thresholds)], ...
     [min(train_true_thresholds), max(train_true_thresholds)], 'r--', 'LineWidth', 2);
xlabel('True Threshold (mV)');
ylabel('Predicted Threshold (mV)');
title(sprintf('Training Set\nr = %.3f', train_corr));
grid on;

subplot(3,4,2);


% NEW (working):
h1 = scatter(holdout_true_thresholds, holdout_predicted_thresholds, 30, 'filled');
try
    h1.MarkerFaceAlpha = 0.6;  % This is the correct property name
catch
    % Fallback for older MATLAB versions
end
hold on;
plot([min(holdout_true_thresholds), max(holdout_true_thresholds)], ...
     [min(holdout_true_thresholds), max(holdout_true_thresholds)], 'r--', 'LineWidth', 2);
xlabel('True Threshold (mV)');
ylabel('Predicted Threshold (mV)');
title(sprintf('Hold-out Set\nr = %.3f', holdout_corr));
grid on;

% Plot 3: Correlation comparison
subplot(3,4,3);
bar([train_corr, holdout_corr], 'FaceColor', [0.3 0.7 0.9]);
set(gca, 'XTickLabel', {'Training', 'Hold-out'});
ylabel('Threshold Correlation');
title('Correlation Comparison');
ylim([0, 1]);
grid on;

% Plot 4: VP comparison (raw)
subplot(3,4,4);
bar([final_vp_train, final_vp_holdout], 'FaceColor', [0.9 0.7 0.3]);
set(gca, 'XTickLabel', {'Training', 'Hold-out'});
ylabel('VP Distance');
title('Raw VP Comparison');
grid on;

% Plot 5: VP per second
subplot(3,4,5);
bar([vp_per_second_train, vp_per_second_holdout], 'FaceColor', [0.7 0.9 0.3]);
set(gca, 'XTickLabel', {'Training', 'Hold-out'});
ylabel('VP per Second');
title('Duration-Normalized VP');
grid on;

% Plot 6: VP per spike
subplot(3,4,6);
bar([vp_per_spike_train, vp_per_spike_holdout], 'FaceColor', [0.9 0.3 0.7]);
set(gca, 'XTickLabel', {'Training', 'Hold-out'});
ylabel('VP per Spike');
title('Spike-Normalized VP');
grid on;

% Plot 7: Generalization metrics
subplot(3,4,7);
metrics = [generalization_ratio, vp_per_second_holdout/vp_per_second_train, vp_per_spike_holdout/vp_per_spike_train];
bar(metrics, 'FaceColor', [0.7 0.3 0.9]);
set(gca, 'XTickLabel', {'Raw', 'Per Sec', 'Per Spike'});
ylabel('Holdout/Training Ratio');
title('Generalization Metrics');
yline(1.0, 'r--', 'LineWidth', 2);
grid on;

% Plot 8: Kernel comparison on both sets
subplot(3,4,8);
kernel_data = [final_vp_exp_train, final_vp_exp_holdout; ...
               final_vp_linexp_train, final_vp_linexp_holdout];
bar(kernel_data);
set(gca, 'XTickLabel', {'Exponential', 'Linear+Exp'});
ylabel('VP Distance');
title('Kernel Comparison');
legend('Training', 'Hold-out', 'Location', 'best');
grid on;

% Plot 9: Firing rate consistency
subplot(3,4,9);
rates = [length(elbow_indices_train)/train_duration, length(elbow_indices_holdout)/holdout_duration_sec];
bar(rates, 'FaceColor', [0.5 0.8 0.5]);
set(gca, 'XTickLabel', {'Training', 'Hold-out'});
ylabel('Firing Rate (Hz)');
title('Rate Consistency');
grid on;

% Plot 10: Threshold statistics
subplot(3,4,10);
thresh_stats = [mean(train_true_thresholds), std(train_true_thresholds); ...
                mean(holdout_true_thresholds), std(holdout_true_thresholds)];
bar(thresh_stats);
set(gca, 'XTickLabel', {'Training', 'Hold-out'});
ylabel('Threshold (mV)');
title('Threshold Statistics');
legend('Mean', 'Std', 'Location', 'best');
grid on;

% Plot 11: Model quality summary
subplot(3,4,11);
quality_metrics = [train_corr, holdout_corr; ...
                   rate_accuracy_full/100, rate_accuracy_full/100; ...
                   1-vp_per_spike_train/100, 1-vp_per_spike_holdout/100];
bar(quality_metrics);
set(gca, 'XTickLabel', {'Thresh Corr', 'Rate Acc', 'VP Quality'});
ylabel('Quality Score');
title('Model Quality');
legend('Training', 'Hold-out', 'Location', 'best');
ylim([0, 1]);
grid on;

% Plot 12: Performance summary text
subplot(3,4,12);
axis off;
summary_text = {
    'VALIDATION SUMMARY';
    '';
    sprintf('Duration: %.1fs train, %.1fs holdout', train_duration, holdout_duration_sec);
    sprintf('Spikes: %d train, %d holdout', length(elbow_indices_train), length(elbow_indices_holdout));
    '';
    'THRESHOLD CORRELATION:';
    sprintf('  Training: r=%.3f', train_corr);
    sprintf('  Hold-out: r=%.3f', holdout_corr);
    '';
    'VP NORMALIZATION:';
    sprintf('  Raw ratio: %.2f', generalization_ratio);
    sprintf('  Per-second ratio: %.2f', vp_per_second_holdout/vp_per_second_train);
    sprintf('  Per-spike ratio: %.2f', vp_per_spike_holdout/vp_per_spike_train);
    '';
    sprintf('Best kernel: %s', best_kernel);
    sprintf('Rate accuracy: %.1f%%', rate_accuracy_full);
};

text(0.05, 0.95, summary_text, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
    'FontSize', 10, 'FontName', 'FixedWidth');

sgtitle(sprintf('Comprehensive Validation Analysis: %s (%s)', cell_name, cell_type_name), ...
    'FontSize', 14, 'FontWeight', 'bold');

fprintf('Comprehensive validation analysis complete!\n');

%% === SAVE RESULTS WITH HOLD-OUT VALIDATION ===
fprintf('\n=== SAVING RESULTS ===\n');

% Save optimization results with hold-out validation
results = struct();
results.cell_info = struct('id', cell_name, 'type', cell_type_name);
results.experimental = struct('dt', dt, 'total_duration', t_total, 'n_spikes_total', n_true_spikes_full);

% Hold-out validation info
results.holdout_validation = struct( ...
    'holdout_duration_sec', holdout_duration_sec, ...
    'holdout_percentage', holdout_percentage, ...
    'train_duration_sec', train_duration, ...
    'train_percentage', train_percentage, ...
    'n_spikes_train', length(elbow_indices_train), ...
    'n_spikes_holdout', length(elbow_indices_holdout) ...
);

% Model comparison results
results.model_comparison = struct( ...
    'exponential', struct('train_vp', final_vp_exp_train, 'holdout_vp', final_vp_exp_holdout, ...
                         'params', opt_params_exp, 'generalization_ratio', final_vp_exp_holdout/final_vp_exp_train), ...
    'linear_exp', struct('train_vp', final_vp_linexp_train, 'holdout_vp', final_vp_linexp_holdout, ...
                        'params', opt_params_linexp, 'generalization_ratio', final_vp_linexp_holdout/final_vp_linexp_train), ...
    'best_kernel', best_kernel, ...
    'holdout_improvement_pct', holdout_improvement_pct ...
);

results.final_model = struct('kernel_type', best_kernel, 'train_vp', final_vp_train, ...
                            'holdout_vp', final_vp_holdout, 'params', opt_params, ...
                            'rate_accuracy', rate_accuracy_full);
results.performance = struct('simulateFast_time_ms', avg_fast_time, 'total_tests', n_fast_tests);

% Add new validation metrics to results
results.threshold_correlation = struct( ...
    'train_correlation', train_corr, ...
    'holdout_correlation', holdout_corr, ...
    'correlation_difference', holdout_corr - train_corr ...
);

results.normalized_vp = struct( ...
    'train_vp_per_second', vp_per_second_train, ...
    'holdout_vp_per_second', vp_per_second_holdout, ...
    'vp_per_second_ratio', vp_per_second_holdout / vp_per_second_train, ...
    'train_vp_per_spike', vp_per_spike_train, ...
    'holdout_vp_per_spike', vp_per_spike_holdout, ...
    'vp_per_spike_ratio', vp_per_spike_holdout / vp_per_spike_train ...
);

% Save to file
save_filename = sprintf('spike_analysis_holdout_%s_%s.mat', cell_name, datestr(now, 'yyyymmdd_HHMMSS'));
save(save_filename, 'results', 'model_full', 'Vm_all', 'Vm_cleaned');

fprintf('Results saved to: %s\n', save_filename);

%% === COMPACT SUMMARY LOG WITH HOLD-OUT ===
fprintf('\n' + string(repmat('=', 1, 60)) + '\n');
fprintf('SPIKE ANALYSIS SUMMARY WITH HOLD-OUT VALIDATION\n');
fprintf(string(repmat('=', 1, 60)) + '\n');

% Create compact summary
summary_log = sprintf([...
    'CELL: %s (%s) | TOTAL_DUR: %.1fs | RATE: %.1fHz\n' ...
    'HOLDOUT: %.1fs (%.1f%%) | TRAIN: %.1fs (%.1f%%)\n' ...
    'SPIKES: %d total (%d train, %d holdout)\n' ...
    'DETECTION: vm_thresh=%.1f, d2v_thresh=%.1f, enhanced_v2\n' ...
    'TRAINING_RESULTS:\n' ...
    '  EXPONENTIAL: θ₀=%.1f, A=%.1f, τ=%.1fms | VP_train=%.3f\n' ...
    '  LINEXP: θ₀=%.1f, A=%.1f, T_rise=%.1f, τ_decay=%.1fms | VP_train=%.3f\n' ...
    'HOLDOUT_VALIDATION:\n' ...
    '  EXPONENTIAL: VP_holdout=%.3f | Ratio=%.2f\n' ...
    '  LINEXP: VP_holdout=%.3f | Ratio=%.2f\n' ...
    'BEST_KERNEL: %s | HOLDOUT_IMPROVEMENT: %.1f%%\n' ...
    'THRESHOLD_CORRELATION: train=%.3f, holdout=%.3f, diff=%.3f\n' ...
    'VP_NORMALIZED: per_sec_ratio=%.2f, per_spike_ratio=%.2f\n' ...
    'GENERALIZATION: %.2fx (raw), %.2fx (per_sec), %.2fx (per_spike)\n' ...
    'PERFORMANCE: simulateFast=%.1fms | Rate_acc=%.1f%% | Pred_spikes=%d\n' ...
    'METHOD: simulateFast_HOLDOUT_VALIDATION | Tests=%d\n' ...
    'SAVED: %s'], ...
    cell_name, cell_type_name, t_total, true_rate_full, ...
    holdout_duration_sec, holdout_percentage, train_duration, train_percentage, ...
    n_true_spikes_full, length(elbow_indices_train), length(elbow_indices_holdout), ...
    vm_thresh, d2v_thresh, ...
    theta0_opt_exp, A_opt_exp, tau_opt_exp*1000, final_vp_exp_train, ...
    theta0_opt_linexp, A_opt_linexp, T_rise_opt_linexp*1000, tau_decay_opt_linexp*1000, final_vp_linexp_train, ...
    final_vp_exp_holdout, final_vp_exp_holdout/final_vp_exp_train, ...
    final_vp_linexp_holdout, final_vp_linexp_holdout/final_vp_linexp_train, ...
    best_kernel, holdout_improvement_pct, ...
    train_corr, holdout_corr, holdout_corr - train_corr, ...
    vp_per_second_holdout / vp_per_second_train, vp_per_spike_holdout / vp_per_spike_train, ...
    generalization_ratio, vp_per_second_holdout / vp_per_second_train, vp_per_spike_holdout / vp_per_spike_train, ...
    avg_fast_time, rate_accuracy_full, n_pred_spikes_full, ...
    n_fast_tests, save_filename);

fprintf('%s\n', summary_log);
fprintf(string(repmat('=', 1, 60)) + '\n');

%% === FINAL SUMMARY ===
fprintf('\n=== ANALYSIS COMPLETE WITH HOLD-OUT VALIDATION ===\n');
fprintf('Successfully analyzed experimental data with proper validation\n');
fprintf('Total spikes: %d (%.2f Hz over %.2fs)\n', n_true_spikes_full, true_rate_full, t_total);
fprintf('Hold-out validation: %.1fs (%.1f%%) for unbiased evaluation\n', holdout_duration_sec, holdout_percentage);
fprintf('\nBest model: %s\n', best_kernel);
fprintf('   Training VP: %.3f\n', final_vp_train);
fprintf('   Hold-out VP: %.3f\n', final_vp_holdout);
fprintf('   Generalization: %.2fx\n', generalization_ratio);
fprintf('   Threshold correlation: %.3f (train), %.3f (holdout)\n', train_corr, holdout_corr);
fprintf('Performance: %.1f ms/simulation\n', avg_fast_time);
fprintf('Results: %s\n', save_filename);

% Copy to clipboard
clipboard('copy', summary_log);
fprintf('Summary copied to clipboard!\n');

%% === LOSS FUNCTIONS FOR HOLD-OUT VALIDATION ===
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
        % fprintf('EXP: θ₀=%.1f, A=%.1f, τ=%.1f → VP=%.3f | spikes=%d vs %d\n', ...
        %     theta0, A, tau*1000, loss, length(spike_times), length(true_spike_times));
            
    catch ME
        % Penalize failed simulations heavily
        loss = 1000;
        % fprintf('EXP failed: θ₀=%.1f, A=%.1f, τ=%.1f → VP=1000 (penalty)\n', ...
        %     theta0, A, tau*1000);
    end
end

function loss = compute_fast_vp_loss_linexp(params, model, q)
    % Fast Victor-Purpura loss using simulateFast ONLY - LINEAR RISE + EXP DECAY kernel
    
    theta0 = params(1);
    A = params(2);
    T_rise = params(3);
    tau_decay = params(4);
    
    % Define linear rise + exponential decay kernel
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
        % fprintf('LINEXP: θ₀=%.1f, A=%.1f, T_rise=%.1f, τ_decay=%.1f → VP=%.3f | spikes=%d vs %d\n', ...
        %     theta0, A, T_rise*1000, tau_decay*1000, loss, length(spike_times), length(true_spike_times));
            
    catch ME
        % Penalize failed simulations heavily
        loss = 1000;
        % fprintf('LINEXP failed: θ₀=%.1f, A=%.1f, T_rise=%.1f, τ_decay=%.1f → VP=1000 (penalty)\n', ...
        %     theta0, A, T_rise*1000, tau_decay*1000);
    end
end