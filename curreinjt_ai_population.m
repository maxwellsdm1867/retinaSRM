%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SODIUM CHANNEL DYNAMICS - DATA EXTRACTION AND ANALYSIS (NO PLOTTING)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%*************************************************************************
% Initialization
%*************************************************************************
close all; clear; clc;

% jauimodel setup
loader = edu.washington.rieke.Analysis.getEntityLoader();
treeFactory = edu.washington.rieke.Analysis.getEpochTreeFactory();

% Data paths (adjust as needed)
dataFolder = '/Users/maxwellsdm/Library/CloudStorage/GoogleDrive-maxwellsdm1867@gmail.com/.shortcut-targets-by-id/1CbZnfugdi-p4jM3fe7t-NyGk5v2ZcPdk/ParasolCenterSurround/Data/';
exportFolder = dataFolder;

import auimodel.*
import vuidocument.*

% Analysis parameters
params = struct();
params.Amp = 'Amp1';
params.SamplingInterval = 0.0001;  % 10 kHz sampling
params.Verbose = 1;

% Log storage configuration
log_config = struct();
log_config.save_path = '/Users/maxwellsdm/Library/CloudStorage/GoogleDrive-maxwellsdm1867@gmail.com/.shortcut-targets-by-id/1CbZnfugdi-p4jM3fe7t-NyGk5v2ZcPdk/ParasolCenterSurround/Analysis/SodiumDynamics/';
log_config.create_plots = false;   % Disable plotting
log_config.verbose_logging = true; % Detailed logging

% Create log directory if it doesn't exist
if ~exist(log_config.save_path, 'dir')
    mkdir(log_config.save_path);
    fprintf('Created log directory: %s\n', log_config.save_path);
end

% Spike detection parameters
spike_params = struct();
spike_params.vm_thresh = -20;       % Voltage threshold (mV)
spike_params.d2v_thresh = 50;       % Second derivative threshold
spike_params.search_back_ms = 2;    % Search back window (ms)
spike_params.plot_flag = false;     % Suppress plots during batch processing

% Voltage preprocessing parameters
filter_params = struct();
filter_params.cutoff_freq_Hz = 90;  % Low-pass cutoff for cleaned voltage

fprintf('=== SODIUM CHANNEL DYNAMICS ANALYSIS INITIALIZED ===\n');

% Load epoch list (modify filename as needed)
list = loader.loadEpochList([exportFolder 'currinjt250503.mat'], dataFolder);

% Build tree structure
dateSplit = @(list)splitOnExperimentDate(list);
dateSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(list, dateSplit);

tree = riekesuite.analysis.buildTree(list, {
    'protocolSettings(source:type)',        % Level 1: Cell Type ← This is what we need!
    dateSplit_java,                         % Level 2: Date
    'cell.label',                          % Level 3: Cell (individual cell)
    'protocolSettings(epochGroup:label)',   % Level 4: Group
    'protocolSettings(frequencyCutoff)',    % Level 5: Frequency
    'protocolSettings(currentSD)'           % Level 6: Current SD
    });
% Launch GUI for selection
gui = epochTreeGUI(tree);

fprintf('Please select the top-level node in the GUI, then run the next section\n');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SECTION 1: TREE NAVIGATION, SELECTION & ELEMENTARY ANALYSIS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n=== STARTING TREE NAVIGATION, SELECTION & ANALYSIS ===\n');


% Get the selected node from GUI (should be top-level node)
selectedNodes = getSelectedEpochTreeNodes(gui);
if length(selectedNodes) ~= 1
    error('Please select exactly one top-level node in the GUI');
end

CurrentNode = selectedNodes{1};  % Cell array access
fprintf('Selected node: %s\n', CurrentNode.splitValue);
fprintf('Number of children: %d\n', CurrentNode.children.length);

% Initialize global logging
global analysis_log log_counter;
analysis_log = {};
log_counter = 1;

% Master log entry
master_log = struct();
master_log.analysis_type = 'SodiumChannelDynamics';
master_log.timestamp = datetime('now');
master_log.purpose = 'Elementary analysis on highest epoch count SD levels';
master_log.selection_strategy = 'Most epochs within frequency groups';
analysis_log{log_counter} = master_log;
log_counter = log_counter + 1;

% Storage for analysis results
analysis_results = {};
failed_analyses = {};
results_counter = 1;

% Navigate tree structure: Root → Protocol → Date → Cell → Frequency → CurrentSD
fprintf('\n--- Navigating Tree Structure and Performing Analysis ---\n');

% Level 1: Protocol (source:type)
for protocol_idx = 1:CurrentNode.children.length
    protocolNode = CurrentNode.children.elements(protocol_idx);
    protocol_name = protocolNode.splitValue;
    fprintf('\nProtocol: %s\n', protocol_name);

    % Level 2: Date
    for date_idx = 1:protocolNode.children.length
        dateNode = protocolNode.children.elements(date_idx);
        date_name = dateNode.splitValue;
        fprintf('  Date: %s\n', date_name);

        % Level 3: Cell
        for cell_idx = 1:dateNode.children.length
            cellNode = dateNode.children.elements(cell_idx);
            cell_name = cellNode.splitValue;
            fprintf('    Cell: %s\n', cell_name);

            % Level 4: Epoch Group (protocolSettings(epochGroup:label))
            for group_idx = 1:cellNode.children.length
                groupNode = cellNode.children.elements(group_idx);
                group_name = groupNode.splitValue;
                fprintf('      Group: %s\n', group_name);

                % Level 5: Frequency Cutoff
                for freq_idx = 1:groupNode.children.length
                    freqNode = groupNode.children.elements(freq_idx);
                    freq_value = freqNode.splitValue;
                    fprintf('        Frequency: %s\n', freq_value);

                    % Level 6: Current SD - Find the one with most epochs
                    if freqNode.children.length > 0
                        fprintf('          Current SD levels found: %d\n', freqNode.children.length);

                        % Count epochs for each SD level
                        sd_epoch_counts = [];
                        sd_values = {};
                        sd_nodes = {};

                        for sd_idx = 1:freqNode.children.length
                            sdNode = freqNode.children.elements(sd_idx);
                            sd_value = sdNode.splitValue;
                            n_epochs = sdNode.epochList.length;

                            sd_epoch_counts(end+1) = n_epochs;
                            sd_values{end+1} = sd_value;
                            sd_nodes{end+1} = sdNode;

                            fprintf('            SD %s: %d epochs\n', sd_value, n_epochs);
                        end

                        % Find SD level with maximum epochs
                        [max_epochs, max_idx] = max(sd_epoch_counts);

                        if max_epochs > 0
                            selected_sd_node = sd_nodes{max_idx};
                            selected_sd_value = sd_values{max_idx};

                            fprintf('          >>> SELECTED: SD %s with %d epochs <<<\n', ...
                                selected_sd_value, max_epochs);

                            % ===================================================================
                            %  IMMEDIATELY PERFORM ELEMENTARY ANALYSIS ON SELECTED SD NODE
                            % ===================================================================

                            fprintf('          >>> STARTING ELEMENTARY ANALYSIS <<<\n');

                            try
                                % ---------------------------------------------------------------
                                %  DATA EXTRACTION
                                % ---------------------------------------------------------------

                                % Extract electrode data and injected current stimulus
                                EpochData = getSelectedData(selected_sd_node.epochList, params.Amp);  % Voltage recordings
                                Stimuli = getNoiseStm(selected_sd_node);                              % Injected current traces

                                % Get data dimensions
                                [n_trials, n_timepoints] = size(Stimuli);
                                dt = params.SamplingInterval;

                                fprintf('            Extracted %d trials with %d timepoints each\n', n_trials, n_timepoints);
                                fprintf('            Sampling interval: %.1f μs\n', dt * 1e6);

                                % Concatenate all trials for analysis
                                I_all = reshape(Stimuli', [], 1);     % Injected current [n_trials × n_timepoints, 1]
                                Vm_all = reshape(EpochData', [], 1);  % Membrane voltage [same dimensions]

                                fprintf('            Concatenated data: %d total timepoints\n', length(Vm_all));

                                % ---------------------------------------------------------------
                                %  SPIKE DETECTION USING ELBOW METHOD
                                % ---------------------------------------------------------------

                                % Advanced spike detection parameters
                                vm_thresh = spike_params.vm_thresh;     % -20 mV
                                d2v_thresh = spike_params.d2v_thresh;   % 50
                                search_back_ms = spike_params.search_back_ms; % 2 ms
                                plot_flag = spike_params.plot_flag;     % false for batch processing

                                % Detect spike initiation points using enhanced elbow method v2
                                [elbow_indices, ~, ~, avg_spike_short, diagnostic_info] = detect_spike_initiation_elbow_v2(...
                                    Vm_all, dt, vm_thresh, d2v_thresh, search_back_ms, plot_flag, ...
                                    'elbow_thresh', -65, 'spike_thresh', -10, 'min_dv_thresh', 0.1, ...
                                    'time_to_peak_thresh', 1.5);

                                % Convert to time units
                                true_spike_times_ms = elbow_indices * dt * 1000; % Convert to ms
                                true_spike_times = elbow_indices * dt; % Keep in seconds
                                fprintf('            Enhanced elbow method detected %d spikes\n', length(elbow_indices));

                                % ---------------------------------------------------------------
                                %  VOLTAGE TRACE SMOOTHING AND PREPROCESSING
                                % ---------------------------------------------------------------

                                % Smooth spikes using low-pass filter to extract subthreshold dynamics
                                sampling_rate_Hz = 1/dt;  % Use actual sampling rate from data
                                cutoff_freq_Hz = filter_params.cutoff_freq_Hz;  % 90 Hz
                                Fs = sampling_rate_Hz;    % Sampling frequency

                                % Design low-pass Butterworth filter
                                Wn = cutoff_freq_Hz / (Fs/2);  % Normalized cutoff frequency
                                [b, a] = butter(4, Wn, 'low'); % 4th order Butterworth filter

                                % Apply zero-phase filtering to avoid phase distortion
                                Vm_cleaned = filtfilt(b, a, Vm_all);

                                fprintf('            Low-pass filtering complete. Cutoff: %.1f Hz\n', cutoff_freq_Hz);

                                % ---------------------------------------------------------------
                                %  LINEAR FILTER ESTIMATION
                                % ---------------------------------------------------------------

                                % Preprocess signals for filter estimation
                                I_preprocessed = I_all - mean(I_all);        % Remove DC component from current
                                Vm_preprocessed = Vm_cleaned - mean(Vm_cleaned);  % Remove DC from cleaned voltage

                                % Filter estimation parameters
                                n_trials_filter = 50;                        % Number of trials for filter estimation
                                regularization = 1e-4;                       % Regularization parameter
                                max_lag_ms = 5;                              % Maximum filter lag (ms)

                                % Create designated figure for filter analysis (only once)
                                if results_counter == 1
                                    filter_fig = figure(100);  % Designated figure number for all filter plots
                                    set(filter_fig, 'Position', [100, 100, 1200, 800]);  % Set figure size
                                    fprintf('            Created designated filter figure (Figure 100)\n');
                                end

                                % Estimate linear filter using regularized FFT method with designated figure
                                [filt, lag, Vm_pred, r] = estimate_filter_fft_trials_regularized(...
                                    I_preprocessed, Vm_preprocessed, dt, n_trials_filter, true, 100, regularization, max_lag_ms, filter_fig);

                                % SAVE FIGURE IMMEDIATELY AFTER PLOTTING
                                % Get cell type label using .parent navigation
                                cellTypeNode = selected_sd_node.parent.parent.parent.parent.parent;  % Go back 5 levels to cell type
                                cell_type_raw = cellTypeNode.splitValue;  % Get cell type splitValue
                                cell_type_label = strrep(cell_type_raw, '\', '/');  % Convert backslash to forward slash

                                % Create figure path with cell type subfolder
                                base_path = '/Users/maxwellsdm/Library/CloudStorage/GoogleDrive-maxwellsdm1867@gmail.com/.shortcut-targets-by-id/1CbZnfugdi-p4jM3fe7t-NyGk5v2ZcPdk/ParasolCenterSurround/figures/ScienceJuiceFactory/currinjt/overview/';
                                fig_path = [base_path cell_type_label '/'];

                                % Create directory if it doesn't exist
                                if ~exist(fig_path, 'dir')
                                    mkdir(fig_path);
                                    fprintf('            Created directory: %s\n', fig_path);
                                end

                                % Create filename: date(1:11)-cellname-freq-##-subthreshold.png
                                date_str = date_name(1:11);  % First 11 characters of date
                                cell_name_lower = lower(cell_name);  % Convert to lowercase (cell3, cell1, etc.)
                                freq_str = num2str(freq_value);  % Convert frequency to string
                                fig_filename = sprintf('%s-%s-freq-%s-subthreshold.png', date_str, cell_name_lower, freq_str);

                                % Save the figure immediately after plotting
                                saveas(filter_fig, [fig_path fig_filename]);
                                fprintf('            Filter figure saved: %s\n', [fig_path fig_filename]);

                                fprintf('            Linear filter estimation complete. Correlation r = %.3f\n', r);

                                % ---------------------------------------------------------------
                                %  SPIKE WAVEFORM EXTRACTION AND PROCESSING
                                % ---------------------------------------------------------------

                                % Process average spike waveform for injection
                                if exist('avg_spike_short', 'var') && ~isempty(avg_spike_short)
                                    avg_spike_corrected = avg_spike_short - avg_spike_short(1);  % Anchor to zero
                                    fprintf('            Spike waveform extracted: %d points (%.1f ms)\n', ...
                                        length(avg_spike_corrected), length(avg_spike_corrected) * dt * 1000);
                                else
                                    fprintf('            Warning: avg_spike_short not available from elbow detection\n');
                                    avg_spike_corrected = [];
                                end

                                % ---------------------------------------------------------------
                                %  SPIKE-TRIGGERED AVERAGE (STA) ANALYSIS
                                % ---------------------------------------------------------------

                                % Use elbow-detected spikes for STA analysis
                                spikes_all = elbow_indices;
                                fprintf('            Using %d elbow-detected spikes for STA analysis\n', length(spikes_all));

                                % Compute Spike-Triggered Average
                                win_before = round(0.1 / dt);  % 100 ms window before spike
                                STA = zeros(win_before, 1);
                                valid_spikes = 0;

                                for k = 1:length(spikes_all)
                                    spike_idx = spikes_all(k);
                                    if spike_idx > win_before
                                        % Extract pre-spike injected current (reversed for causality)
                                        STA = STA + flipud(I_all(spike_idx - win_before + 1 : spike_idx));
                                        valid_spikes = valid_spikes + 1;
                                    end
                                end

                                STA = STA / valid_spikes;
                                fprintf('            STA computed using %d valid spikes\n', valid_spikes);

                                % ---------------------------------------------------------------
                                %  SPIKE-TRIGGERED AFTERPOTENTIAL ANALYSIS
                                % ---------------------------------------------------------------

                                % Compute post-spike membrane potential dynamics (afterpotential)
                                win_after = round(0.1 / dt);  % 100 ms after spike
                                STA_Vm = zeros(win_after, 1);
                                valid_afterspikes = 0;

                                for k = 1:length(spikes_all)
                                    spike_idx = spikes_all(k);
                                    if spike_idx + win_after - 1 <= length(Vm_all)
                                        STA_Vm = STA_Vm + Vm_all(spike_idx : spike_idx + win_after - 1);
                                        valid_afterspikes = valid_afterspikes + 1;
                                    end
                                end

                                STA_Vm = STA_Vm / valid_afterspikes;
                                fprintf('            Afterpotential computed using %d valid spikes\n', valid_afterspikes);

                                % ---------------------------------------------------------------
                                %  SPIKE WAVEFORM ANALYSIS
                                % ---------------------------------------------------------------

                                % Parameters for spike alignment and analysis
                                pre_spike_ms = 10;   % ms before peak
                                post_spike_ms = 10;  % ms after peak
                                pre_pts = round(pre_spike_ms / 1000 / dt);
                                post_pts = round(post_spike_ms / 1000 / dt);

                                % Extract and align individual spike waveforms
                                all_spikes_aligned = [];

                                for k = 1:length(spikes_all)
                                    spike_idx = spikes_all(k);

                                    % Find local maximum within reasonable window after threshold crossing
                                    search_end = min(spike_idx + post_pts, length(Vm_all));
                                    search_window = spike_idx : search_end;

                                    if length(search_window) > 1
                                        [~, peak_rel_idx] = max(Vm_all(search_window));
                                        peak_idx = spike_idx + peak_rel_idx - 1;

                                        % Extract spike waveform centered on peak
                                        start_idx = peak_idx - pre_pts;
                                        end_idx = peak_idx + post_pts;

                                        if start_idx >= 1 && end_idx <= length(Vm_all)
                                            spike_waveform = Vm_all(start_idx : end_idx);
                                            all_spikes_aligned = [all_spikes_aligned, spike_waveform];
                                        end
                                    end
                                end

                                % Calculate average spike waveform and width
                                if ~isempty(all_spikes_aligned)
                                    avg_spike_waveform = mean(all_spikes_aligned, 2);

                                    % Compute spike width at half-maximum
                                    spike_peak = max(avg_spike_waveform);
                                    half_max = spike_peak / 2;
                                    above_half_idx = find(avg_spike_waveform >= half_max);

                                    if length(above_half_idx) >= 2
                                        width_pts = above_half_idx(end) - above_half_idx(1) + 1;
                                        width_ms = width_pts * dt * 1000;
                                        fprintf('            Average spike width at half-maximum: %.2f ms\n', width_ms);
                                    else
                                        width_ms = NaN;
                                        fprintf('            Could not compute spike width reliably\n');
                                    end

                                    fprintf('            Analyzed %d aligned spike waveforms\n', size(all_spikes_aligned, 2));
                                else
                                    fprintf('            Warning: No valid spike waveforms found for alignment\n');
                                    width_ms = NaN;
                                    avg_spike_waveform = [];
                                end

                                % ---------------------------------------------------------------
                                %  COMPILE AND STORE RESULTS
                                % ---------------------------------------------------------------

                                % Create comprehensive results structure
                                results = struct();

                                % Basic info
                                results.cell_id = cell_name;
                                results.Vm_cleaned = Vm_cleaned;  % Store cleaned voltage
                                results.Vm_all = Vm_all;
                                results.protocol = protocol_name;
                                results.date = date_name;
                                results.group = group_name;
                                results.frequency = num2str(freq_value);  % Convert to string for display
                                results.selected_sd = num2str(selected_sd_value);  % Convert to string for display
                                results.n_epochs = max_epochs;
                                results.n_trials = n_trials;
                                results.n_timepoints = n_timepoints;
                                results.dt = dt;
                                results.total_duration_s = length(Vm_all) * dt;

                                % Spike detection results
                                results.spike_indices = elbow_indices;
                                results.spike_times_s = true_spike_times;
                                results.spike_times_ms = true_spike_times_ms;
                                results.n_spikes = length(elbow_indices);
                                results.firing_rate_Hz = length(elbow_indices) / (length(Vm_all) * dt);
                                results.spike_diagnostic = diagnostic_info;

                                % Linear filter results
                                results.linear_filter = filt;
                                results.filter_lag = lag;
                                results.filter_correlation = r;
                                results.predicted_voltage = Vm_pred;

                                % Spike waveform results
                                results.avg_spike_short = avg_spike_short;
                                results.avg_spike_corrected = avg_spike_corrected;
                                results.avg_spike_waveform = avg_spike_waveform;
                                results.spike_width_ms = width_ms;
                                results.n_aligned_spikes = size(all_spikes_aligned, 2);

                                % STA results
                                results.STA_current = STA;
                                results.STA_voltage = STA_Vm;
                                results.STA_valid_spikes = valid_spikes;
                                results.STA_window_before_ms = win_before * dt * 1000;
                                results.STA_window_after_ms = win_after * dt * 1000;

                                % Processing info
                                results.analysis_timestamp = datetime('now');
                                results.success = true;

                                % Store results at the SD node using correct key
                                selected_sd_node.custom.put('results', riekesuite.util.toJavaMap(results));
                                selected_sd_node.parent.custom.put('results', riekesuite.util.toJavaMap(results));

                                % Add to collection
                                analysis_results{results_counter} = results;
                                results_counter = results_counter + 1;

                                fprintf('          >>> SUCCESS: Analysis complete <<<\n');
                                fprintf('            Spikes: %d (%.2f Hz) | Filter r: %.3f | Width: %.2f ms\n', ...
                                    results.n_spikes, results.firing_rate_Hz, results.filter_correlation, results.spike_width_ms);

                                % Log success using LoggingUtils
                                log_entry = LoggingUtils.logSuccessfulAnalysis(cell_name, protocol_name, ...
                                    freq_value, selected_sd_value, max_epochs, results);
                                analysis_log{log_counter} = log_entry;
                                log_counter = log_counter + 1;

                            catch ME
                                fprintf('          >>> ERROR: Analysis failed <<<\n');
                                fprintf('            Error: %s\n', ME.message);

                                % Log error using LoggingUtils
                                [log_entry, failed_analysis] = LoggingUtils.logFailedAnalysis(cell_name, ...
                                    protocol_name, freq_value, selected_sd_value, max_epochs, ME);

                                failed_analyses{end+1} = failed_analysis;
                                analysis_log{log_counter} = log_entry;
                                log_counter = log_counter + 1;
                            end

                        else
                            fprintf('          No epochs found in any SD level\n');
                        end
                    else
                        fprintf('          No Current SD children found\n');
                    end
                end
            end
        end
    end
end

% -----------------------------------------------------------------------
%  FINAL SUMMARY section 1
% -----------------------------------------------------------------------

fprintf('\n=== NAVIGATION, SELECTION & ANALYSIS SUMMARY ===\n');
fprintf('Total analyses attempted: %d\n', length(analysis_results) + length(failed_analyses));
fprintf('Successful analyses: %d\n', length(analysis_results));
fprintf('Failed analyses: %d\n', length(failed_analyses));

if ~isempty(analysis_results)
    % Extract summary statistics
    firing_rates = cellfun(@(x) x.firing_rate_Hz, analysis_results);
    filter_corrs = cellfun(@(x) x.filter_correlation, analysis_results);
    spike_counts = cellfun(@(x) x.n_spikes, analysis_results);

    fprintf('\nSummary Statistics:\n');
    fprintf('  Firing rates: %.2f ± %.2f Hz (range: %.2f - %.2f)\n', ...
        mean(firing_rates), std(firing_rates), min(firing_rates), max(firing_rates));
    fprintf('  Filter correlations: %.3f ± %.3f (range: %.3f - %.3f)\n', ...
        mean(filter_corrs), std(filter_corrs), min(filter_corrs), max(filter_corrs));
    fprintf('  Spike counts: %.1f ± %.1f (range: %d - %d)\n', ...
        mean(spike_counts), std(spike_counts), min(spike_counts), max(spike_counts));

    % Display successful analyses
    fprintf('\nSuccessful Analyses:\n');
    for i = 1:length(analysis_results)
        res = analysis_results{i};
        % Convert frequency and SD to strings for proper display
        freq_str = num2str(res.frequency);
        sd_str = num2str(res.selected_sd);
        fprintf('  %d. %s | %s | Freq:%s | SD:%s | Epochs:%d | Spikes:%d (%.2f Hz) | r:%.3f\n', ...
            i, res.cell_id, res.protocol, freq_str, sd_str, ...
            res.n_epochs, res.n_spikes, res.firing_rate_Hz, res.filter_correlation);
    end
end

if ~isempty(failed_analyses)
    fprintf('\nFailed Analyses:\n');
    for i = 1:length(failed_analyses)
        fail = failed_analyses{i};
        fprintf('  %d. %s | %s | Freq:%s | SD:%s | Error: %s\n', ...
            i, fail.cell_id, fail.protocol, fail.frequency, fail.selected_sd, fail.error_message);
    end
end

fprintf('\n=== RESULTS STORED AT SD LEVEL NODES ===\n');
fprintf('Results can be queried later using: node.custom.get(''results'')\n');
fprintf('Ready for population analysis when needed.\n');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SECTION 2: SPIKE RESPONSE MODEL ANALYSIS WITH HOLD-OUT VALIDATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
fprintf('\n=== SECTION 2: SPIKE RESPONSE MODEL ANALYSIS ===\n');

% Check that Section 1 has been completed
if ~exist('analysis_results', 'var') || isempty(analysis_results)
    error(['Section 1 must be completed first.\n' ...
        'Run Section 1 to perform elementary analysis and store results at SD nodes.']);
end

fprintf('Found %d successful analyses from Section 1\n', length(analysis_results));

% Initialize SRM analysis tracking
srm_results = {};
srm_failed = {};
srm_counter = 1;

% Quality filter parameters for SRM analysis
quality_thresholds = struct();
quality_thresholds.min_spikes = 50;              % Minimum spike count
quality_thresholds.min_firing_rate = 0.5;        % Minimum firing rate (Hz)
quality_thresholds.min_filter_correlation = 0.3; % Minimum linear filter correlation
quality_thresholds.min_duration = 30;            % Minimum recording duration (s)
quality_thresholds.max_firing_rate = 500;         % Maximum firing rate (Hz) to exclude bursting

fprintf('\nQuality thresholds for SRM analysis:\n');
fprintf('  Min spikes: %d\n', quality_thresholds.min_spikes);
fprintf('  Min firing rate: %.1f Hz\n', quality_thresholds.min_firing_rate);
fprintf('  Max firing rate: %.1f Hz\n', quality_thresholds.max_firing_rate);
fprintf('  Min filter correlation: %.2f\n', quality_thresholds.min_filter_correlation);
fprintf('  Min duration: %.1f s\n', quality_thresholds.min_duration);

% SRM-specific parameters
srm_params = struct();
srm_params.tau_ref_ms = 2.0;                     % Refractory period (ms)
srm_params.vp_q = 4;                             % Victor-Purpura distance parameter
srm_params.holdout_duration_sec = 30;            % Hold-out validation duration
srm_params.max_iter = 2000;                      % Optimization iterations (increased)
srm_params.max_fun_evals = 5000;                 % Maximum function evaluations (increased)

fprintf('\nSRM analysis parameters:\n');
fprintf('  Refractory period: %.1f ms\n', srm_params.tau_ref_ms);
fprintf('  VP distance parameter: %.1f\n', srm_params.vp_q);
fprintf('  Hold-out duration: %.1f s\n', srm_params.holdout_duration_sec);

% Global debug log storage
global DEBUG_LOG;
global DEBUG_LOG_COUNTER;

% ===================================================================
%  SRM ANALYSIS LOOP WITH ENHANCED DEBUG LOGGING
% ===================================================================

% DEBUG MODE FLAG - Set to true to skip optimization for fast testing
DEBUG_MODE = false;  % Set to false for full analysis

% ===================================================================
%  SRM ANALYSIS LOOP WITH ENHANCED DEBUG LOGGING
% ===================================================================


% Initialize debug logging system
DebugUtils.initializeDebugLog();

if DEBUG_MODE
    fprintf('>>> RUNNING IN DEBUG MODE - Optimization skipped <<<\n');
    fprintf('>>> Debug logging enabled - tracking variable availability and indexing <<<\n');
else
    fprintf('>>> RUNNING IN FULL MODE - Optimization enabled <<<\n');
end

% Initialize counters and results collection
srm_results = {};
srm_counter = 1;

for protocol_idx = 1:CurrentNode.children.length
    protocolNode = CurrentNode.children.elements(protocol_idx);
    protocol_name = protocolNode.splitValue;
    fprintf('\nSRM Analysis - Protocol: %s\n', protocol_name);

    for date_idx = 1:protocolNode.children.length
        dateNode = protocolNode.children.elements(date_idx);
        date_name = dateNode.splitValue;
        fprintf('  Date: %s\n', date_name);

        for cell_idx = 1:dateNode.children.length
            cellNode = dateNode.children.elements(cell_idx);
            cell_name = cellNode.splitValue;
            fprintf('    Cell: %s\n', cell_name);

            % Level 4: Epoch Group (protocolSettings(epochGroup:label))
            for group_idx = 1:cellNode.children.length
                groupNode = cellNode.children.elements(group_idx);
                group_name = groupNode.splitValue;
                fprintf('      Group: %s\n', group_name);

                % ===================================================================
                %  FREQUENCY LOOP: Process each frequency independently
                % ===================================================================
                fprintf('        >>> Debug: Group has %d frequency children <<<\n', groupNode.children.length);

                for freq_idx = 1:groupNode.children.length
                    freqNode = groupNode.children.elements(freq_idx);
                    freq_value = freqNode.splitValue;
                    freq_str = num2str(freq_value);
                    fprintf('        Frequency: %s (idx %d/%d)\n', freq_str, freq_idx, groupNode.children.length);

                    % For THIS frequency, find the SD level with most epochs
                    if freqNode.children.length > 0
                        fprintf('          Current SD levels found: %d\n', freqNode.children.length);

                        % Count epochs for each SD level for THIS frequency
                        sd_epoch_counts = [];
                        sd_values = {};
                        sd_nodes = {};

                        for sd_idx = 1:freqNode.children.length
                            sdNode_temp = freqNode.children.elements(sd_idx);
                            sd_value_temp = sdNode_temp.splitValue;
                            sd_str = num2str(sd_value_temp);
                            n_epochs = sdNode_temp.epochList.length;

                            sd_epoch_counts(end+1) = n_epochs;
                            sd_values{end+1} = sd_value_temp;
                            sd_nodes{end+1} = sdNode_temp;

                            fprintf('            SD %s: %d epochs\n', sd_str, n_epochs);
                        end

                        % Find SD level with maximum epochs FOR THIS FREQUENCY
                        [max_epochs, max_idx] = max(sd_epoch_counts);

                        if max_epochs > 0
                            selected_sd_node = sd_nodes{max_idx};
                            selected_sd_value = sd_values{max_idx};
                            sd_str = num2str(selected_sd_value);

                            fprintf('          >>> SELECTED for freq %s: SD %s with %d epochs <<<\n', ...
                                freq_str, sd_str, max_epochs);

                            % DEBUG LOG: Record indexing success
                            if DEBUG_MODE
                                DebugUtils.debugLogIndexing(cell_name, freq_str, sd_str, 'SD_selection', true, ...
                                    sprintf('Selected SD %s from %d options with %d epochs', ...
                                    sd_str, length(sd_nodes), max_epochs));
                            end

                            % Use the selected node as sdNode for the rest of the analysis
                            sdNode = selected_sd_node;
                            sd_value = selected_sd_value;

                            % Check if this node has stored results (from Section 1)
                            if sdNode.custom.containsKey('results')
                                stored_results = sdNode.custom.get('results');

                                if ~isempty(stored_results)
                                    fprintf('          >>> Found stored results for freq %s, SD %s <<<\n', ...
                                        freq_value, sd_value);

                                    % DEBUG LOG: Record successful entry into analysis
                                    if DEBUG_MODE
                                        debugLogEntry(cell_name, freq_str, sd_str, max_epochs);
                                    end

                                    % Extract key metrics for quality filtering
                                    try
                                        % DEBUG LOG: Check each required variable
                                        if DEBUG_MODE
                                            fprintf('            >>> DEBUG: Checking variable availability <<<\n');

                                            % Check basic quality metrics
                                            [exists_spikes, n_spikes, ~] = checkAndLogVariable(stored_results, 'n_spikes', cell_name, freq_str, sd_str);
                                            [exists_rate, firing_rate, ~] = checkAndLogVariable(stored_results, 'firing_rate_Hz', cell_name, freq_str, sd_str);
                                            [exists_corr, filter_corr, ~] = checkAndLogVariable(stored_results, 'filter_correlation', cell_name, freq_str, sd_str);
                                            [exists_dur, duration, ~] = checkAndLogVariable(stored_results, 'total_duration_s', cell_name, freq_str, sd_str);

                                            % Check analysis data variables
                                            [exists_vm_clean, ~, ~] = checkAndLogVariable(stored_results, 'Vm_cleaned', cell_name, freq_str, sd_str);
                                            [exists_vm_all, ~, ~] = checkAndLogVariable(stored_results, 'Vm_all', cell_name, freq_str, sd_str);
                                            [exists_spikes_idx, ~, ~] = checkAndLogVariable(stored_results, 'spike_indices', cell_name, freq_str, sd_str);
                                            [exists_avg_spike, ~, ~] = checkAndLogVariable(stored_results, 'avg_spike_short', cell_name, freq_str, sd_str);
                                            [exists_dt, ~, ~] = checkAndLogVariable(stored_results, 'dt', cell_name, freq_str, sd_str);
                                        else
                                            % Standard variable retrieval for full mode
                                            n_spikes = get(stored_results, 'n_spikes');
                                            firing_rate = get(stored_results, 'firing_rate_Hz');
                                            filter_corr = get(stored_results, 'filter_correlation');
                                            duration = get(stored_results, 'total_duration_s');
                                        end

                                        fprintf('            Quality check: %d spikes, %.2f Hz, r=%.3f, %.1fs\n', ...
                                            n_spikes, firing_rate, filter_corr, duration);

                                        % Apply quality filters - WARNINGS ONLY, NO SKIPPING
                                        passes_quality = true;
                                        quality_warnings = {};

                                        % Quality checks with debug logging
                                        if n_spikes < quality_thresholds.min_spikes
                                            quality_warnings{end+1} = sprintf('low_spikes(%d)', n_spikes);
                                        end

                                        if firing_rate < quality_thresholds.min_firing_rate
                                            quality_warnings{end+1} = sprintf('low_rate(%.2f)', firing_rate);
                                        end

                                        if firing_rate > quality_thresholds.max_firing_rate
                                            quality_warnings{end+1} = sprintf('high_rate(%.2f)', firing_rate);
                                        end

                                        % Only real failure condition
                                        if filter_corr < quality_thresholds.min_filter_correlation
                                            passes_quality = false;
                                            quality_warnings{end+1} = sprintf('low_corr(%.3f)', filter_corr);
                                        end

                                        if duration < quality_thresholds.min_duration
                                            quality_warnings{end+1} = sprintf('short_dur(%.1f)', duration);
                                        end

                                        % Display warnings if any
                                        if ~isempty(quality_warnings)
                                            fprintf('            ⚠ QUALITY WARNINGS: %s\n', strjoin(quality_warnings, ', '));
                                            if DEBUG_MODE
                                                debugLog('WARN', 'QUALITY', cell_name, freq_str, sd_str, ...
                                                    'Quality warnings detected', 'warnings', strjoin(quality_warnings, ', '));
                                            end
                                        end

                                        if passes_quality
                                            fprintf('            ✓ QUALITY CHECK PASSED - Starting SRM analysis\n');
                                            fprintf('            >>> Analyzing: %s | freq=%s | SD=%s <<<\n', ...
                                                cell_name, freq_str, sd_str);

                                            % ===================================================================
                                            %  PERFORM SPIKE RESPONSE MODEL ANALYSIS WITH DEBUG MODE
                                            % ===================================================================

                                            try
                                                fprintf('            Retrieving data from Section 1...\n');

                                                % Check for the specific keys we need
                                                required_keys = {'Vm_cleaned', 'Vm_all', 'spike_indices', 'avg_spike_short', 'dt'};
                                                missing_keys = {};

                                                for i = 1:length(required_keys)
                                                    key = required_keys{i};
                                                    if ~stored_results.containsKey(key)
                                                        missing_keys{end+1} = key;
                                                    end
                                                end

                                                % DEBUG LOG: Record missing keys
                                                if DEBUG_MODE && ~isempty(missing_keys)
                                                    debugLog('WARN', 'VARIABLE', cell_name, freq_str, sd_str, ...
                                                        'Missing required keys for analysis', ...
                                                        'missing_keys', strjoin(missing_keys, ', '), ...
                                                        'total_required', length(required_keys));
                                                end

                                                % Get data with fallback if needed
                                                if ~isempty(missing_keys)
                                                    fprintf('            Missing keys detected: %s, using fallback...\n', strjoin(missing_keys, ', '));

                                                    % DEBUG LOG: Record fallback data extraction
                                                    if DEBUG_MODE
                                                        debugLog('INFO', 'INDEXING', cell_name, freq_str, sd_str, ...
                                                            'Using fallback data extraction', ...
                                                            'reason', 'missing_stored_keys');
                                                    end

                                                    % Re-extract data
                                                    EpochData = getSelectedData(sdNode.epochList, params.Amp);
                                                    Vm_all = reshape(EpochData', [], 1);

                                                    % Re-filter voltage
                                                    sampling_rate_Hz = 1/params.SamplingInterval;
                                                    cutoff_freq_Hz = 90;
                                                    Wn = cutoff_freq_Hz / (sampling_rate_Hz/2);
                                                    [b, a] = butter(4, Wn, 'low');
                                                    Vm_cleaned = filtfilt(b, a, Vm_all);

                                                    dt = params.SamplingInterval;
                                                    elbow_indices = get(stored_results, 'spike_indices');
                                                    avg_spike_short = zeros(100, 1); % Default

                                                else
                                                    % All keys found, retrieve normally
                                                    if DEBUG_MODE
                                                        debugLog('SUCCESS', 'VARIABLE', cell_name, freq_str, sd_str, ...
                                                            'All required keys found in stored results');
                                                    end

                                                    Vm_cleaned = get(stored_results, 'Vm_cleaned');
                                                    Vm_all = get(stored_results, 'Vm_all');
                                                    elbow_indices = get(stored_results, 'spike_indices');
                                                    avg_spike_short = get(stored_results, 'avg_spike_short');
                                                    dt = get(stored_results, 'dt');
                                                end

                                                % Final validation
                                                if isempty(Vm_all) || isempty(Vm_cleaned) || isempty(elbow_indices)
                                                    error('Critical data missing: Vm_all=%d, Vm_cleaned=%d, spikes=%d', ...
                                                        length(Vm_all), length(Vm_cleaned), length(elbow_indices));
                                                end

                                                % DEBUG LOG: Record successful data retrieval
                                                if DEBUG_MODE
                                                    debugLog('SUCCESS', 'VARIABLE', cell_name, freq_str, sd_str, ...
                                                        'Data validation passed', ...
                                                        'vm_all_length', length(Vm_all), ...
                                                        'vm_cleaned_length', length(Vm_cleaned), ...
                                                        'n_spikes', length(elbow_indices));
                                                end

                                                fprintf('            Data retrieved: %d samples, %d spikes\n', ...
                                                    length(Vm_all), length(elbow_indices));

                                                % Calculate total duration and check hold-out feasibility
                                                t_total = (length(Vm_all) - 1) * dt;
                                                holdout_samples = round(srm_params.holdout_duration_sec / dt);

                                                % Ensure holdout is never longer than training (max 30% of total)
                                                max_holdout_ratio = 0.3;
                                                max_holdout_samples = round(max_holdout_ratio * length(Vm_all));

                                                if holdout_samples >= max_holdout_samples
                                                    holdout_samples = max_holdout_samples;
                                                    actual_holdout_duration = holdout_samples * dt;
                                                    fprintf('            Warning: Holdout duration reduced to %.1fs (%.1f%% of total)\n', ...
                                                        actual_holdout_duration, max_holdout_ratio * 100);
                                                else
                                                    actual_holdout_duration = srm_params.holdout_duration_sec;
                                                end

                                                % Ensure we have enough data for both training and holdout
                                                min_train_samples = round(0.5 * length(Vm_all)); % At least 50% for training
                                                if holdout_samples > (length(Vm_all) - min_train_samples)
                                                    holdout_samples = length(Vm_all) - min_train_samples;
                                                    actual_holdout_duration = holdout_samples * dt;
                                                    fprintf('            Warning: Holdout adjusted to %.1fs to ensure sufficient training data\n', ...
                                                        actual_holdout_duration);
                                                end

                                                % Split data for hold-out validation
                                                train_end_idx = length(Vm_all) - holdout_samples;
                                                holdout_start_idx = train_end_idx + 1;

                                                Vm_all_train = Vm_all(1:train_end_idx);
                                                Vm_cleaned_train = Vm_cleaned(1:train_end_idx);
                                                Vm_all_holdout = Vm_all(holdout_start_idx:end);
                                                Vm_cleaned_holdout = Vm_cleaned(holdout_start_idx:end);

                                                train_duration = train_end_idx * dt;
                                                holdout_percentage = (actual_holdout_duration / t_total) * 100;

                                                % Split spike indices for training/holdout
                                                elbow_indices_train = elbow_indices(elbow_indices <= train_end_idx);
                                                elbow_indices_holdout = elbow_indices(elbow_indices > train_end_idx) - train_end_idx;

                                                % Calculate threshold values at spike times
                                                threshold_values_train = Vm_all_train(elbow_indices_train);
                                                threshold_values_holdout = Vm_all_holdout(elbow_indices_holdout);

                                                fprintf('            Hold-out: %.1fs train, %.1fs holdout, %d/%d spikes\n', ...
                                                    train_duration, actual_holdout_duration, length(elbow_indices_train), length(elbow_indices_holdout));

                                                % DEBUG LOG: Record data splitting success
                                                if DEBUG_MODE
                                                    debugLog('SUCCESS', 'INDEXING', cell_name, freq_str, sd_str, ...
                                                        'Data splitting completed', ...
                                                        'train_duration', train_duration, ...
                                                        'holdout_duration', actual_holdout_duration, ...
                                                        'train_spikes', length(elbow_indices_train), ...
                                                        'holdout_spikes', length(elbow_indices_holdout));
                                                end

                                                % Create SpikeResponseModel objects
                                                model_train = SpikeResponseModel( ...
                                                    Vm_cleaned_train, Vm_all_train, dt, avg_spike_short, ...
                                                    srm_params.tau_ref_ms, elbow_indices_train, threshold_values_train, ...
                                                    cell_name, protocol_name);

                                                model_holdout = SpikeResponseModel( ...
                                                    Vm_cleaned_holdout, Vm_all_holdout, dt, avg_spike_short, ...
                                                    srm_params.tau_ref_ms, elbow_indices_holdout, threshold_values_holdout, ...
                                                    cell_name, protocol_name);

                                                % DEBUG LOG: Record model creation success
                                                if DEBUG_MODE
                                                    debugLog('SUCCESS', 'INDEXING', cell_name, freq_str, sd_str, ...
                                                        'SpikeResponseModel objects created successfully');
                                                end

                                                % ---------------------------------------------------------------
                                                %  CONDITIONAL OPTIMIZATION BASED ON DEBUG MODE
                                                % ---------------------------------------------------------------

                                                if DEBUG_MODE
                                                    fprintf('            >>> DEBUG MODE: Using default parameters (no optimization) <<<\n');

                                                    % Use reasonable default parameters without optimization
                                                    theta0_opt = mean(threshold_values_train) - 2;
                                                    best_kernel = 'exponential';

                                                    % Default exponential kernel parameters
                                                    A_opt_exp = 5.0;
                                                    tau_opt_exp = 0.03;
                                                    kernel_params = [A_opt_exp, tau_opt_exp];

                                                    % Mock VP values for testing (realistic but not optimized)
                                                    final_vp_train_raw = 0.5 + 0.1 * randn();
                                                    final_vp_holdout_raw = final_vp_train_raw * (1 + 0.2 * randn());

                                                    % Normalize VP distances by time duration for fair comparison
                                                    final_vp_train = final_vp_train_raw / train_duration;
                                                    final_vp_holdout = final_vp_holdout_raw / actual_holdout_duration;
                                                    generalization_ratio = final_vp_holdout / final_vp_train;

                                                    fprintf('            DEBUG: Using theta0=%.2f, A=%.2f, tau=%.3fs\n', ...
                                                        theta0_opt, A_opt_exp, tau_opt_exp);
                                                    fprintf('            DEBUG: Mock VP values (normalized) - train=%.4f/s, holdout=%.4f/s\n', ...
                                                        final_vp_train, final_vp_holdout);
                                                    fprintf('            DEBUG: Raw VP values - train=%.4f, holdout=%.4f\n', ...
                                                        final_vp_train_raw, final_vp_holdout_raw);

                                                    % DEBUG LOG: Record debug mode parameters
                                                    debugLog('INFO', 'RESULT', cell_name, freq_str, sd_str, ...
                                                        'Debug mode parameters set', ...
                                                        'theta0', theta0_opt, ...
                                                        'kernel', best_kernel, ...
                                                        'A', A_opt_exp, ...
                                                        'tau', tau_opt_exp, ...
                                                        'mock_vp_train_raw', final_vp_train_raw, ...
                                                        'mock_vp_holdout_raw', final_vp_holdout_raw, ...
                                                        'mock_vp_train_norm', final_vp_train, ...
                                                        'mock_vp_holdout_norm', final_vp_holdout);

                                                    % Create simple figure save path for debug mode (no actual figure creation)
                                                    cellTypeNode = sdNode.parent.parent.parent.parent.parent;
                                                    cell_type_raw = cellTypeNode.splitValue;
                                                    cell_type_label = strrep(cell_type_raw, '\', '/');
                                                    base_path = '/Users/maxwellsdm/Library/CloudStorage/GoogleDrive-maxwellsdm1867@gmail.com/.shortcut-targets-by-id/1CbZnfugdi-p4jM3fe7t-NyGk5v2ZcPdk/ParasolCenterSurround/figures/ScienceJuiceFactory/currinjt/overview/';
                                                    fig_path = [base_path cell_type_label '/'];

                                                    if ~exist(fig_path, 'dir')
                                                        mkdir(fig_path);
                                                    end

                                                    date_str = date_name(1:11);
                                                    cell_name_lower = lower(cell_name);
                                                    srm_filename = sprintf('%s-%s-freq-%s-spikemodel-DEBUG.png', date_str, cell_name_lower, freq_str);
                                                    srm_save_path = [fig_path srm_filename];

                                                    fprintf('            DEBUG: Figure path created (no actual figure): %s\n', srm_filename);

                                                    % DEBUG LOG: Record path creation
                                                    debugLog('SUCCESS', 'INDEXING', cell_name, freq_str, sd_str, ...
                                                        'Figure path created', ...
                                                        'cell_type', cell_type_label, ...
                                                        'filename', srm_filename);

                                                else
                                                    fprintf('            >>> FULL MODE: Running optimization <<<\n');

                                                    % ---------------------------------------------------------------
                                                    %  EXPONENTIAL KERNEL OPTIMIZATION (FASTER)
                                                    % ---------------------------------------------------------------

                                                    theta0_init = mean(threshold_values_train) - 2;
                                                    A_init = 5;
                                                    tau_init = 0.03;
                                                    init_params_exp = [theta0_init, A_init, tau_init];

                                                    % Faster optimization settings
                                                    options = optimset('Display', 'off', 'MaxFunEvals', 200, ...
                                                        'MaxIter', 100, 'TolX', 1e-6, 'TolFun', 1e-4);

                                                    fast_loss_exp_fn = @(params) compute_fast_vp_loss_exponential( ...
                                                        params, model_train, srm_params.vp_q);

                                                    % Reduced multi-start optimization (faster)
                                                    n_starts = 2;  % Reduced from 3
                                                    best_params_exp = init_params_exp;
                                                    best_vp_exp = inf;

                                                    for start_idx = 1:n_starts
                                                        if start_idx == 1
                                                            start_params = init_params_exp;
                                                        else
                                                            theta0_start = init_params_exp(1) + randn() * 2;  % Smaller variation
                                                            A_start = max(1, init_params_exp(2) + randn() * 1);
                                                            tau_start = max(0.01, init_params_exp(3) + randn() * 0.01);
                                                            start_params = [theta0_start, A_start, tau_start];
                                                        end

                                                        try
                                                            [opt_params_trial, vp_trial] = fminsearch(fast_loss_exp_fn, start_params, options);
                                                            if vp_trial < best_vp_exp
                                                                best_params_exp = opt_params_trial;
                                                                best_vp_exp = vp_trial;
                                                            end
                                                        catch
                                                            % Continue with other starts
                                                        end
                                                    end

                                                    opt_params_exp = best_params_exp;
                                                    final_vp_exp_train = best_vp_exp;
                                                    theta0_opt_exp = opt_params_exp(1);
                                                    A_opt_exp = opt_params_exp(2);
                                                    tau_opt_exp = opt_params_exp(3);

                                                    % ---------------------------------------------------------------
                                                    %  LINEAR RISE + EXPONENTIAL DECAY KERNEL OPTIMIZATION (FASTER)
                                                    % ---------------------------------------------------------------

                                                    T_rise_init = 0.003;
                                                    tau_decay_init = 0.04;
                                                    init_params_linexp = [theta0_init, A_init, T_rise_init, tau_decay_init];

                                                    fast_loss_linexp_fn = @(params) compute_fast_vp_loss_linexp( ...
                                                        params, model_train, srm_params.vp_q);

                                                    best_params_linexp = init_params_linexp;
                                                    best_vp_linexp = inf;

                                                    for start_idx = 1:n_starts
                                                        if start_idx == 1
                                                            start_params = init_params_linexp;
                                                        else
                                                            theta0_start = init_params_linexp(1) + randn() * 2;  % Smaller variation
                                                            A_start = max(1, init_params_linexp(2) + randn() * 1);
                                                            T_rise_start = max(0.001, init_params_linexp(3) + randn() * 0.002);
                                                            tau_decay_start = max(0.01, init_params_linexp(4) + randn() * 0.01);
                                                            start_params = [theta0_start, A_start, T_rise_start, tau_decay_start];
                                                        end

                                                        try
                                                            [opt_params_trial, vp_trial] = fminsearch(fast_loss_linexp_fn, start_params, options);
                                                            if vp_trial < best_vp_linexp
                                                                best_params_linexp = opt_params_trial;
                                                                best_vp_linexp = vp_trial;
                                                            end
                                                        catch
                                                            % Continue with other starts
                                                        end
                                                    end

                                                    opt_params_linexp = best_params_linexp;
                                                    final_vp_linexp_train = best_vp_linexp;
                                                    theta0_opt_linexp = opt_params_linexp(1);
                                                    A_opt_linexp = opt_params_linexp(2);
                                                    T_rise_opt_linexp = opt_params_linexp(3);
                                                    tau_decay_opt_linexp = opt_params_linexp(4);

                                                    % ---------------------------------------------------------------
                                                    %  HOLD-OUT VALIDATION
                                                    % ---------------------------------------------------------------

                                                    fast_loss_exp_holdout_fn = @(params) compute_fast_vp_loss_exponential( ...
                                                        params, model_holdout, srm_params.vp_q);
                                                    final_vp_exp_holdout = fast_loss_exp_holdout_fn(opt_params_exp);

                                                    fast_loss_linexp_holdout_fn = @(params) compute_fast_vp_loss_linexp( ...
                                                        params, model_holdout, srm_params.vp_q);
                                                    final_vp_linexp_holdout = fast_loss_linexp_holdout_fn(opt_params_linexp);

                                                    % Select best model based on hold-out performance
                                                    if final_vp_exp_holdout < final_vp_linexp_holdout
                                                        best_kernel = 'exponential';
                                                        theta0_opt = theta0_opt_exp;
                                                        final_vp_train = final_vp_exp_train;
                                                        final_vp_holdout = final_vp_exp_holdout;
                                                        kernel_params = [A_opt_exp, tau_opt_exp];
                                                    else
                                                        best_kernel = 'linear_rise_exp_decay';
                                                        theta0_opt = theta0_opt_linexp;
                                                        final_vp_train = final_vp_linexp_train;
                                                        final_vp_holdout = final_vp_linexp_holdout;
                                                        kernel_params = [A_opt_linexp, T_rise_opt_linexp, tau_decay_opt_linexp];
                                                    end

                                                    % Normalize VP distances by time duration for fair comparison
                                                    vp_train_normalized = final_vp_train / train_duration;
                                                    vp_holdout_normalized = final_vp_holdout / actual_holdout_duration;

                                                    % Store both raw and normalized values
                                                    final_vp_train_raw = final_vp_train;
                                                    final_vp_holdout_raw = final_vp_holdout;
                                                    final_vp_train = vp_train_normalized;
                                                    final_vp_holdout = vp_holdout_normalized;

                                                    generalization_ratio = final_vp_holdout / final_vp_train;

                                                    % ---------------------------------------------------------------
                                                    %  CREATE AND SAVE DIAGNOSTIC FIGURE
                                                    % ---------------------------------------------------------------

                                                    % Create full model for diagnostics
                                                    model_full = SpikeResponseModel( ...
                                                        Vm_cleaned, Vm_all, dt, avg_spike_short, srm_params.tau_ref_ms, ...
                                                        elbow_indices, [threshold_values_train(:); threshold_values_holdout(:)], ...
                                                        cell_name, protocol_name);

                                                    % Define optimal kernel function
                                                    if strcmp(best_kernel, 'exponential')
                                                        kernel_opt = @(t) A_opt_exp * exp(-t / tau_opt_exp);
                                                    else
                                                        kernel_opt = @(t) (t < T_rise_opt_linexp) .* (A_opt_linexp / T_rise_opt_linexp .* t) + ...
                                                            (t >= T_rise_opt_linexp) .* (A_opt_linexp * exp(-(t - T_rise_opt_linexp) / tau_decay_opt_linexp));
                                                    end

                                                    % Run simulation for diagnostics
                                                    [spikes_full, V_pred_full, threshold_trace_full, spike_times_full, spike_V_full] = ...
                                                        model_full.simulateFast(theta0_opt, kernel_opt, 'profile', false);

                                                    % Create figure path (same as Section 1)
                                                    cellTypeNode = sdNode.parent.parent.parent.parent.parent;
                                                    cell_type_raw = cellTypeNode.splitValue;
                                                    cell_type_label = strrep(cell_type_raw, '\', '/');
                                                    base_path = '/Users/maxwellsdm/Library/CloudStorage/GoogleDrive-maxwellsdm1867@gmail.com/.shortcut-targets-by-id/1CbZnfugdi-p4jM3fe7t-NyGk5v2ZcPdk/ParasolCenterSurround/figures/ScienceJuiceFactory/currinjt/overview/';
                                                    fig_path = [base_path cell_type_label '/'];

                                                    if ~exist(fig_path, 'dir')
                                                        mkdir(fig_path);
                                                    end

                                                    % Create filename
                                                    date_str = date_name(1:11);
                                                    cell_name_lower = lower(cell_name);
                                                    srm_filename = sprintf('%s-%s-freq-%s-spikemodel.png', date_str, cell_name_lower, freq_str);
                                                    srm_save_path = [fig_path srm_filename];

                                                    % Create and save figure using model.diagnostics()
                                                    try
                                                        zoom_window = [0, min(3, t_total)];
                                                        current_fig = gcf;
                                                        model_full.diagnostics(V_pred_full, threshold_trace_full, spikes_full, final_vp_holdout, zoom_window, srm_save_path, kernel_params, spike_times_full, spike_V_full);

                                                        if ~exist(srm_save_path, 'file')
                                                            saveas(current_fig, srm_save_path);
                                                        end

                                                        close(current_fig);
                                                        clear V_pred_full threshold_trace_full spikes_full spike_times_full spike_V_full;
                                                        fprintf('            SRM diagnostic figure saved: %s\n', srm_filename);
                                                    catch
                                                        close all;
                                                        srm_save_path = 'FIGURE_FAILED';
                                                        fprintf('            Warning: Figure creation failed\n');
                                                    end

                                                end  % End DEBUG_MODE conditional

                                                % ---------------------------------------------------------------
                                                %  COMMON POST-PROCESSING (regardless of debug mode)
                                                % ---------------------------------------------------------------

                                                % Calculate performance metrics
                                                n_true_spikes = length(elbow_indices);
                                                true_rate = n_true_spikes / t_total;
                                                pred_rate = true_rate; % Simplified estimate
                                                rate_accuracy = 100 * min(pred_rate, true_rate) / max(pred_rate, true_rate);

                                                % ---------------------------------------------------------------
                                                %  STORE RESULTS AT FREQUENCY NODE
                                                % ---------------------------------------------------------------

                                                % Create SRM results structure with additional fields for spike generation
                                                srm_results_struct = struct();

                                                % Original fields
                                                srm_results_struct.debug_mode = DEBUG_MODE;
                                                srm_results_struct.best_kernel = best_kernel;
                                                srm_results_struct.best_theta0 = theta0_opt;
                                                srm_results_struct.best_kernel_params = kernel_params;
                                                srm_results_struct.best_holdout_vp = final_vp_holdout;
                                                srm_results_struct.best_train_vp = final_vp_train;
                                                srm_results_struct.best_holdout_vp_raw = final_vp_holdout_raw;
                                                srm_results_struct.best_train_vp_raw = final_vp_train_raw;
                                                srm_results_struct.generalization_ratio = generalization_ratio;
                                                srm_results_struct.rate_accuracy_pct = rate_accuracy;
                                                srm_results_struct.n_spikes_total = n_true_spikes;
                                                srm_results_struct.total_duration_s = t_total;
                                                srm_results_struct.train_duration_s = train_duration;
                                                srm_results_struct.holdout_duration_s = actual_holdout_duration;
                                                srm_results_struct.holdout_percentage = holdout_percentage;
                                                srm_results_struct.selected_sd = sd_str;
                                                srm_results_struct.cell_id = cell_name;
                                                srm_results_struct.frequency = freq_str;
                                                srm_results_struct.srm_figure_path = srm_save_path;
                                                srm_results_struct.analysis_timestamp = datetime('now');
                                                srm_results_struct.success = true;

                                                % ===================================================================
                                                %  FIELDS FOR COMPLETE SPIKERESPONSEMODEL RECONSTRUCTION
                                                % ===================================================================

                                                % Timing and sampling
                                                srm_results_struct.dt = dt;                                    % Sampling interval (s)
                                                srm_results_struct.tau_ref_ms = srm_params.tau_ref_ms;        % Refractory period (ms)

                                                % Spike data for model reconstruction
                                                srm_results_struct.elbow_indices = elbow_indices;             % Original spike indices
                                                srm_results_struct.threshold_values_train = threshold_values_train;  % Training threshold values
                                                srm_results_struct.threshold_values_holdout = threshold_values_holdout; % Holdout threshold values
                                                srm_results_struct.threshold_values_all = [threshold_values_train(:); threshold_values_holdout(:)]; % Combined for model
                                                srm_results_struct.avg_spike_short = avg_spike_short;         % Average spike waveform

                                                % Data characteristics for reconstruction
                                                srm_results_struct.n_trials = n_trials;                      % Number of trials
                                                srm_results_struct.n_timepoints = n_timepoints;              % Timepoints per trial
                                                srm_results_struct.train_end_idx = train_end_idx;            % Where training data ends
                                                srm_results_struct.holdout_start_idx = holdout_start_idx;    % Where holdout data starts

                                                % Protocol info for SpikeResponseModel constructor
                                                srm_results_struct.protocol_name = protocol_name;            % Protocol name

                                                % Store at FREQUENCY NODE (not SD node) - using parent navigation
                                                selected_sd_node.parent.custom.put('srm_results', riekesuite.util.toJavaMap(srm_results_struct));
                                                % DEBUG LOG: Record successful storage
                                                if DEBUG_MODE
                                                    debugLogResult(cell_name, freq_str, sd_str, true, final_vp_train, final_vp_holdout);
                                                    debugLog('SUCCESS', 'INDEXING', cell_name, freq_str, sd_str, ...
                                                        'Results stored at frequency node', ...
                                                        'storage_location', 'selected_sd_node.parent', ...
                                                        'key', 'srm_results');
                                                end

                                                % Clear large variables
                                                clear Vm_cleaned Vm_all Vm_all_train Vm_cleaned_train Vm_all_holdout Vm_cleaned_holdout;
                                                if exist('EpochData', 'var'), clear EpochData; end

                                                % Add to collection with complete SRM results
                                                srm_results{srm_counter} = struct( ...
                                                    'cell_id', cell_name, ...
                                                    'protocol', protocol_name, ...
                                                    'frequency', freq_str, ...
                                                    'selected_sd', sd_str, ...
                                                    'debug_mode', DEBUG_MODE, ...
                                                    'best_kernel', best_kernel, ...
                                                    'best_train_vp', final_vp_train, ...
                                                    'best_holdout_vp', final_vp_holdout, ...
                                                    'best_generalization_ratio', generalization_ratio, ...
                                                    'rate_accuracy_pct', rate_accuracy);
                                                srm_counter = srm_counter + 1;

                                                if DEBUG_MODE
                                                    fprintf('          >>> SRM DEBUG SUCCESS: %s freq=%s SD=%s, VP=%.4f→%.4f/s (MOCK) <<<\n', ...
                                                        cell_name, freq_str, sd_str, final_vp_train, final_vp_holdout);
                                                else
                                                    fprintf('          >>> SRM SUCCESS: %s freq=%s SD=%s, VP=%.4f→%.4f/s <<<\n', ...
                                                        cell_name, freq_str, sd_str, final_vp_train, final_vp_holdout);
                                                end

                                            catch ME
                                                fprintf('          >>> SRM ERROR for freq %s: %s <<<\n', freq_value, ME.message);

                                                % DEBUG LOG: Record analysis error
                                                if DEBUG_MODE
                                                    debugLogResult(cell_name, freq_str, sd_str, false, NaN, NaN);
                                                    debugLog('ERROR', 'RESULT', cell_name, freq_str, sd_str, ...
                                                        'Analysis failed with error', ...
                                                        'error_message', ME.message, ...
                                                        'error_id', ME.identifier);
                                                end

                                                % Standard error logging
                                                error_log = struct();
                                                error_log.cell_id = cell_name;
                                                error_log.frequency = freq_str;
                                                error_log.selected_sd = sd_str;
                                                error_log.error_message = ME.message;
                                                error_log.error_id = ME.identifier;
                                                error_log.debug_mode = DEBUG_MODE;
                                                error_log.timestamp = datetime('now');

                                                try
                                                    selected_sd_node.parent.custom.put('srm_error', riekesuite.util.toJavaMap(error_log));
                                                catch
                                                    fprintf('            Failed to store error log\n');
                                                end
                                            end

                                        else
                                            fprintf('            ✗ QUALITY CHECK FAILED for freq %s: %s\n', ...
                                                freq_str, strjoin(quality_warnings, ', '));

                                            % DEBUG LOG: Record quality failure
                                            if DEBUG_MODE
                                                debugLog('ERROR', 'QUALITY', cell_name, freq_str, sd_str, ...
                                                    'Quality check failed', ...
                                                    'warnings', strjoin(quality_warnings, ', '), ...
                                                    'filter_correlation', filter_corr, ...
                                                    'threshold', quality_thresholds.min_filter_correlation);
                                            end

                                            % Standard quality failure logging
                                            quality_log = struct();
                                            quality_log.cell_id = cell_name;
                                            quality_log.frequency = freq_str;
                                            quality_log.selected_sd = sd_str;
                                            quality_log.quality_warnings = quality_warnings;
                                            quality_log.n_spikes = n_spikes;
                                            quality_log.firing_rate = firing_rate;
                                            quality_log.filter_corr = filter_corr;
                                            quality_log.duration = duration;
                                            quality_log.debug_mode = DEBUG_MODE;
                                            quality_log.timestamp = datetime('now');
                                            quality_log.skip_reason = 'quality_check_failed';

                                            try
                                                selected_sd_node.parent.custom.put('srm_quality_fail', riekesuite.util.toJavaMap(quality_log));
                                            catch
                                                fprintf('            Failed to store quality failure log\n');
                                            end
                                        end

                                    catch ME_quality
                                        fprintf('            ✗ ERROR reading stored results for freq %s: %s\n', ...
                                            freq_str, ME_quality.message);

                                        % DEBUG LOG: Record data extraction error
                                        if DEBUG_MODE
                                            debugLog('ERROR', 'VARIABLE', cell_name, freq_str, sd_str, ...
                                                'Failed to read stored results', ...
                                                'error_message', ME_quality.message, ...
                                                'error_id', ME_quality.identifier);
                                        end

                                        % Standard data error logging
                                        data_error_log = struct();
                                        data_error_log.cell_id = cell_name;
                                        data_error_log.frequency = freq_str;
                                        data_error_log.selected_sd = sd_str;
                                        data_error_log.error_message = ME_quality.message;
                                        data_error_log.error_id = ME_quality.identifier;
                                        data_error_log.debug_mode = DEBUG_MODE;
                                        data_error_log.timestamp = datetime('now');
                                        data_error_log.skip_reason = 'data_extraction_failed';

                                        try
                                            selected_sd_node.parent.custom.put('srm_data_error', riekesuite.util.toJavaMap(data_error_log));
                                        catch
                                            fprintf('            Failed to store data error log\n');
                                        end
                                    end
                                else
                                    fprintf('          No stored results found for freq %s, SD %s\n', freq_value, sd_value);

                                    % DEBUG LOG: Record missing stored results
                                    if DEBUG_MODE
                                        debugLog('WARN', 'VARIABLE', cell_name, freq_str, sd_str, ...
                                            'No stored results found', 'containsKey_results', false);
                                    end

                                    % Standard missing results logging
                                    missing_log = struct();
                                    missing_log.cell_id = cell_name;
                                    missing_log.frequency = freq_str;
                                    missing_log.selected_sd = sd_str;
                                    missing_log.debug_mode = DEBUG_MODE;
                                    missing_log.timestamp = datetime('now');
                                    missing_log.skip_reason = 'no_stored_results';

                                    try
                                        selected_sd_node.parent.custom.put('srm_missing_data', riekesuite.util.toJavaMap(missing_log));
                                    catch
                                        fprintf('            Failed to store missing data log\n');
                                    end
                                end
                            else
                                fprintf('          No stored results container found for freq %s, SD %s\n', freq_value, sd_value);

                                % DEBUG LOG: Record missing results container
                                if DEBUG_MODE
                                    debugLog('ERROR', 'INDEXING', cell_name, freq_str, sd_str, ...
                                        'SD node custom results container not found', ...
                                        'containsKey_check', 'failed');
                                end
                            end

                        else
                            fprintf('          No epochs found in any SD level for freq %s\n', freq_value);

                            % DEBUG LOG: Record no epochs found
                            if DEBUG_MODE
                                debugLog('WARN', 'INDEXING', cell_name, freq_str, 'N/A', ...
                                    'No epochs found in any SD level', ...
                                    'max_epochs', max_epochs, ...
                                    'total_sd_levels', length(sd_values));
                            end

                            % Standard no epochs logging
                            no_epochs_log = struct();
                            no_epochs_log.cell_id = cell_name;
                            no_epochs_log.frequency = freq_str;
                            no_epochs_log.debug_mode = DEBUG_MODE;
                            no_epochs_log.timestamp = datetime('now');
                            no_epochs_log.skip_reason = 'no_epochs_found';
                            no_epochs_log.total_sd_levels = length(sd_values);

                            try
                                freqNode.custom.put('srm_no_epochs', riekesuite.util.toJavaMap(no_epochs_log));
                            catch
                                fprintf('            Failed to store no epochs log\n');
                            end
                        end
                    else
                        fprintf('          No Current SD children found for freq %s\n', freq_value);

                        % DEBUG LOG: Record no SD children
                        if DEBUG_MODE
                            debugLog('WARN', 'INDEXING', cell_name, freq_str, 'N/A', ...
                                'No SD children found for frequency', ...
                                'freqNode_children_length', freqNode.children.length);
                        end

                        % Standard no SD children logging
                        no_sd_log = struct();
                        no_sd_log.cell_id = cell_name;
                        no_sd_log.frequency = freq_str;
                        no_sd_log.debug_mode = DEBUG_MODE;
                        no_sd_log.timestamp = datetime('now');
                        no_sd_log.skip_reason = 'no_sd_children';

                        try
                            freqNode.custom.put('srm_no_sd', riekesuite.util.toJavaMap(no_sd_log));
                        catch
                            fprintf('            Failed to store no SD log\n');
                        end
                    end
                end      % End freq_idx loop
            end          % End group_idx loop
        end              % End cell_idx loop
    end                  % End date_idx loop
end                      % End protocol_idx loop

% ===================================================================
%  FINAL SUMMARY AND DEBUG LOG ANALYSIS
% ===================================================================

fprintf('\n===============================================\n');
if DEBUG_MODE
    fprintf('SRM ANALYSIS COMPLETE (DEBUG MODE)\n');
else
    fprintf('SRM ANALYSIS COMPLETE (FULL MODE)\n');
end
fprintf('===============================================\n');

% Count successful analyses
n_successful = length(srm_results);
fprintf('Successful SRM analyses: %d\n', n_successful);

if n_successful > 0
    % Extract debug mode info
    debug_modes = [srm_results{:}];
    n_debug = sum([debug_modes.debug_mode]);
    n_full = n_successful - n_debug;

    fprintf('  - Debug mode: %d\n', n_debug);
    fprintf('  - Full mode: %d\n', n_full);

    % Show sample results
    fprintf('\nSample results:\n');
    for i = 1:min(5, n_successful)
        result = srm_results{i};
        mode_str = '';
        if result.debug_mode
            mode_str = ' (DEBUG)';
        end
        fprintf('  %s | freq=%s | VP=%.4f→%.4f/s%s\n', ...
            result.cell_id, result.frequency, ...
            result.best_train_vp, result.best_holdout_vp, mode_str);
    end

    if n_successful > 5
        fprintf('  ... and %d more\n', n_successful - 5);
    end
else
    fprintf('No successful analyses completed.\n');
end

% Display debug log summary if in debug mode
if DEBUG_MODE
    fprintf('\n=== DEBUG LOG ANALYSIS ===\n');
    displayDebugSummary();

    fprintf('\n=== DEBUG LOG COMMANDS ===\n');
    fprintf('Available debug functions:\n');
    fprintf('  DebugUtils.exportDebugLogForSharing() - Full export with display\n');
    fprintf('  displayDebugSummary()      - Show summary statistics\n');
    fprintf('  DebugUtils.clearDebugLog()            - Reset log for next run\n');
end

% Clear debug flag for next run
clear DEBUG_MODE;
fprintf('\nAnalysis loop complete. Results stored at frequency node level.\n');
fprintf('\n=== DEBUG LOG ANALYSIS ===\n');
displayDebugSummary();

fprintf('\n=== DEBUG LOG COMMANDS ===\n');
fprintf('Available debug functions:\n');
fprintf('  DebugUtils.exportDebugLogForSharing() - Full export with display\n');
fprintf('  displayDebugSummary()      - Show summary statistics\n');
fprintf('  DebugUtils.clearDebugLog()            - Reset log for next run\n');


% Clear debug flag for next run
clear DEBUG_MODE;
fprintf('\nAnalysis loop complete. Results stored at frequency node level.\n');
fprintf('\nAnalysis loop complete. Results stored at frequency node level.\n');
fprintf('\nAnalysis loop complete. Results stored at frequency node level.\n');
% Print summary showing frequency-specific results
fprintf('\n=== SRM ANALYSIS SUMMARY (FREQUENCY-SPECIFIC) ===\n');
fprintf('Total successful analyses: %d\n', length(srm_results));
if ~isempty(srm_results)
    for i = 1:length(srm_results)
        r = srm_results{i};
        fprintf('  %s | %s | freq=%s | SD=%s\n', ...
            r.cell_id, r.protocol, r.frequency, r.selected_sd);
    end
end
% Print summary showing frequency-specific results
fprintf('\n=== SRM ANALYSIS SUMMARY (FREQUENCY-SPECIFIC) ===\n');
fprintf('Total successful analyses: %d\n', length(srm_results));
if ~isempty(srm_results)
    for i = 1:length(srm_results)
        r = srm_results{i};
        fprintf('  %s | %s | freq=%s | SD=%s\n', ...
            r.cell_id, r.protocol, r.frequency, r.selected_sd);
    end
end
% -----------------------------------------------------------------------
%  SRM ANALYSIS SUMMARY
% -----------------------------------------------------------------------

fprintf('\n=== SRM ANALYSIS SUMMARY ===\n');
fprintf('Total SRM analyses attempted: %d\n', length(srm_results) + length(srm_failed));
fprintf('Successful SRM analyses: %d\n', length(srm_results));
fprintf('Failed SRM analyses: %d\n', length(srm_failed));

if ~isempty(srm_results)
    % Extract summary statistics
    train_vps = cellfun(@(x) x.best_train_vp, srm_results);
    holdout_vps = cellfun(@(x) x.best_holdout_vp, srm_results);
    gen_ratios = cellfun(@(x) x.best_generalization_ratio, srm_results);
    rate_accs = cellfun(@(x) x.rate_accuracy_pct, srm_results);

    % Count kernel types
    kernel_types = cellfun(@(x) x.best_kernel, srm_results, 'UniformOutput', false);
    exp_count = sum(strcmp(kernel_types, 'exponential'));
    linexp_count = sum(strcmp(kernel_types, 'linear_rise_exp_decay'));

    fprintf('\nSRM Summary Statistics:\n');
    fprintf('  Training VP: %.4f ± %.4f (range: %.4f - %.4f) /s\n', ...
        mean(train_vps), std(train_vps), min(train_vps), max(train_vps));
    fprintf('  Hold-out VP: %.4f ± %.4f (range: %.4f - %.4f) /s\n', ...
        mean(holdout_vps), std(holdout_vps), min(holdout_vps), max(holdout_vps));
    fprintf('  Generalization ratio: %.3f ± %.3f (range: %.3f - %.3f)\n', ...
        mean(gen_ratios), std(gen_ratios), min(gen_ratios), max(gen_ratios));
    fprintf('  Rate accuracy: %.1f ± %.1f%% (range: %.1f - %.1f%%)\n', ...
        mean(rate_accs), std(rate_accs), min(rate_accs), max(rate_accs));

    fprintf('\nKernel Selection:\n');
    fprintf('  Exponential: %d (%.1f%%)\n', exp_count, 100*exp_count/length(srm_results));
    fprintf('  Linear+Exp: %d (%.1f%%)\n', linexp_count, 100*linexp_count/length(srm_results));

    fprintf('\nSuccessful SRM Analyses:\n');
    for i = 1:length(srm_results)
        res = srm_results{i};
        fprintf('  %d. %s | %s | Freq:%s | SD:%s | %s | VP:%.4f→%.4f/s | Acc:%.1f%%\n', ...
            i, res.cell_id, res.protocol, num2str(res.frequency), num2str(res.selected_sd), ...
            res.best_kernel(1:3), res.best_train_vp, res.best_holdout_vp, res.rate_accuracy_pct);
    end
end

if ~isempty(srm_failed)
    fprintf('\nFailed SRM Analyses:\n');
    for i = 1:length(srm_failed)
        fail = srm_failed{i};
        fprintf('  %d. %s | %s | Freq:%s | SD:%s | Error: %s\n', ...
            i, fail.cell_id, fail.protocol, fail.frequency, fail.selected_sd, fail.error_message);
    end
end

% Should be:
fprintf('SRM results stored at frequency level nodes with key "srm_results"\n');
fprintf('Query using: stored = freqNode.custom.get(''srm_results''); srm_data = get(stored, ''best_kernel'');\n');
%%
% ===================================================================
% section thress NAVIGATION LOOP TO FREQUENCY NODES
% ===================================================================

fprintf('=== NAVIGATING TO FREQUENCY NODES ===\n');

% Initialize collection for frequency node data
freq_node_data = {};
data_counter = 1;

% Navigate through the tree hierarchy to reach frequency nodes
for protocol_idx = 1:CurrentNode.children.length
    protocolNode = CurrentNode.children.elements(protocol_idx);
    protocol_name = protocolNode.splitValue;
    fprintf('\nProtocol: %s\n', protocol_name);

    for date_idx = 1:protocolNode.children.length
        dateNode = protocolNode.children.elements(date_idx);
        date_name = dateNode.splitValue;
        fprintf('  Date: %s\n', date_name);

        for cell_idx = 1:dateNode.children.length
            cellNode = dateNode.children.elements(cell_idx);
            cell_name = cellNode.splitValue;
            fprintf('    Cell: %s\n', cell_name);

            % Level 4: Epoch Group
            for group_idx = 1:cellNode.children.length
                groupNode = cellNode.children.elements(group_idx);
                group_name = groupNode.splitValue;
                fprintf('      Group: %s\n', group_name);

                % Level 5: FREQUENCY NODES - This is our target level
                for freq_idx = 1:groupNode.children.length
                    freqNode = groupNode.children.elements(freq_idx);
                    freq_value = freqNode.splitValue;
                    freq_str = num2str(freq_value);

                    fprintf('        >>> REACHED FREQUENCY NODE: %s Hz <<<\n', freq_str);

                    % ===================================================================
                    %  DO YOUR ANALYSIS/QUERIES ON FREQUENCY NODE HERE
                    % ===================================================================

                    % Example: Check for stored SRM results
                    if freqNode.custom.containsKey('srm_results')
                        stored_srm = freqNode.custom.get('srm_results');

                        if ~isempty(stored_srm)
                            fprintf('          Found SRM results for freq %s\n', freq_str);

                            % Extract SRM data
                            best_kernel = get(stored_srm, 'best_kernel');
                            vp_holdout = get(stored_srm, 'best_holdout_vp');  % This is already normalized
                            vp_train = get(stored_srm, 'best_train_vp');      % This is already normalized
                            n_spikes = get(stored_srm, 'n_spikes_total');
                            duration = get(stored_srm, 'total_duration_s');
                            selected_sd = get(stored_srm, 'selected_sd');

                            fprintf('          SRM: %s kernel, VP=%.4f→%.4f/s, %d spikes, %.1fs\n', ...
                                best_kernel, vp_train, vp_holdout, n_spikes, duration);

                            % Store frequency node data
                            freq_data = struct();
                            freq_data.protocol = protocol_name;
                            freq_data.date = date_name;
                            freq_data.cell_id = cell_name;
                            freq_data.group = group_name;
                            freq_data.frequency = freq_str;
                            freq_data.freq_node = freqNode;  % Store the actual node reference
                            freq_data.best_kernel = best_kernel;
                            freq_data.vp_holdout = vp_holdout;
                            freq_data.vp_train = vp_train;
                            freq_data.n_spikes = n_spikes;
                            freq_data.duration = duration;
                            freq_data.selected_sd = selected_sd;
                            freq_data.has_srm_results = true;

                            freq_node_data{data_counter} = freq_data;
                            data_counter = data_counter + 1;

                        else
                            fprintf('          SRM results container empty for freq %s\n', freq_str);
                        end
                    else
                        fprintf('          No SRM results found for freq %s\n', freq_str);

                        % Still store basic info even without SRM results
                        freq_data = struct();
                        freq_data.protocol = protocol_name;
                        freq_data.date = date_name;
                        freq_data.cell_id = cell_name;
                        freq_data.group = group_name;
                        freq_data.frequency = freq_str;
                        freq_data.freq_node = freqNode;
                        freq_data.has_srm_results = false;

                        freq_node_data{data_counter} = freq_data;
                        data_counter = data_counter + 1;
                    end

                    % Example: Check for other stored data types
                    if freqNode.custom.containsKey('srm_error')
                        fprintf('          Found SRM error log for freq %s\n', freq_str);
                    end

                    if freqNode.custom.containsKey('srm_quality_fail')
                        fprintf('          Found quality failure log for freq %s\n', freq_str);
                    end

                end  % End frequency loop
                % ===================================================================
                % ===================================================================
                % KERNEL PARAMETER PLOTTING - Add this in your cell loop
                % ===================================================================

                % After the cell loop ends (after "end % End cell loop"), add this:

                % At the end of each CELL (not frequency), collect all frequencies and plot
                % This should be added after the "end % End frequency loop" but before "end % End group loop"

                % ===================================================================
                %  PLOT KERNEL PARAMETERS FOR THIS CELL ACROSS ALL FREQUENCIES
                % ===================================================================

                % Collect kernel data for this specific cell across all frequencies
                cell_kernel_data = [];
                cell_frequencies = [];
                cell_kernels = {};
                cell_vp_values = [];

                fprintf('      >>> Collecting kernel parameters for cell %s <<<\n', cell_name);

                % Loop through all frequencies for this cell to collect kernel data
                for freq_idx = 1:groupNode.children.length
                    freqNode = groupNode.children.elements(freq_idx);
                    freq_value = freqNode.splitValue;
                    freq_str = num2str(freq_value);

                    if freqNode.custom.containsKey('srm_results')
                        stored_srm = freqNode.custom.get('srm_results');

                        if ~isempty(stored_srm)
                            % Extract kernel parameters
                            best_kernel = get(stored_srm, 'best_kernel');
                            kernel_params = get(stored_srm, 'best_kernel_params');
                            vp_holdout = get(stored_srm, 'best_holdout_vp');

                            % Store data for plotting
                            cell_frequencies(end+1) = freq_value;
                            cell_kernels{end+1} = best_kernel;
                            cell_vp_values(end+1) = vp_holdout;

                            if strcmp(best_kernel, 'exponential')
                                % [A, tau] for exponential
                                A = kernel_params(1);
                                tau = kernel_params(2);
                                cell_kernel_data(end+1, :) = [A, tau, 0, 0]; % [A, tau, T_rise, tau_decay]
                                fprintf('        Freq %s Hz: %s, A=%.3f, tau=%.3f\n', freq_str, best_kernel, A, tau);

                            elseif strcmp(best_kernel, 'linear_rise_exp_decay')
                                % [A, T_rise, tau_decay] for linear+exp
                                A = kernel_params(1);
                                T_rise = kernel_params(2);
                                tau_decay = kernel_params(3);
                                cell_kernel_data(end+1, :) = [A, 0, T_rise, tau_decay]; % [A, tau, T_rise, tau_decay]
                                fprintf('        Freq %s Hz: %s, A=%.3f, T_rise=%.3f, tau_decay=%.3f\n', ...
                                    freq_str, best_kernel, A, T_rise, tau_decay);
                            end
                        end
                    end
                end

                % Only plot if we have data for this cell
                if ~isempty(cell_kernel_data) && length(cell_frequencies) >= 2

                    fprintf('      >>> Creating kernel parameter plots for %s <<<\n', cell_name);

                    % Create figure for this cell
                    figure('Position', [100, 100, 1400, 800]);
                    sgtitle(sprintf('Kernel Parameters vs Frequency - %s (%s)', cell_name, group_name), 'FontSize', 14);

                    % Colors for different kernel types
                    exp_color = [0.2, 0.4, 0.8];      % Blue for exponential
                    linexp_color = [0.8, 0.2, 0.4];   % Red for linear+exp

                    % Separate data by kernel type
                    exp_mask = strcmp(cell_kernels, 'exponential');
                    linexp_mask = strcmp(cell_kernels, 'linear_rise_exp_decay');

                    exp_freqs = cell_frequencies(exp_mask);
                    linexp_freqs = cell_frequencies(linexp_mask);
                    exp_data = cell_kernel_data(exp_mask, :);
                    linexp_data = cell_kernel_data(linexp_mask, :);
                    exp_vp = cell_vp_values(exp_mask);
                    linexp_vp = cell_vp_values(linexp_mask);

                    % Subplot 1: Amplitude (A) vs Frequency
                    subplot(2, 3, 1);
                    hold on;
                    if any(exp_mask)
                        % Create color values for VP mapping
                        vp_normalized = (exp_vp - min(cell_vp_values)) / (max(cell_vp_values) - min(cell_vp_values));
                        for i = 1:length(exp_freqs)
                            color_intensity = vp_normalized(i);
                            marker_color = [0.2, 0.4, 0.8] * (1 - color_intensity) + [1, 1, 1] * color_intensity;
                            scatter(exp_freqs(i), exp_data(i, 1), 80, 'o', 'filled', ...
                                'MarkerFaceColor', marker_color, 'MarkerEdgeColor', exp_color);
                        end
                        plot(exp_freqs, exp_data(:, 1), '--', 'Color', exp_color, 'LineWidth', 1.5);
                    end
                    if any(linexp_mask)
                        vp_normalized = (linexp_vp - min(cell_vp_values)) / (max(cell_vp_values) - min(cell_vp_values));
                        for i = 1:length(linexp_freqs)
                            color_intensity = vp_normalized(i);
                            marker_color = [0.8, 0.2, 0.4] * (1 - color_intensity) + [1, 1, 1] * color_intensity;
                            scatter(linexp_freqs(i), linexp_data(i, 1), 80, 's', 'filled', ...
                                'MarkerFaceColor', marker_color, 'MarkerEdgeColor', linexp_color);
                        end
                        plot(linexp_freqs, linexp_data(:, 1), '--', 'Color', linexp_color, 'LineWidth', 1.5);
                    end
                    xlabel('Frequency (Hz)');
                    ylabel('Amplitude A (mV)');
                    title('Kernel Amplitude');
                    grid on;
                    set(gca, 'XScale', 'log');

                    % Subplot 2: Time constants vs Frequency
                    subplot(2, 3, 2);
                    hold on;
                    if any(exp_mask)
                        for i = 1:length(exp_freqs)
                            scatter(exp_freqs(i), exp_data(i, 2) * 1000, 80, 'o', 'filled', ...
                                'MarkerFaceColor', exp_color, 'MarkerEdgeColor', 'black');
                        end
                        plot(exp_freqs, exp_data(:, 2) * 1000, '--', 'Color', exp_color, 'LineWidth', 1.5);
                    end
                    if any(linexp_mask)
                        for i = 1:length(linexp_freqs)
                            scatter(linexp_freqs(i), linexp_data(i, 4) * 1000, 80, 's', 'filled', ...
                                'MarkerFaceColor', linexp_color, 'MarkerEdgeColor', 'black');
                        end
                        plot(linexp_freqs, linexp_data(:, 4) * 1000, '--', 'Color', linexp_color, 'LineWidth', 1.5);
                    end
                    xlabel('Frequency (Hz)');
                    ylabel('Time Constant (ms)');
                    title('Decay Time Constants');
                    grid on;
                    set(gca, 'XScale', 'log');

                    % Subplot 3: Rise time (for linear+exp only)
                    subplot(2, 3, 3);
                    if any(linexp_mask)
                        for i = 1:length(linexp_freqs)
                            scatter(linexp_freqs(i), linexp_data(i, 3) * 1000, 80, 's', 'filled', ...
                                'MarkerFaceColor', linexp_color, 'MarkerEdgeColor', 'black');
                        end
                        plot(linexp_freqs, linexp_data(:, 3) * 1000, '--', 'Color', linexp_color, 'LineWidth', 1.5);
                        xlabel('Frequency (Hz)');
                        ylabel('Rise Time (ms)');
                        title('Rise Time (Linear+Exp only)');
                        grid on;
                        set(gca, 'XScale', 'log');
                    else
                        text(0.5, 0.5, 'No Linear+Exp kernels', 'HorizontalAlignment', 'center', ...
                            'VerticalAlignment', 'middle', 'Units', 'normalized', 'FontSize', 12);
                        title('Rise Time (Linear+Exp only)');
                    end

                    % Subplot 4: VP Performance vs Frequency
                    subplot(2, 3, 4);
                    hold on;
                    if any(exp_mask)
                        for i = 1:length(exp_freqs)
                            scatter(exp_freqs(i), exp_vp(i), 80, 'o', 'filled', ...
                                'MarkerFaceColor', exp_color, 'MarkerEdgeColor', 'black');
                        end
                        plot(exp_freqs, exp_vp, '--', 'Color', exp_color, 'LineWidth', 1.5);
                    end
                    if any(linexp_mask)
                        for i = 1:length(linexp_freqs)
                            scatter(linexp_freqs(i), linexp_vp(i), 80, 's', 'filled', ...
                                'MarkerFaceColor', linexp_color, 'MarkerEdgeColor', 'black');
                        end
                        plot(linexp_freqs, linexp_vp, '--', 'Color', linexp_color, 'LineWidth', 1.5);
                    end
                    xlabel('Frequency (Hz)');
                    ylabel('VP Loss (Hold-out) (/s)');
                    title('Model Performance (Normalized)');
                    grid on;
                    set(gca, 'XScale', 'log');

                    % Subplot 5: Kernel type distribution
                    subplot(2, 3, 5);
                    kernel_counts = [sum(exp_mask), sum(linexp_mask)];
                    kernel_labels = {'Exponential', 'Linear+Exp'};
                    pie(kernel_counts, kernel_labels);
                    title('Kernel Type Distribution');

                    % Subplot 6: All 4 frequency kernels overlaid
                    subplot(2, 3, 6);
                    hold on;
                    t_kernel = 0:0.001:0.1; % 0-100ms

                    % Use all 4 frequencies if available, otherwise use what we have
                    plot_freqs = cell_frequencies; % Use whatever frequencies we have

                    % Create distinct colors for each frequency
                    colors = lines(length(plot_freqs));

                    % Plot kernel for each frequency
                    for i = 1:length(plot_freqs)
                        freq = plot_freqs(i);
                        freq_idx = find(cell_frequencies == freq, 1);

                        if ~isempty(freq_idx)
                            kernel_type = cell_kernels{freq_idx};
                            params = cell_kernel_data(freq_idx, :);
                            vp_val = cell_vp_values(freq_idx);

                            if strcmp(kernel_type, 'exponential')
                                A = params(1);
                                tau = params(2);
                                kernel_vals = A * exp(-t_kernel / tau);
                                line_style = '-';  % Solid line for exponential
                                marker_style = 'o';
                            else % linear_rise_exp_decay
                                A = params(1);
                                T_rise = params(3);
                                tau_decay = params(4);
                                kernel_vals = (t_kernel < T_rise) .* (A / T_rise .* t_kernel) + ...
                                    (t_kernel >= T_rise) .* (A * exp(-(t_kernel - T_rise) / tau_decay));
                                line_style = '--'; % Dashed line for linear+exp
                                marker_style = 's';
                            end

                            % Plot the kernel
                            plot(t_kernel * 1000, kernel_vals, 'Color', colors(i, :), ...
                                'LineWidth', 2.5, 'LineStyle', line_style);

                            % Add a marker at the peak for identification
                            [peak_val, peak_idx] = max(kernel_vals);
                            plot(t_kernel(peak_idx) * 1000, peak_val, marker_style, ...
                                'Color', colors(i, :), 'MarkerSize', 8, 'MarkerFaceColor', colors(i, :));

                            fprintf('        Plotted %s Hz kernel: %s, peak=%.2f mV at %.1f ms\n', ...
                                num2str(freq), kernel_type, peak_val, t_kernel(peak_idx)*1000);
                        end
                    end

                    xlabel('Time (ms)');
                    ylabel('Threshold Change (mV)');
                    title(sprintf('All Frequency Kernels (%d total)', length(plot_freqs)));

                    % Create detailed legend
                    legend_labels = {};
                    for i = 1:length(plot_freqs)
                        freq = plot_freqs(i);
                        freq_idx = find(cell_frequencies == freq, 1);
                        if ~isempty(freq_idx)
                            kernel_type = cell_kernels{freq_idx};
                            vp_val = cell_vp_values(freq_idx);

                            if strcmp(kernel_type, 'exponential')
                                type_abbrev = 'Exp';
                            else
                                type_abbrev = 'LinExp';
                            end

                            legend_labels{i} = sprintf('%.0f Hz (%s, VP=%.2f/s)', freq, type_abbrev, vp_val);
                        end
                    end

                    legend(legend_labels, 'Location', 'best', 'FontSize', 8);
                    grid on;

                    % Add text box with kernel summary
                    kernel_summary = sprintf('Solid = Exponential\nDashed = Linear+Exp\nMarkers show peaks');
                    text(0.02, 0.98, kernel_summary, 'Units', 'normalized', ...
                        'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', ...
                        'BackgroundColor', 'white', 'EdgeColor', 'black', 'FontSize', 8);

                    % Add legend for kernel types
                    legend_elements = {};
                    if any(exp_mask)
                        legend_elements{end+1} = 'Exponential (○)';
                    end
                    if any(linexp_mask)
                        legend_elements{end+1} = 'Linear+Exp (□)';
                    end

                    % Save figure as PNG with high resolution
                    cellTypeNode = groupNode.parent.parent.parent; % Navigate to cell type level
                    cell_type_raw = cellTypeNode.splitValue;
                    cell_type_label = strrep(cell_type_raw, '\', '/');

                    base_path = '/Users/maxwellsdm/Library/CloudStorage/GoogleDrive-maxwellsdm1867@gmail.com/.shortcut-targets-by-id/1CbZnfugdi-p4jM3fe7t-NyGk5v2ZcPdk/ParasolCenterSurround/figures/ScienceJuiceFactory/currinjt/overview/';
                    fig_path = [base_path cell_type_label '/'];

                    if ~exist(fig_path, 'dir')
                        mkdir(fig_path);
                        fprintf('      Created directory: %s\n', fig_path);
                    end

                    % Use date from current analysis
                    date_str = date_name(1:11);
                    cell_name_lower = lower(cell_name);
                    kernel_filename = sprintf('%s-%s-kernel-parameters.png', date_str, cell_name_lower);
                    kernel_save_path = [fig_path kernel_filename];

                    % Save as PNG with high resolution (300 DPI)
                    print(gcf, kernel_save_path, '-dpng', '-r300');
                    fprintf('      >>> PNG saved: %s <<<\n', kernel_filename);
                    fprintf('      Full path: %s\n', kernel_save_path);

                    % Close figure to save memory
                    close(gcf);

                else
                    fprintf('      Insufficient data for kernel plotting (need ≥2 frequencies with results)\n');
                end

                % Clear temporary variables
                clear cell_kernel_data cell_frequencies cell_kernels cell_vp_values;
                clear exp_mask linexp_mask exp_freqs linexp_freqs exp_data linexp_data exp_vp linexp_vp;
            end      % End group loop
        end          % End cell loop
    end              % End date loop
end                  % End protocol loop

% ===================================================================
%  SUMMARY OF NAVIGATION RESULTS
% ===================================================================

fprintf('\n=== FREQUENCY NODE NAVIGATION SUMMARY ===\n');
fprintf('Total frequency nodes found: %d\n', length(freq_node_data));

% Count nodes with SRM results
nodes_with_srm = sum(cellfun(@(x) x.has_srm_results, freq_node_data));
fprintf('Nodes with SRM results: %d\n', nodes_with_srm);
fprintf('Nodes without SRM results: %d\n', length(freq_node_data) - nodes_with_srm);

% Show sample of collected data
if ~isempty(freq_node_data)
    fprintf('\nSample frequency node data:\n');
    for i = 1:min(5, length(freq_node_data))
        data = freq_node_data{i};
        if data.has_srm_results
            fprintf('  %s | %s Hz | %s kernel | VP=%.3f\n', ...
                data.cell_id, data.frequency, data.best_kernel, data.vp_holdout);
        else
            fprintf('  %s | %s Hz | No SRM results\n', data.cell_id, data.frequency);
        end
    end

    if length(freq_node_data) > 5
        fprintf('  ... and %d more\n', length(freq_node_data) - 5);
    end
end

fprintf('\nFrequency node navigation complete.\n');
fprintf('Access collected data using: freq_node_data{index}\n');
fprintf('Access specific node using: freq_node_data{index}.freq_node\n');

% Redefine DEBUG_MODE for section 4
DEBUG_MODE = false;  % Set to false for full analysis

%%
% ===================================================================
% section four NAVIGATION LOOP TO FREQUENCY NODES - Vm_all EXTRACTION
% ===================================================================

fprintf('\n=== SECTION 4: Vm_all DATA EXTRACTION ===\n');

% Initialize collection for Vm_all data
vm_all_data = {};
vm_data_counter = 1;
DEBUG_MODE = false;
% Navigate through the tree hierarchy to reach frequency nodes
for protocol_idx = 1:CurrentNode.children.length
    protocolNode = CurrentNode.children.elements(protocol_idx);
    protocol_name = protocolNode.splitValue;
    fprintf('\nProtocol: %s\n', protocol_name);

    for date_idx = 1:protocolNode.children.length
        dateNode = protocolNode.children.elements(date_idx);
        date_name = dateNode.splitValue;
        fprintf('  Date: %s\n', date_name);

        for cell_idx = 1:dateNode.children.length
            cellNode = dateNode.children.elements(cell_idx);
            cell_name = cellNode.splitValue;
            fprintf('    Cell: %s\n', cell_name);

            % Level 4: Epoch Group
            for group_idx = 1:cellNode.children.length
                groupNode = cellNode.children.elements(group_idx);
                group_name = groupNode.splitValue;
                fprintf('      Group: %s\n', group_name);

                % Level 5: FREQUENCY NODES - This is our target level
                for freq_idx = 1:groupNode.children.length
                    freqNode = groupNode.children.elements(freq_idx);
                    freq_value = freqNode.splitValue;
                    freq_str = num2str(freq_value);

                    fprintf('        >>> REACHED FREQUENCY NODE: %s Hz <<<\n', freq_str);

                    % ===================================================================
                    %  EXTRACT Vm_all DATA FROM FREQUENCY NODE
                    % ===================================================================

                    % Check for stored results at frequency node level
                    if freqNode.custom.containsKey('results')
                        stored_results = freqNode.custom.get('results');

                        if ~isempty(stored_results)
                            fprintf('          Found stored results for freq %s\n', freq_str);

                            % Extract Vm_all data
                            try
                                Vm_all = get(stored_results, 'Vm_all');
                                Vm_cleaned = get(stored_results, 'Vm_cleaned');
                                dt = get(stored_results, 'dt');
                                n_spikes = get(stored_results, 'n_spikes');
                                firing_rate = get(stored_results, 'firing_rate_Hz');
                                duration = get(stored_results, 'total_duration_s');
                                filter_corr = get(stored_results, 'filter_correlation');

                                fprintf('          Vm_all: %d samples, Duration: %.1fs, Spikes: %d, Rate: %.1f Hz\n', ...
                                    length(Vm_all), duration, n_spikes, firing_rate);

                                % Store Vm_all data
                                vm_data = struct();
                                vm_data.protocol = protocol_name;
                                vm_data.date = date_name;
                                vm_data.cell_id = cell_name;
                                vm_data.group = group_name;
                                vm_data.frequency = freq_str;
                                vm_data.freq_node = freqNode;  % Store the actual node reference
                                vm_data.Vm_all = Vm_all;
                                vm_data.Vm_cleaned = Vm_cleaned;
                                vm_data.dt = dt;
                                vm_data.n_spikes = n_spikes;
                                vm_data.firing_rate_Hz = firing_rate;
                                vm_data.duration_s = duration;
                                vm_data.filter_correlation = filter_corr;
                                vm_data.n_samples = length(Vm_all);
                                vm_data.has_vm_data = true;

                                % Run spike detection using elbow method
                                fprintf('          Running elbow spike detection...\n');
                                try
                                    % Use the detect_spike_initiation_elbow_v2 function with spike_params
                                    vm_thresh = spike_params.vm_thresh;     % -20 mV
                                    d2v_thresh = spike_params.d2v_thresh;   % 50
                                    search_back_ms = spike_params.search_back_ms; % 2 ms
                                    plot_flag = spike_params.plot_flag;     % false for batch processing

                                    % Detect spike initiation points using enhanced elbow method v2
                                    [spike_indices, spike_peaks, isi, avg_spike, diagnostic_info] = ...
                                        detect_spike_initiation_elbow_v2(Vm_all, dt, vm_thresh, d2v_thresh, search_back_ms, plot_flag, ...
                                        'elbow_thresh', -65, 'spike_thresh', -10, 'min_dv_thresh', 0.1, ...
                                        'time_to_peak_thresh', 1.5);

                                    % Calculate spike statistics
                                    n_spikes_elbow = length(spike_indices);
                                    firing_rate_elbow = n_spikes_elbow / duration;

                                    % Convert spike indices to times
                                    true_spike_times_ms = spike_indices * dt * 1000; % Convert to ms
                                    true_spike_times = spike_indices * dt; % Keep in seconds

                                    % Get threshold from diagnostic info if available
                                    if isfield(diagnostic_info, 'elbow_threshold')
                                        elbow_threshold = diagnostic_info.elbow_threshold;
                                    else
                                        elbow_threshold = -55; % Default threshold
                                    end

                                    fprintf('          Elbow detection: %d spikes, %.1f Hz, threshold=%.1f mV\n', ...
                                        n_spikes_elbow, firing_rate_elbow, elbow_threshold);

                                    % Store elbow detection results
                                    vm_data.elbow_spike_indices = spike_indices;
                                    vm_data.elbow_spike_peaks = spike_peaks;
                                    vm_data.elbow_isi = isi;
                                    vm_data.elbow_avg_spike = avg_spike;
                                    vm_data.elbow_diagnostic_info = diagnostic_info;
                                    vm_data.elbow_threshold = elbow_threshold;
                                    vm_data.n_spikes_elbow = n_spikes_elbow;
                                    vm_data.firing_rate_elbow_Hz = firing_rate_elbow;
                                    vm_data.true_spike_times_ms = true_spike_times_ms;
                                    vm_data.true_spike_times = true_spike_times;
                                    vm_data.elbow_detection_success = true;

                                    % ===================================================================
                                    %  4-FACTOR SPIKE INITIATION ANALYSIS (FUNCTION CALL)
                                    % ===================================================================

                                    if vm_data.elbow_detection_success && n_spikes_elbow >= 5
                                        fprintf('          >>> Starting 4-factor spike initiation analysis <<<\n');

                                        % Prepare cell information structure
                                        cell_info = struct();
                                        cell_info.cell_name = cell_name;
                                        cell_info.freq_str = freq_str;
                                        cell_info.date_name = date_name;
                                        cell_info.protocol_name = protocol_name;

                                        % Create save path (same structure as other figures)
                                        cellTypeNode = freqNode.parent.parent.parent.parent.parent;
                                        cell_type_raw = cellTypeNode.splitValue;
                                        cell_type_label = strrep(cell_type_raw, '\', '/');

                                        base_path = '/Users/maxwellsdm/Library/CloudStorage/GoogleDrive-maxwellsdm1867@gmail.com/.shortcut-targets-by-id/1CbZnfugdi-p4jM3fe7t-NyGk5v2ZcPdk/ParasolCenterSurround/figures/ScienceJuiceFactory/currinjt/overview/';
                                        fig_path = [base_path cell_type_label '/'];

                                        if ~exist(fig_path, 'dir')
                                            mkdir(fig_path);
                                        end

                                        % Create filename (without extension - function adds .png)
                                        date_str = date_name(1:11);
                                        cell_name_lower = lower(cell_name);
                                        save_path_base = [fig_path sprintf('%s-%s-freq-%s-spike-initiation-4factor', date_str, cell_name_lower, freq_str)];

                                        % Run the 4-factor analysis using the function
                                        try
                                            analysis_results = analyzeSpikeInitiation4Factors(Vm_all, spike_indices, dt, cell_info, save_path_base, ...
                                                'PreSpikeWindowMs', 3, ...
                                                'SpikeCountWindowMs', 50, ...
                                                'MinSpikes', 5, ...
                                                'CreatePlot', true, ...
                                                'Verbose', true);

                                            if analysis_results.success
                                                % Store results in vm_data structure
                                                vm_data.spike_initiation_analysis = analysis_results;

                                                fprintf('          >>> 4-factor analysis SUCCESS: strongest correlation = %.3f <<<\n', ...
                                                    analysis_results.correlations.strongest_correlation);

                                                % Print correlation summary
                                                fprintf('          Correlations: Vm=%.3f, dV/dt=%.3f, ISI=%.3f, Count=%.3f\n', ...
                                                    analysis_results.correlations.vm_correlation, ...
                                                    analysis_results.correlations.dvdt_correlation, ...
                                                    analysis_results.correlations.isi_correlation, ...
                                                    analysis_results.correlations.count_correlation);
                                            else
                                                fprintf('          >>> 4-factor analysis FAILED: %s <<<\n', analysis_results.error_message);
                                                vm_data.spike_initiation_analysis_error = analysis_results.error_message;
                                            end

                                        catch ME
                                            fprintf('          >>> ERROR in 4-factor spike initiation analysis: %s <<<\n', ME.message);
                                            vm_data.spike_initiation_analysis_error = ME.message;
                                        end

                                    else
                                        fprintf('          Skipping 4-factor analysis: insufficient spikes or elbow detection failed\n');
                                    end

                                catch ME
                                    fprintf('          ERROR in elbow spike detection: %s\n', ME.message);

                                    % Store error info
                                    vm_data.elbow_detection_success = false;
                                    vm_data.elbow_error_message = ME.message;
                                    vm_data.n_spikes_elbow = 0;
                                    vm_data.firing_rate_elbow_Hz = 0;
                                end

                                vm_all_data{vm_data_counter} = vm_data;
                                vm_data_counter = vm_data_counter + 1;

                                % DEBUG LOG: Record successful Vm_all extraction
                                if DEBUG_MODE
                                    debugLog('SUCCESS', 'DATA', cell_name, freq_str, 'N/A', ...
                                        'Vm_all data extracted successfully', ...
                                        'n_samples', length(Vm_all), 'n_spikes', n_spikes, 'duration', duration);
                                end

                            catch ME
                                fprintf('          ERROR extracting Vm_all data: %s\n', ME.message);

                                % DEBUG LOG: Record extraction error
                                if DEBUG_MODE
                                    debugLog('ERROR', 'DATA', cell_name, freq_str, 'N/A', ...
                                        'Failed to extract Vm_all data', ...
                                        'error_message', ME.message);
                                end

                                % Store error info
                                vm_data = struct();
                                vm_data.protocol = protocol_name;
                                vm_data.date = date_name;
                                vm_data.cell_id = cell_name;
                                vm_data.group = group_name;
                                vm_data.frequency = freq_str;
                                vm_data.freq_node = freqNode;
                                vm_data.has_vm_data = false;
                                vm_data.error_message = ME.message;

                                vm_all_data{vm_data_counter} = vm_data;
                                vm_data_counter = vm_data_counter + 1;
                            end

                        else
                            fprintf('          Stored results container empty for freq %s\n', freq_str);

                            % Store empty data info
                            vm_data = struct();
                            vm_data.protocol = protocol_name;
                            vm_data.date = date_name;
                            vm_data.cell_id = cell_name;
                            vm_data.group = group_name;
                            vm_data.frequency = freq_str;
                            vm_data.freq_node = freqNode;
                            vm_data.has_vm_data = false;
                            vm_data.error_message = 'Empty results container';

                            vm_all_data{vm_data_counter} = vm_data;
                            vm_data_counter = vm_data_counter + 1;
                        end
                    else
                        fprintf('          No stored results found for freq %s\n', freq_str);

                        % Store missing data info
                        vm_data = struct();
                        vm_data.protocol = protocol_name;
                        vm_data.date = date_name;
                        vm_data.cell_id = cell_name;
                        vm_data.group = group_name;
                        vm_data.frequency = freq_str;
                        vm_data.freq_node = freqNode;
                        vm_data.has_vm_data = false;
                        vm_data.error_message = 'No stored results found';

                        vm_all_data{vm_data_counter} = vm_data;
                        vm_data_counter = vm_data_counter + 1;
                    end

                end  % End frequency loop
            end  % End group loop
        end  % End cell loop
    end  % End date loop
end  % End protocol loop
% ===================================================================
%  Vm_all DATA SUMMARY AND ANALYSIS
% ===================================================================

fprintf('\n=== Vm_all DATA EXTRACTION SUMMARY ===\n');
fprintf('Total frequency nodes processed: %d\n', length(vm_all_data));

% Count successful extractions
successful_vm = vm_all_data(cellfun(@(x) x.has_vm_data, vm_all_data));
failed_vm = vm_all_data(cellfun(@(x) ~x.has_vm_data, vm_all_data));

fprintf('Successful Vm_all extractions: %d\n', length(successful_vm));
fprintf('Failed Vm_all extractions: %d\n', length(failed_vm));

if ~isempty(successful_vm)
    % Extract summary statistics
    n_samples_all = cellfun(@(x) x.n_samples, successful_vm);
    firing_rates = cellfun(@(x) x.firing_rate_Hz, successful_vm);
    durations = cellfun(@(x) x.duration_s, successful_vm);
    n_spikes_all = cellfun(@(x) x.n_spikes, successful_vm);
    filter_corrs = cellfun(@(x) x.filter_correlation, successful_vm);

    % Extract elbow detection statistics
    elbow_success_mask = cellfun(@(x) isfield(x, 'elbow_detection_success') && x.elbow_detection_success, successful_vm);
    successful_elbow = successful_vm(elbow_success_mask);

    fprintf('\nVm_all Summary Statistics:\n');
    fprintf('  Duration: %.2f ± %.2f s (range: %.2f - %.2f s)\n', ...
        mean(durations), std(durations), min(durations), max(durations));
    fprintf('  Firing Rate: %.2f ± %.2f Hz (range: %.2f - %.2f Hz)\n', ...
        mean(firing_rates), std(firing_rates), min(firing_rates), max(firing_rates));
    fprintf('  Total Spikes: %.0f ± %.0f (range: %.0f - %.0f)\n', ...
        mean(n_spikes_all), std(n_spikes_all), min(n_spikes_all), max(n_spikes_all));
    fprintf('  Filter Correlation: %.3f ± %.3f (range: %.3f - %.3f)\n', ...
        mean(filter_corrs), std(filter_corrs), min(filter_corrs), max(filter_corrs));
    fprintf('  Vm Samples: %.0f ± %.0f (range: %.0f - %.0f)\n', ...
        mean(n_samples_all), std(n_samples_all), min(n_samples_all), max(n_samples_all));

    % Elbow detection summary
    if ~isempty(successful_elbow)
        elbow_firing_rates = cellfun(@(x) x.firing_rate_elbow_Hz, successful_elbow);
        elbow_n_spikes = cellfun(@(x) x.n_spikes_elbow, successful_elbow);
        elbow_thresholds = cellfun(@(x) x.elbow_threshold, successful_elbow);

        fprintf('\nElbow Spike Detection Summary:\n');
        fprintf('  Successful elbow detections: %d/%d (%.1f%%)\n', ...
            length(successful_elbow), length(successful_vm), ...
            100*length(successful_elbow)/length(successful_vm));
        fprintf('  Elbow Firing Rate: %.2f ± %.2f Hz (range: %.2f - %.2f Hz)\n', ...
            mean(elbow_firing_rates), std(elbow_firing_rates), min(elbow_firing_rates), max(elbow_firing_rates));
        fprintf('  Elbow Spike Count: %.0f ± %.0f (range: %.0f - %.0f)\n', ...
            mean(elbow_n_spikes), std(elbow_n_spikes), min(elbow_n_spikes), max(elbow_n_spikes));
        fprintf('  Elbow Threshold: %.2f ± %.2f mV (range: %.2f - %.2f mV)\n', ...
            mean(elbow_thresholds), std(elbow_thresholds), min(elbow_thresholds), max(elbow_thresholds));

        % Compare with original detection
        original_firing_rates = cellfun(@(x) x.firing_rate_Hz, successful_elbow);
        original_n_spikes = cellfun(@(x) x.n_spikes, successful_elbow);

        firing_rate_diff = elbow_firing_rates - original_firing_rates;
        spike_count_diff = elbow_n_spikes - original_n_spikes;

        fprintf('\nComparison (Elbow vs Original):\n');
        fprintf('  Firing Rate Difference: %.2f ± %.2f Hz (range: %.2f - %.2f Hz)\n', ...
            mean(firing_rate_diff), std(firing_rate_diff), min(firing_rate_diff), max(firing_rate_diff));
        fprintf('  Spike Count Difference: %.1f ± %.1f (range: %.0f - %.0f)\n', ...
            mean(spike_count_diff), std(spike_count_diff), min(spike_count_diff), max(spike_count_diff));
    else
        fprintf('\nNo successful elbow detections to report\n');
    end

    % Show sample successful extractions
    fprintf('\nSample Vm_all extractions:\n');
    n_samples = min(5, length(successful_vm));
    for i = 1:n_samples
        vm = successful_vm{i};
        if isfield(vm, 'elbow_detection_success') && vm.elbow_detection_success
            fprintf('  %s | freq=%s | %d samples | %d spikes (orig) | %d spikes (elbow) | %.1f Hz | %.1fs\n', ...
                vm.cell_id, vm.frequency, vm.n_samples, vm.n_spikes, vm.n_spikes_elbow, vm.firing_rate_Hz, vm.duration_s);
        else
            fprintf('  %s | freq=%s | %d samples | %d spikes | %.1f Hz | %.1fs | elbow: failed\n', ...
                vm.cell_id, vm.frequency, vm.n_samples, vm.n_spikes, vm.firing_rate_Hz, vm.duration_s);
        end
    end

    if length(successful_vm) > n_samples
        fprintf('  ... and %d more\n', length(successful_vm) - n_samples);
    end
end

if ~isempty(failed_vm)
    fprintf('\nFailed Vm_all extractions:\n');
    for i = 1:length(failed_vm)
        vm = failed_vm{i};
        fprintf('  %s | freq=%s | Error: %s\n', ...
            vm.cell_id, vm.frequency, vm.error_message);
    end
end

fprintf('\nVm_all data stored in: vm_all_data\n');
fprintf('Access using: vm_all_data{i}.Vm_all for the raw voltage trace\n');
fprintf('Access using: vm_all_data{i}.n_samples for summary statistics\n');

% ===================================================================
%  OPTIONAL: CREATE Vm_all VISUALIZATION - ALL CELLS IN ONE PLOT
% ===================================================================

if ~isempty(successful_vm)
    fprintf('\n=== CREATING Vm_all VISUALIZATION (ALL CELLS, COLOR CODED) ===\n');

    % Group data by cell
    cell_ids = cellfun(@(x) x.cell_id, successful_vm, 'UniformOutput', false);
    unique_cells = unique(cell_ids);

    fprintf('Creating combined plot for %d unique cells\n', length(unique_cells));

    % Create one figure for all cells
    figure('Position', [100, 100, 1400, 800]);
    sgtitle('Vm_all Data Summary - All Cells (Color Coded)', 'FontSize', 14);

    % Create color map for cells
    cell_colors = lines(length(unique_cells));

    % Create cell color mapping - ensure same length
    cell_color_map = containers.Map();
    for i = 1:length(unique_cells)
        cell_color_map(unique_cells{i}) = cell_colors(i, :);
    end

    % Extract all data
    frequencies = cellfun(@(x) str2double(x.frequency), successful_vm);
    durations = cellfun(@(x) x.duration_s, successful_vm);
    n_samples = cellfun(@(x) x.n_samples, successful_vm);
    firing_rates = cellfun(@(x) x.firing_rate_Hz, successful_vm);
    n_spikes = cellfun(@(x) x.n_spikes, successful_vm);
    filter_corrs = cellfun(@(x) x.filter_correlation, successful_vm);

    % Subplot 1: Duration vs frequency (color coded by cell)
    subplot(2, 3, 1);
    hold on;
    for i = 1:length(successful_vm)
        vm = successful_vm{i};
        cell_color = cell_color_map(vm.cell_id);
        scatter(frequencies(i), durations(i), 80, 'filled', 'MarkerFaceColor', cell_color);
    end
    xlabel('Frequency (Hz)');
    ylabel('Duration (s)');
    title('Recording Duration vs Frequency');
    grid on;
    set(gca, 'XScale', 'log');

    % Subplot 2: Number of samples vs frequency (color coded by cell)
    subplot(2, 3, 2);
    hold on;
    for i = 1:length(successful_vm)
        vm = successful_vm{i};
        cell_color = cell_color_map(vm.cell_id);
        scatter(frequencies(i), n_samples(i), 80, 'filled', 'MarkerFaceColor', cell_color);
    end
    xlabel('Frequency (Hz)');
    ylabel('Number of Vm Samples');
    title('Vm Samples vs Frequency');
    grid on;
    set(gca, 'XScale', 'log');

    % Subplot 3: Firing rate vs frequency (color coded by cell)
    subplot(2, 3, 3);
    hold on;
    for i = 1:length(successful_vm)
        vm = successful_vm{i};
        cell_color = cell_color_map(vm.cell_id);
        scatter(frequencies(i), firing_rates(i), 80, 'filled', 'MarkerFaceColor', cell_color);
    end
    xlabel('Frequency (Hz)');
    ylabel('Firing Rate (Hz)');
    title('Firing Rate vs Frequency');
    grid on;
    set(gca, 'XScale', 'log');

    % Subplot 4: Number of spikes vs frequency (color coded by cell)
    subplot(2, 3, 4);
    hold on;
    for i = 1:length(successful_vm)
        vm = successful_vm{i};
        cell_color = cell_color_map(vm.cell_id);
        scatter(frequencies(i), n_spikes(i), 80, 'filled', 'MarkerFaceColor', cell_color);
    end
    xlabel('Frequency (Hz)');
    ylabel('Number of Spikes');
    title('Spike Count vs Frequency');
    grid on;
    set(gca, 'XScale', 'log');

    % Subplot 5: Filter correlation vs frequency (color coded by cell)
    subplot(2, 3, 5);
    hold on;
    for i = 1:length(successful_vm)
        vm = successful_vm{i};
        cell_color = cell_color_map(vm.cell_id);
        scatter(frequencies(i), filter_corrs(i), 80, 'filled', 'MarkerFaceColor', cell_color);
    end
    xlabel('Frequency (Hz)');
    ylabel('Filter Correlation');
    title('Filter Correlation vs Frequency');
    grid on;
    set(gca, 'XScale', 'log');

    % Subplot 6: Vm traces (limited to 1 second, color coded by cell)
    subplot(2, 3, 6);
    hold on;

    for i = 1:length(successful_vm)
        vm = successful_vm{i};
        cell_color = cell_color_map(vm.cell_id);
        time_axis = (0:length(vm.Vm_all)-1) * vm.dt;

        % Limit to 1 second
        max_time = 1.0;  % 1 second
        time_mask = time_axis <= max_time;
        time_axis_limited = time_axis(time_mask);
        vm_limited = vm.Vm_all(time_mask);

        plot(time_axis_limited, vm_limited, 'Color', cell_color, 'LineWidth', 1);
    end

    xlabel('Time (s)');
    ylabel('Vm (mV)');
    title(sprintf('Vm Traces (≤1s, %d total)', length(successful_vm)));
    grid on;

    % Create legend for cells
    legend_labels = {};
    for i = 1:length(unique_cells)
        cell_id = unique_cells{i};
        cell_mask = strcmp(cell_ids, cell_id);
        cell_data = successful_vm(cell_mask);
        legend_labels{i} = sprintf('%s (%d freq)', cell_id, length(cell_data));
    end

    % Add legend to the Vm traces subplot
    legend(legend_labels, 'Location', 'best', 'FontSize', 8);

    % Add summary info text box
    summary_info = sprintf('Total cells: %d\nTotal recordings: %d\nTime window: 1s', ...
        length(unique_cells), length(successful_vm));
    text(0.02, 0.98, summary_info, 'Units', 'normalized', ...
        'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', ...
        'BackgroundColor', 'white', 'EdgeColor', 'black', 'FontSize', 9);

    fprintf('Combined plot created with %d cells and %d total recordings\n', ...
        length(unique_cells), length(successful_vm));
else
    fprintf('No successful Vm_all extractions for visualization\n');
end

fprintf('\n=== SECTION 4 COMPLETE ===\n');


%%
% ===================================================================
%  GUI DATA EXTRACTION - Add this AFTER your main vm_all_data loop
% ===================================================================

fprintf('\n=== PREPARING GUI DATA CONTAINER ===\n');

% Initialize GUI data container
gui_data = struct();
gui_data.cell_names = {};
gui_data.organized_data = struct();
gui_data.total_entries = 0;
gui_data.successful_entries = 0;

% Navigate through the tree hierarchy to reach frequency nodes (same as main loop)
for protocol_idx = 1:CurrentNode.children.length
    protocolNode = CurrentNode.children.elements(protocol_idx);
    protocol_name = protocolNode.splitValue;
    fprintf('Protocol: %s\n', protocol_name);

    for date_idx = 1:protocolNode.children.length
        dateNode = protocolNode.children.elements(date_idx);
        date_name = dateNode.splitValue;
        fprintf('  Date: %s\n', date_name);

        for cell_idx = 1:dateNode.children.length
            cellNode = dateNode.children.elements(cell_idx);
            cell_name = cellNode.splitValue;
            fprintf('    Cell: %s\n', cell_name);

            % Level 4: Epoch Group
            for group_idx = 1:cellNode.children.length
                groupNode = cellNode.children.elements(group_idx);
                group_name = groupNode.splitValue;

                % Level 5: FREQUENCY NODES - This is our target level
                for freq_idx = 1:groupNode.children.length
                    freqNode = groupNode.children.elements(freq_idx);
                    freq_value = freqNode.splitValue;
                    freq_str = num2str(freq_value);
                    
                    gui_data.total_entries = gui_data.total_entries + 1;
                    
                    % Check for stored results at frequency node level (same as main loop)
                    if freqNode.custom.containsKey('results')
                        stored_results = freqNode.custom.get('results');

                        if ~isempty(stored_results)
                            fprintf('      >>> Processing freq node: %s Hz <<<\n', freq_str);
                            
                            try
                                % Extract data same way as main loop
                                Vm_all = get(stored_results, 'Vm_all');
                                dt = get(stored_results, 'dt');
                                n_spikes = get(stored_results, 'n_spikes');
                                firing_rate = get(stored_results, 'firing_rate_Hz');
                                duration = get(stored_results, 'total_duration_s');
                                
                                % Get spike data if available
                                spike_indices = [];
                                n_spikes_elbow = 0;
                                if stored_results.containsKey('spike_indices')
                                    spike_indices = get(stored_results, 'spike_indices');
                                    n_spikes_elbow = length(spike_indices);
                                end
                                
                                % Only proceed if we have good data
                                if n_spikes_elbow >= 5
                                    % Create valid field name for struct
                                    cell_field = matlab.lang.makeValidName(cell_name);
                                    
                                    % Initialize cell structure if not exists
                                    if ~isfield(gui_data.organized_data, cell_field)
                                        gui_data.organized_data.(cell_field) = struct();
                                        gui_data.organized_data.(cell_field).original_name = cell_name;
                                        gui_data.organized_data.(cell_field).frequencies = {};
                                        gui_data.organized_data.(cell_field).data = containers.Map('KeyType', 'char', 'ValueType', 'any');
                                        
                                        % Add to unique cells list
                                        if ~ismember(cell_name, gui_data.cell_names)
                                            gui_data.cell_names{end+1} = cell_name;
                                        end
                                    end
                                    
                                    % Extract essential data for GUI
                                    gui_entry = struct();
                                    gui_entry.cell_id = cell_name;
                                    gui_entry.frequency = freq_str;
                                    gui_entry.protocol = protocol_name;
                                    gui_entry.date = date_name;
                                    gui_entry.group = group_name;
                                    
                                    % Core analysis data from stored results
                                    gui_entry.Vm_all = Vm_all;
                                    gui_entry.spike_indices = spike_indices;
                                    gui_entry.dt = dt;
                                    gui_entry.n_spikes = n_spikes_elbow;
                                    gui_entry.firing_rate_Hz = firing_rate;
                                    gui_entry.duration_s = duration;
                                    
                                    % Extract injected current using the freq_node
                                    gui_entry.injected_current = [];
                                    try
                                        Stimuli = getNoiseStm(freqNode);
                                        if ~isempty(Stimuli)
                                            gui_entry.injected_current = reshape(Stimuli', [], 1);
                                            fprintf('        Extracted injected current: %d samples\n', length(gui_entry.injected_current));
                                        end
                                    catch ME
                                        fprintf('        Warning: Could not extract injected current: %s\n', ME.message);
                                    end
                                    
                                    % Calculate auto-suggested window from current autocorrelation
                                    gui_entry.auto_window_ms = 3; % Default fallback
                                    if ~isempty(gui_entry.injected_current)
                                        try
                                            gui_entry.auto_window_ms = calculateOptimalWindowFromCurrent(gui_entry.injected_current, gui_entry.dt, false);
                                            fprintf('        Auto-calculated window: %.1f ms\n', gui_entry.auto_window_ms);
                                        catch
                                            % Keep default
                                        end
                                    end
                                    
                                    % Pre-compute for faster GUI updates
                                    gui_entry.max_window_samples = round(10 / (gui_entry.dt * 1000));
                                    gui_entry.time_ms = (0:length(gui_entry.Vm_all)-1) * gui_entry.dt * 1000;
                                    
                                    % Store in organized structure
                                    gui_data.organized_data.(cell_field).data(freq_str) = gui_entry;
                                    
                                    % Add frequency to list if not already there
                                    if ~ismember(freq_str, gui_data.organized_data.(cell_field).frequencies)
                                        gui_data.organized_data.(cell_field).frequencies{end+1} = freq_str;
                                    end
                                    
                                    gui_data.successful_entries = gui_data.successful_entries + 1;
                                    fprintf('        Added to GUI data: %d spikes, %.1f Hz\n', n_spikes_elbow, firing_rate);
                                else
                                    fprintf('        Skipped: insufficient spikes (%d < 5)\n', n_spikes_elbow);
                                end
                                
                            catch ME
                                fprintf('        Error extracting data: %s\n', ME.message);
                            end
                        end
                    end
                end  % End frequency loop
            end  % End group loop
        end  % End cell loop
    end  % End date loop
end  % End protocol loop

% Sort frequencies for each cell
for cell_idx = 1:length(gui_data.cell_names)
    cell_name = gui_data.cell_names{cell_idx};
    cell_field = matlab.lang.makeValidName(cell_name);
    if isfield(gui_data.organized_data, cell_field)
        frequencies = gui_data.organized_data.(cell_field).frequencies;
        if ~isempty(frequencies)
            % Convert to numbers, sort, then back to strings
            freq_nums = str2double(frequencies);
            [~, sort_idx] = sort(freq_nums);
            gui_data.organized_data.(cell_field).frequencies = frequencies(sort_idx);
        end
    end
end

fprintf('\nGUI Data Container Summary:\n');
fprintf('  Total frequency nodes processed: %d\n', gui_data.total_entries);
fprintf('  Successfully extracted entries: %d\n', gui_data.successful_entries);
fprintf('  Cells with data: %d\n', length(gui_data.cell_names));

% Display summary of available data
for cell_idx = 1:length(gui_data.cell_names)
    cell_name = gui_data.cell_names{cell_idx};
    cell_field = matlab.lang.makeValidName(cell_name);
    if isfield(gui_data.organized_data, cell_field)
        frequencies = gui_data.organized_data.(cell_field).frequencies;
        fprintf('    %s: %s Hz\n', cell_name, strjoin(frequencies, ', '));
    end
end
gui_data.save_path = '/Users/maxwellsdm/Library/CloudStorage/GoogleDrive-maxwellsdm1867@gmail.com/.shortcut-targets-by-id/1CbZnfugdi-p4jM3fe7t-NyGk5v2ZcPdk/ParasolCenterSurround/figures/ScienceJuiceFactory/currinjt/overview/RGC';
fprintf('\nGUI data container ready. Use createSpikeInitiationGUI(gui_data) to launch GUI.\n');
% fourFactorGuiSeparated(gui_data);
% fourFactorGuiSeparated2(gui_data);
gui_data.cleanSpikeVis = false;
fourFactorGuiSeparated3(gui_data);
%% === SRM LOSS FUNCTIONS ===
% These functions are used by the optimization routines above

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

catch ME
    % Penalize failed simulations heavily
    loss = 1000;
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

catch ME
    % Penalize failed simulations heavily
    loss = 1000;
end
end
% ===================================================================
%  DEBUG LOGGING SYSTEM FOR SRM ANALYSIS
% ===================================================================


% Initialize debug logging system
function initializeDebugLog()
global DEBUG_LOG;
global DEBUG_LOG_COUNTER;

DEBUG_LOG = {};
DEBUG_LOG_COUNTER = 1;

fprintf('\n=== DEBUG LOGGING INITIALIZED ===\n');
end

% Main debug logging function
function debugLog(level, category, cell_id, freq_str, sd_str, message, varargin)
global DEBUG_LOG;
global DEBUG_LOG_COUNTER;

% Create timestamp
timestamp = datetime('now', 'Format', 'HH:mm:ss.SSS');

% Parse optional data
data = struct();
if ~isempty(varargin)
    for i = 1:2:length(varargin)
        if i+1 <= length(varargin)
            data.(varargin{i}) = varargin{i+1};
        end
    end
end

% Create log entry
log_entry = struct();
log_entry.timestamp = timestamp;
log_entry.level = level;        % INFO, WARN, ERROR, SUCCESS
log_entry.category = category;  % ENTRY, VARIABLE, INDEXING, RESULT
log_entry.cell_id = cell_id;
log_entry.frequency = freq_str;
log_entry.sd = sd_str;
log_entry.message = message;
log_entry.data = data;

% Store in global log
DEBUG_LOG{DEBUG_LOG_COUNTER} = log_entry;
DEBUG_LOG_COUNTER = DEBUG_LOG_COUNTER + 1;

% Format and print to console
location_str = sprintf('%s|freq=%s|SD=%s', cell_id, freq_str, sd_str);
level_prefix = sprintf('[%s-%s]', level, category);

fprintf('%s %s %s: %s\n', char(timestamp), level_prefix, location_str, message);

% Print data if present
if ~isempty(fieldnames(data))
    data_fields = fieldnames(data);
    for i = 1:length(data_fields)
        field = data_fields{i};
        value = data.(field);
        if isnumeric(value)
            fprintf('    %s: %g\n', field, value);
        else
            fprintf('    %s: %s\n', field, char(value));
        end
    end
end
end

% Specialized logging functions for common scenarios
function debugLogEntry(cell_id, freq_str, sd_str, n_epochs)
debugLog('INFO', 'ENTRY', cell_id, freq_str, sd_str, ...
    sprintf('Entering debug mode analysis with %d epochs', n_epochs), ...
    'n_epochs', n_epochs);
end

function debugLogVariable(cell_id, freq_str, sd_str, var_name, exists, details)
if exists
    debugLog('SUCCESS', 'VARIABLE', cell_id, freq_str, sd_str, ...
        sprintf('Variable %s found', var_name), ...
        'variable', var_name, 'details', details);
else
    debugLog('WARN', 'VARIABLE', cell_id, freq_str, sd_str, ...
        sprintf('Variable %s MISSING', var_name), ...
        'variable', var_name, 'details', details);
end
end

function debugLogIndexing(cell_id, freq_str, sd_str, operation, success, details)
if success
    debugLog('SUCCESS', 'INDEXING', cell_id, freq_str, sd_str, ...
        sprintf('Indexing operation: %s', operation), ...
        'operation', operation, 'details', details);
else
    debugLog('ERROR', 'INDEXING', cell_id, freq_str, sd_str, ...
        sprintf('Indexing FAILED: %s', operation), ...
        'operation', operation, 'details', details);
end
end

function debugLogResult(cell_id, freq_str, sd_str, success, vp_train, vp_holdout)
if success
    debugLog('SUCCESS', 'RESULT', cell_id, freq_str, sd_str, ...
        sprintf('Analysis completed successfully'), ...
        'vp_train', vp_train, 'vp_holdout', vp_holdout);
else
    debugLog('ERROR', 'RESULT', cell_id, freq_str, sd_str, ...
        sprintf('Analysis failed'), ...
        'vp_train', vp_train, 'vp_holdout', vp_holdout);
end
end

% Function to check and log variable availability
function [exists, value, details] = checkAndLogVariable(stored_results, var_name, cell_id, freq_str, sd_str)
exists = false;
value = [];
details = '';

try
    if stored_results.containsKey(var_name)
        value = get(stored_results, var_name);
        exists = true;

        % Get details about the variable
        if isnumeric(value)
            if isscalar(value)
                details = sprintf('scalar: %g', value);
            else
                details = sprintf('array: %dx%d', size(value, 1), size(value, 2));
            end
        else
            details = sprintf('type: %s', class(value));
        end
    else
        details = 'key not found';
    end
catch ME
    exists = false;
    details = sprintf('error: %s', ME.message);
end

debugLogVariable(cell_id, freq_str, sd_str, var_name, exists, details);
end

% Helper function to standardize debug log entries
function standardized_entry = standardizeDebugEntry(entry)
% STANDARDIZEDEBUGENTRY - Convert debug entry to standard format
% This function ensures all debug entries have consistent field names
% regardless of which logging system created them

standardized_entry = struct();

% Copy timestamp
if isfield(entry, 'timestamp')
    standardized_entry.timestamp = entry.timestamp;
else
    standardized_entry.timestamp = datetime('now');
end

% Standardize level field
if isfield(entry, 'level')
    standardized_entry.level = entry.level;
else
    standardized_entry.level = 'UNKNOWN';
end

% Standardize category field
if isfield(entry, 'category')
    standardized_entry.category = entry.category;
else
    standardized_entry.category = 'UNKNOWN';
end

% Standardize cell identifier (prefer cell_id, fallback to cell)
if isfield(entry, 'cell_id')
    standardized_entry.cell_id = entry.cell_id;
elseif isfield(entry, 'cell')
    standardized_entry.cell_id = entry.cell;
else
    standardized_entry.cell_id = 'UNKNOWN';
end

% Copy frequency and SD
if isfield(entry, 'frequency')
    standardized_entry.frequency = entry.frequency;
else
    standardized_entry.frequency = 'UNKNOWN';
end

if isfield(entry, 'sd')
    standardized_entry.sd = entry.sd;
else
    standardized_entry.sd = 'UNKNOWN';
end

% Copy message
if isfield(entry, 'message')
    standardized_entry.message = entry.message;
else
    standardized_entry.message = 'No message';
end

% Standardize data/details field (prefer data, fallback to details)
if isfield(entry, 'data')
    standardized_entry.data = entry.data;
elseif isfield(entry, 'details')
    standardized_entry.data = struct('details', entry.details);
else
    standardized_entry.data = struct();
end
end

% Function to display debug log summary
% This function handles debug entries from both the local logging system
% and the DebugUtils system, which have different field names.
% The standardizeDebugEntry() helper ensures compatibility.
function displayDebugSummary()
global DEBUG_LOG;

if isempty(DEBUG_LOG)
    fprintf('\n=== NO DEBUG LOG ENTRIES ===\n');
    return;
end

fprintf('\n===============================================\n');
fprintf('DEBUG LOG SUMMARY\n');
fprintf('===============================================\n');

% Count by level and category
levels = {};
categories = {};
cells = {};

for i = 1:length(DEBUG_LOG)
    % Standardize the entry to ensure consistent field names
    entry = standardizeDebugEntry(DEBUG_LOG{i});

    % Now we can safely access standardized fields
    levels{end+1} = entry.level;
    categories{end+1} = entry.category;
    cells{end+1} = entry.cell_id;
end

% Summary statistics
unique_levels = unique(levels);
unique_categories = unique(categories);
unique_cells = unique(cells);

fprintf('Total log entries: %d\n', length(DEBUG_LOG));
fprintf('Unique cells processed: %d\n', length(unique_cells));

fprintf('\nBy Level:\n');
for i = 1:length(unique_levels)
    level = unique_levels{i};
    count = sum(strcmp(levels, level));
    fprintf('  %s: %d\n', level, count);
end

fprintf('\nBy Category:\n');
for i = 1:length(unique_categories)
    category = unique_categories{i};
    count = sum(strcmp(categories, category));
    fprintf('  %s: %d\n', category, count);
end

% Show cells that entered debug mode
entry_logs = DEBUG_LOG(strcmp(categories, 'ENTRY'));
if ~isempty(entry_logs)
    fprintf('\nCells that ENTERED debug mode:\n');
    for i = 1:length(entry_logs)
        entry = standardizeDebugEntry(entry_logs{i});
        fprintf('  %s | freq=%s | SD=%s\n', entry.cell_id, entry.frequency, entry.sd);
    end
end

% Show variable check failures
var_logs = DEBUG_LOG(strcmp(categories, 'VARIABLE') & strcmp(levels, 'WARN'));
if ~isempty(var_logs)
    fprintf('\nMISSING VARIABLES:\n');
    for i = 1:length(var_logs)
        entry = standardizeDebugEntry(var_logs{i});

        % Get variable details from standardized data field
        if isfield(entry.data, 'variable') && isfield(entry.data, 'details')
            fprintf('  %s | freq=%s: %s (%s)\n', entry.cell_id, entry.frequency, ...
                entry.data.variable, entry.data.details);
        elseif isfield(entry.data, 'details')
            fprintf('  %s | freq=%s: %s\n', entry.cell_id, entry.frequency, entry.data.details);
        else
            fprintf('  %s | freq=%s: Entry %d: Missing variable details\n', entry.cell_id, entry.frequency, i);
        end
    end
end

% Show indexing failures
index_logs = DEBUG_LOG(strcmp(categories, 'INDEXING') & strcmp(levels, 'ERROR'));
if ~isempty(index_logs)
    fprintf('\nINDEXING FAILURES:\n');
    for i = 1:length(index_logs)
        entry = standardizeDebugEntry(index_logs{i});
        fprintf('  %s | freq=%s: %s\n', entry.cell_id, entry.frequency, entry.message);
    end
end

fprintf('\n===============================================\n');
end






