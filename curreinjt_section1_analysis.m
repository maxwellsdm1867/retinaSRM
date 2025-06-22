fprintf('\n=== STARTING TREE NAVIGATION, SELECTION & ANALYSIS ===\n');

% Set up foreground figure mode for entire analysis
set(0, 'DefaultFigureVisible', 'on');  % All new figures visible by default
set(0, 'DefaultFigureWindowStyle', 'normal');  % Normal window style
set(0, 'DefaultFigureHandleVisibility', 'on');  % Bring to front

fprintf('Foreground figure mode enabled - figures will be visible\n');

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

                                % Create designated figure for filter analysis (only once) - FOREGROUND MODE
                                if results_counter == 1
                                    filter_fig = figure(100);  % Designated figure number for all filter plots
                                    set(filter_fig, 'Position', [100, 100, 1200, 800], 'Visible', 'on');  % Set figure size and show
                                    % Ensure figure is visible in foreground
                                    set(filter_fig, 'WindowStyle', 'normal', 'HandleVisibility', 'on');
                                    fprintf('            Created designated filter figure (Figure 100) - Foreground mode\n');
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

                                % Save the figure immediately after plotting (foreground mode)
                                saveas(filter_fig, [fig_path fig_filename]);
                                fprintf('            Filter figure saved: %s (foreground mode)\n', [fig_path fig_filename]);

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

                                % Log success
                                log_entry = LoggingUtils.logSuccessfulAnalysis(cell_name, protocol_name, ...
                                    freq_value, selected_sd_value, max_epochs, results);

                                analysis_log{log_counter} = log_entry;
                                log_counter = log_counter + 1;

                            catch ME
                                fprintf('          >>> ERROR: Analysis failed <<<\n');
                                fprintf('            Error: %s\n', ME.message);

                                % Log error and store failed analysis info
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

% -----------------------------------------------------------------------
%  LOG COPYING OPTION
% -----------------------------------------------------------------------

% Use ClipboardUtils to handle log copying
ClipboardUtils.copyAnalysisLogToClipboard(analysis_log);

fprintf('\n=== SECTION 1 COMPLETE ===\n'); 