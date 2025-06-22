classdef DataExtractor
    % DATAEXTRACTOR - Class for extracting data from epoch nodes
    % This class handles the extraction of voltage recordings, current
    % stimuli, and other data from epoch nodes in the tree structure.
    %
    % The class provides methods to:
    % - Extract voltage and current data from nodes
    % - Preprocess and concatenate data from multiple trials
    % - Validate data quality and dimensions
    % - Handle missing or corrupted data
    %
    % Properties:
    %   amp_channel - Amplifier channel name for data extraction
    %   verbose - Verbosity level for logging
    %
    % Author: Maxwell
    % Date: 2024
    
    properties
        amp_channel = 'Amp1'
        verbose = true
    end
    
    methods
        function obj = DataExtractor(amp_channel, verbose)
            % DATAEXTRACTOR Constructor
            % Input:
            %   amp_channel - Amplifier channel name (default: 'Amp1')
            %   verbose - Verbosity level (default: true)
            % Output:
            %   obj - DataExtractor instance
            
            if nargin >= 1
                obj.amp_channel = amp_channel;
            end
            
            if nargin >= 2
                obj.verbose = verbose;
            end
        end
        
        function [voltage_data, current_data, metadata] = extractNodeData(obj, node)
            % EXTRACTNODEDATA - Extract voltage and current data from a node
            % Input:
            %   node - Node containing epoch data
            % Output:
            %   voltage_data - Voltage recordings (concatenated trials)
            %   current_data - Current stimuli (concatenated trials)
            %   metadata - Structure containing data metadata
            
            if obj.verbose
                fprintf('Extracting data from node: %s\n', node.splitValue);
            end
            
            try
                % Check if required functions are available
                if ~exist('getSelectedData', 'file') && ~exist('getSelectedData', 'builtin')
                    error('getSelectedData function not available');
                end
                
                if ~exist('getNoiseStm', 'file') && ~exist('getNoiseStm', 'builtin')
                    error('getNoiseStm function not available');
                end
                
                % Extract electrode data and injected current stimulus
                epoch_data = getSelectedData(node.epochList, obj.amp_channel);
                stimuli_data = getNoiseStm(node);
                
                % Get data dimensions
                [n_trials, n_timepoints] = size(stimuli_data);
                dt = 0.0001; % 10 kHz sampling (should be configurable)
                
                if obj.verbose
                    fprintf('  Extracted %d trials with %d timepoints each\n', n_trials, n_timepoints);
                    fprintf('  Sampling interval: %.1f μs\n', dt * 1e6);
                end
                
                % Concatenate all trials for analysis
                current_data = reshape(stimuli_data', [], 1);
                voltage_data = reshape(epoch_data', [], 1);
                
                if obj.verbose
                    fprintf('  Concatenated data: %d total timepoints\n', length(voltage_data));
                end
                
                % Create metadata
                metadata = struct();
                metadata.n_trials = n_trials;
                metadata.n_timepoints = n_timepoints;
                metadata.dt = dt;
                metadata.total_duration = length(voltage_data) * dt;
                metadata.sampling_rate = 1 / dt;
                metadata.extraction_timestamp = datetime('now');
                metadata.success = true;
                
            catch ME
                if obj.verbose
                    fprintf('  ERROR: Failed to extract data: %s\n', ME.message);
                end
                
                voltage_data = [];
                current_data = [];
                metadata = struct();
                metadata.error = ME;
                metadata.success = false;
            end
        end
        
        function [cleaned_voltage, filter_info] = cleanVoltageSignal(obj, voltage_data, dt, cutoff_freq)
            % CLEANVOLTAGESIGNAL - Clean voltage signal using low-pass filter
            % Input:
            %   voltage_data - Raw voltage data
            %   dt - Sampling interval (seconds)
            %   cutoff_freq - Cutoff frequency for low-pass filter (Hz)
            % Output:
            %   cleaned_voltage - Filtered voltage signal
            %   filter_info - Structure containing filter information
            
            if nargin < 4
                cutoff_freq = 90; % Default cutoff frequency
            end
            
            if obj.verbose
                fprintf('Cleaning voltage signal with %.1f Hz low-pass filter\n', cutoff_freq);
            end
            
            try
                % Calculate sampling rate
                sampling_rate = 1 / dt;
                
                % Design low-pass Butterworth filter
                normalized_cutoff = cutoff_freq / (sampling_rate / 2);
                [b, a] = butter(4, normalized_cutoff, 'low');
                
                % Apply zero-phase filtering to avoid phase distortion
                cleaned_voltage = filtfilt(b, a, voltage_data);
                
                % Create filter information
                filter_info = struct();
                filter_info.cutoff_freq = cutoff_freq;
                filter_info.normalized_cutoff = normalized_cutoff;
                filter_info.filter_order = 4;
                filter_info.filter_type = 'low';
                filter_info.sampling_rate = sampling_rate;
                filter_info.success = true;
                
                if obj.verbose
                    fprintf('  Low-pass filtering complete\n');
                end
                
            catch ME
                if obj.verbose
                    fprintf('  ERROR: Failed to clean voltage signal: %s\n', ME.message);
                end
                
                cleaned_voltage = voltage_data; % Return original data
                filter_info = struct();
                filter_info.error = ME;
                filter_info.success = false;
            end
        end
        
        function [spike_indices, spike_times, diagnostic_info] = detectSpikes(obj, voltage_data, dt, spike_config)
            % DETECTSPIKES - Detect spikes using enhanced elbow method
            % Input:
            %   voltage_data - Voltage signal
            %   dt - Sampling interval (seconds)
            %   spike_config - SpikeConfig object with detection parameters
            % Output:
            %   spike_indices - Indices of detected spikes
            %   spike_times - Times of detected spikes (seconds)
            %   diagnostic_info - Structure containing detection diagnostics
            
            if obj.verbose
                fprintf('Detecting spikes using enhanced elbow method\n');
            end
            
            try
                % Extract parameters from spike config (same as original working code)
                vm_thresh = spike_config.vm_thresh;     % -20 mV
                d2v_thresh = spike_config.d2v_thresh;   % 50
                search_back_ms = spike_config.search_back_ms; % 2 ms
                plot_flag = spike_config.plot_flag;     % false for batch processing
                
                % Detect spike initiation points using enhanced elbow method v2
                % Use the EXACT same parameters as the original working code
                [elbow_indices, ~, ~, avg_spike_short, diagnostic_info] = detect_spike_initiation_elbow_v2(...
                    voltage_data, dt, vm_thresh, d2v_thresh, search_back_ms, plot_flag, ...
                    'elbow_thresh', -65, 'spike_thresh', -10, 'min_dv_thresh', 0.1, ...
                    'time_to_peak_thresh', 1.5);
                
                % Assign output variables
                spike_indices = elbow_indices;
                spike_times = elbow_indices * dt;
                
                % Add success flag to diagnostic info
                diagnostic_info.success = true;
                diagnostic_info.n_spikes = length(elbow_indices);
                
                if obj.verbose
                    fprintf('  Enhanced elbow method detected %d spikes\n', length(elbow_indices));
                end
                
            catch ME
                if obj.verbose
                    fprintf('  ERROR: Failed to detect spikes: %s\n', ME.message);
                end
                
                spike_indices = [];
                spike_times = [];
                diagnostic_info = struct();
                diagnostic_info.error = ME;
                diagnostic_info.success = false;
            end
        end
        
        function [sta_current, sta_voltage, sta_info] = computeSTA(obj, voltage_data, current_data, spike_indices, dt, analysis_config)
            % COMPUTESTA - Compute Spike-Triggered Average
            % Input:
            %   voltage_data - Voltage signal
            %   current_data - Current stimulus
            %   spike_indices - Indices of detected spikes
            %   dt - Sampling interval (seconds)
            %   analysis_config - AnalysisConfig object with window parameters
            % Output:
            %   sta_current - Spike-triggered average of current
            %   sta_voltage - Spike-triggered average of voltage
            %   sta_info - Structure containing STA information
            
            if obj.verbose
                fprintf('Computing Spike-Triggered Average\n');
            end
            
            try
                % Get window parameters
                win_before = round(analysis_config.sta_window_before_ms / 1000 / dt);
                win_after = round(analysis_config.sta_window_after_ms / 1000 / dt);
                
                % Compute STA for current
                sta_current = zeros(win_before, 1);
                valid_spikes_current = 0;
                
                for k = 1:length(spike_indices)
                    spike_idx = spike_indices(k);
                    if spike_idx > win_before
                        sta_current = sta_current + flipud(current_data(spike_idx - win_before + 1 : spike_idx));
                        valid_spikes_current = valid_spikes_current + 1;
                    end
                end
                
                if valid_spikes_current > 0
                    sta_current = sta_current / valid_spikes_current;
                end
                
                % Compute STA for voltage (afterpotential)
                sta_voltage = zeros(win_after, 1);
                valid_spikes_voltage = 0;
                
                for k = 1:length(spike_indices)
                    spike_idx = spike_indices(k);
                    if spike_idx + win_after - 1 <= length(voltage_data)
                        sta_voltage = sta_voltage + voltage_data(spike_idx : spike_idx + win_after - 1);
                        valid_spikes_voltage = valid_spikes_voltage + 1;
                    end
                end
                
                if valid_spikes_voltage > 0
                    sta_voltage = sta_voltage / valid_spikes_voltage;
                end
                
                % Create STA information
                sta_info = struct();
                sta_info.win_before_samples = win_before;
                sta_info.win_after_samples = win_after;
                sta_info.win_before_ms = analysis_config.sta_window_before_ms;
                sta_info.win_after_ms = analysis_config.sta_window_after_ms;
                sta_info.valid_spikes_current = valid_spikes_current;
                sta_info.valid_spikes_voltage = valid_spikes_voltage;
                sta_info.success = true;
                
                if obj.verbose
                    fprintf('  STA computed using %d/%d valid spikes (current/voltage)\n', ...
                        valid_spikes_current, valid_spikes_voltage);
                end
                
            catch ME
                if obj.verbose
                    fprintf('  ERROR: Failed to compute STA: %s\n', ME.message);
                end
                
                sta_current = [];
                sta_voltage = [];
                sta_info = struct();
                sta_info.error = ME;
                sta_info.success = false;
            end
        end
        
        function validateData(obj, voltage_data, current_data, metadata)
            % VALIDATEDATA - Validate extracted data quality
            % Input:
            %   voltage_data - Voltage signal
            %   current_data - Current stimulus
            %   metadata - Data metadata
            % Output:
            %   validation_result - Structure containing validation results
            
            validation_result = struct();
            validation_result.passed = true;
            validation_result.warnings = {};
            validation_result.errors = {};
            
            % Check data dimensions
            if length(voltage_data) ~= length(current_data)
                validation_result.passed = false;
                validation_result.errors{end+1} = 'Voltage and current data have different lengths';
            end
            
            % Check for NaN values
            if any(isnan(voltage_data))
                validation_result.warnings{end+1} = 'Voltage data contains NaN values';
            end
            
            if any(isnan(current_data))
                validation_result.warnings{end+1} = 'Current data contains NaN values';
            end
            
            % Check for infinite values
            if any(isinf(voltage_data))
                validation_result.errors{end+1} = 'Voltage data contains infinite values';
                validation_result.passed = false;
            end
            
            if any(isinf(current_data))
                validation_result.errors{end+1} = 'Current data contains infinite values';
                validation_result.passed = false;
            end
            
            % Check data range
            voltage_range = range(voltage_data);
            if voltage_range < 1
                validation_result.warnings{end+1} = sprintf('Voltage range is small: %.2f mV', voltage_range);
            end
            
            % Check metadata
            if metadata.n_trials <= 0
                validation_result.errors{end+1} = 'Invalid number of trials';
                validation_result.passed = false;
            end
            
            if metadata.n_timepoints <= 0
                validation_result.errors{end+1} = 'Invalid number of timepoints';
                validation_result.passed = false;
            end
            
            if obj.verbose && ~isempty(validation_result.warnings)
                fprintf('Data validation warnings:\n');
                for i = 1:length(validation_result.warnings)
                    fprintf('  - %s\n', validation_result.warnings{i});
                end
            end
            
            if obj.verbose && ~isempty(validation_result.errors)
                fprintf('Data validation errors:\n');
                for i = 1:length(validation_result.errors)
                    fprintf('  - %s\n', validation_result.errors{i});
                end
            end
        end
    end
end 