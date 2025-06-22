classdef AnalysisExecutor
    % ANALYSISEXECUTOR - Class for executing the complete analysis pipeline
    % This class orchestrates the entire analysis process for a frequency node,
    % including data extraction, spike detection, signal processing, and
    % result compilation.
    %
    % The analysis pipeline includes:
    % 1. Data extraction from the node
    % 2. Voltage signal cleaning and preprocessing
    % 3. Spike detection using enhanced elbow method
    % 4. Spike-triggered average computation
    % 5. Linear filter estimation
    % 6. Quality assessment and validation
    % 7. Result compilation and storage
    %
    % Properties:
    %   analysis_config - AnalysisConfig object with analysis parameters
    %   spike_config - SpikeConfig object with spike detection parameters
    %   data_extractor - DataExtractor object for data extraction
    %   verbose - Verbosity level for logging
    %
    % Author: Maxwell
    % Date: 2024
    
    properties
        analysis_config
        spike_config
        data_extractor
        verbose = true
    end
    
    methods
        function obj = AnalysisExecutor(analysis_config, spike_config, verbose)
            % ANALYSISEXECUTOR Constructor
            % Input:
            %   analysis_config - AnalysisConfig object
            %   spike_config - SpikeConfig object
            %   verbose - Verbosity level (default: true)
            % Output:
            %   obj - AnalysisExecutor instance
            
            obj.analysis_config = analysis_config;
            obj.spike_config = spike_config;
            
            if nargin >= 3
                obj.verbose = verbose;
            end
            
            % Create data extractor
            obj.data_extractor = DataExtractor(analysis_config.Amp, obj.verbose);
        end
        
        function result = analyzeFrequencyNode(obj, frequencyNode)
            % ANALYZEFREQUENCYNODE - Perform complete analysis on a frequency node
            % Input:
            %   frequencyNode - Frequency node structure from TreeNavigator
            % Output:
            %   result - Structure containing all analysis results
            
            if obj.verbose
                fprintf('\n=== ANALYZING FREQUENCY NODE ===\n');
                fprintf('Cell: %s | Protocol: %s | Frequency: %s\n', ...
                    frequencyNode.cell, frequencyNode.protocol, frequencyNode.frequency);
            end
            
            % Initialize result structure
            result = struct();
            result.node_info = frequencyNode;
            result.timestamp = datetime('now');
            result.success = false;
            
            try
                % Step 1: Select best SD level
                selected_sd = obj.selectBestSDLevel(frequencyNode);
                if isempty(selected_sd)
                    error('No valid SD level found for frequency node');
                end
                
                result.selected_sd = selected_sd;
                
                % Step 2: Extract data from selected SD node
                [voltage_data, current_data, data_metadata] = obj.data_extractor.extractNodeData(selected_sd.node);
                
                if ~data_metadata.success
                    error('Failed to extract data: %s', data_metadata.error.message);
                end
                
                result.data_metadata = data_metadata;
                
                % Step 3: Clean voltage signal
                [cleaned_voltage, filter_info] = obj.data_extractor.cleanVoltageSignal(...
                    voltage_data, data_metadata.dt, obj.analysis_config.cutoff_freq_Hz);
                
                result.filter_info = filter_info;
                
                % Step 4: Detect spikes
                [spike_indices, spike_times, spike_diagnostics] = obj.data_extractor.detectSpikes(...
                    cleaned_voltage, data_metadata.dt, obj.spike_config);
                
                if ~spike_diagnostics.success
                    error('Failed to detect spikes: %s', spike_diagnostics.error.message);
                end
                
                result.spike_diagnostics = spike_diagnostics;
                
                % Step 5: Compute spike statistics
                spike_stats = obj.computeSpikeStatistics(spike_indices, spike_times, data_metadata);
                result.spike_stats = spike_stats;
                
                % Step 6: Quality check
                quality_result = obj.performQualityCheck(spike_stats, data_metadata);
                result.quality_result = quality_result;
                
                if ~quality_result.passes_quality
                    if obj.verbose
                        fprintf('Quality check failed: %s\n', strjoin(quality_result.warnings, ', '));
                    end
                    result.success = false;
                    return;
                end
                
                % Step 7: Compute STA
                [sta_current, sta_voltage, sta_info] = obj.data_extractor.computeSTA(...
                    voltage_data, current_data, spike_indices, data_metadata.dt, obj.analysis_config);
                
                result.sta_current = sta_current;
                result.sta_voltage = sta_voltage;
                result.sta_info = sta_info;
                
                % Step 8: Estimate linear filter
                [linear_filter, filter_correlation] = obj.estimateLinearFilter(...
                    current_data, cleaned_voltage, data_metadata.dt);
                
                result.linear_filter = linear_filter;
                result.filter_correlation = filter_correlation;
                
                % Step 9: Analyze spike waveforms
                spike_waveform_analysis = obj.analyzeSpikeWaveforms(...
                    voltage_data, spike_indices, data_metadata.dt);
                
                result.spike_waveform_analysis = spike_waveform_analysis;
                
                % Step 10: Compile final results
                result = obj.compileResults(result, frequencyNode);
                result.success = true;
                
                if obj.verbose
                    fprintf('Analysis completed successfully\n');
                    fprintf('  Spikes: %d (%.2f Hz)\n', spike_stats.n_spikes, spike_stats.firing_rate);
                    fprintf('  Filter correlation: %.3f\n', filter_correlation);
                    if spike_waveform_analysis.success
                        fprintf('  Spike width: %.2f ms\n', spike_waveform_analysis.avg_width_ms);
                    else
                        fprintf('  Spike width: N/A (no spikes)\n');
                    end
                end
                
            catch ME
                if obj.verbose
                    fprintf('Analysis failed: %s\n', ME.message);
                end
                
                result.error = ME;
                result.success = false;
            end
        end
        
        function selected_sd = selectBestSDLevel(obj, frequencyNode)
            % SELECTBESTSDLEVEL - Select the SD level with the most epochs
            % Input:
            %   frequencyNode - Frequency node structure
            % Output:
            %   selected_sd - Selected SD level information
            
            sd_levels = frequencyNode.sd_levels;
            
            if isempty(sd_levels)
                selected_sd = [];
                return;
            end
            
            % Find SD level with maximum epochs
            epoch_counts = [sd_levels.n_epochs];
            [max_epochs, max_idx] = max(epoch_counts);
            
            selected_sd = sd_levels(max_idx);
            selected_sd.max_epochs = max_epochs;
            
            if obj.verbose
                fprintf('Selected SD level: %s with %d epochs\n', ...
                    selected_sd.value, max_epochs);
            end
        end
        
        function spike_stats = computeSpikeStatistics(obj, spike_indices, spike_times, data_metadata)
            % COMPUTESPIKESTATISTICS - Compute statistics from detected spikes
            % Input:
            %   spike_indices - Indices of detected spikes
            %   spike_times - Times of detected spikes (seconds)
            %   data_metadata - Data metadata
            % Output:
            %   spike_stats - Structure containing spike statistics
            
            spike_stats = struct();
            spike_stats.n_spikes = length(spike_indices);
            spike_stats.spike_indices = spike_indices;
            spike_stats.spike_times = spike_times;
            spike_stats.firing_rate = spike_stats.n_spikes / data_metadata.total_duration;
            
            % Compute inter-spike intervals
            if length(spike_times) > 1
                isi = diff(spike_times);
                spike_stats.mean_isi = mean(isi);
                spike_stats.std_isi = std(isi);
                spike_stats.cv_isi = spike_stats.std_isi / spike_stats.mean_isi;
            else
                spike_stats.mean_isi = NaN;
                spike_stats.std_isi = NaN;
                spike_stats.cv_isi = NaN;
            end
        end
        
        function quality_result = performQualityCheck(obj, spike_stats, data_metadata)
            % PERFORMQUALITYCHECK - Perform quality assessment of the analysis
            % Input:
            %   spike_stats - Spike statistics
            %   data_metadata - Data metadata
            % Output:
            %   quality_result - Structure containing quality assessment
            
            quality_result = struct();
            
            % Get quality warnings
            quality_result.warnings = obj.spike_config.getQualityWarnings(...
                spike_stats.n_spikes, spike_stats.firing_rate, 0.5, data_metadata.total_duration);
            
            % Check if analysis passes quality criteria
            quality_result.passes_quality = obj.spike_config.passesQualityCheck(...
                spike_stats.n_spikes, spike_stats.firing_rate, 0.5, data_metadata.total_duration);
            
            if obj.verbose && ~isempty(quality_result.warnings)
                fprintf('Quality warnings: %s\n', strjoin(quality_result.warnings, ', '));
            end
        end
        
        function [linear_filter, correlation] = estimateLinearFilter(obj, current_data, voltage_data, dt)
            % ESTIMATELINEARFILTER - Estimate linear filter using FFT method
            % Input:
            %   current_data - Current stimulus
            %   voltage_data - Voltage response (cleaned)
            %   dt - Sampling interval
            % Output:
            %   linear_filter - Estimated linear filter
            %   correlation - Correlation between predicted and actual voltage
            
            if obj.verbose
                fprintf('Estimating linear filter using FFT method\n');
            end
            
            try
                % Preprocess signals for filter estimation
                current_preprocessed = current_data - mean(current_data);
                voltage_preprocessed = voltage_data - mean(voltage_data);
                
                % Use analysis config parameters
                n_trials_filter = obj.analysis_config.n_trials_filter;
                regularization = obj.analysis_config.regularization;
                max_lag_ms = obj.analysis_config.max_lag_ms;
                
                % Estimate linear filter using regularized FFT method
                [linear_filter, lag, predicted_voltage, correlation] = estimate_filter_fft_trials_regularized(...
                    current_preprocessed, voltage_preprocessed, dt, n_trials_filter, false, [], ...
                    regularization, max_lag_ms, []);
                
                if obj.verbose
                    fprintf('  Linear filter estimation complete. Correlation r = %.3f\n', correlation);
                end
                
            catch ME
                if obj.verbose
                    fprintf('  ERROR: Failed to estimate linear filter: %s\n', ME.message);
                end
                
                linear_filter = [];
                correlation = NaN;
            end
        end
        
        function waveform_analysis = analyzeSpikeWaveforms(obj, voltage_data, spike_indices, dt)
            % ANALYZESPIKEWAVEFORMS - Analyze individual spike waveforms
            % Input:
            %   voltage_data - Voltage signal
            %   spike_indices - Indices of detected spikes
            %   dt - Sampling interval
            % Output:
            %   waveform_analysis - Structure containing waveform analysis
            
            waveform_analysis = struct();
            
            if obj.verbose
                fprintf('Analyzing spike waveforms\n');
            end
            
            try
                % Parameters for spike alignment and analysis
                pre_spike_ms = obj.analysis_config.spike_pre_window_ms;
                post_spike_ms = obj.analysis_config.spike_post_window_ms;
                pre_pts = round(pre_spike_ms / 1000 / dt);
                post_pts = round(post_spike_ms / 1000 / dt);
                
                % Extract and align individual spike waveforms
                all_spikes_aligned = [];
                
                for k = 1:length(spike_indices)
                    spike_idx = spike_indices(k);
                    
                    % Find local maximum within reasonable window after threshold crossing
                    search_end = min(spike_idx + post_pts, length(voltage_data));
                    search_window = spike_idx : search_end;
                    
                    if length(search_window) > 1
                        [~, peak_rel_idx] = max(voltage_data(search_window));
                        peak_idx = spike_idx + peak_rel_idx - 1;
                        
                        % Extract spike waveform centered on peak
                        start_idx = peak_idx - pre_pts;
                        end_idx = peak_idx + post_pts;
                        
                        if start_idx >= 1 && end_idx <= length(voltage_data)
                            spike_waveform = voltage_data(start_idx : end_idx);
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
                    else
                        width_ms = NaN;
                    end
                    
                    waveform_analysis.avg_spike_waveform = avg_spike_waveform;
                    waveform_analysis.avg_width_ms = width_ms;
                    waveform_analysis.n_aligned_spikes = size(all_spikes_aligned, 2);
                    waveform_analysis.success = true;
                    
                    if obj.verbose
                        fprintf('  Analyzed %d aligned spike waveforms\n', size(all_spikes_aligned, 2));
                        fprintf('  Average spike width: %.2f ms\n', width_ms);
                    end
                else
                    waveform_analysis.success = false;
                    waveform_analysis.error = 'No valid spike waveforms found';
                    waveform_analysis.avg_width_ms = NaN;
                    waveform_analysis.avg_spike_waveform = [];
                    waveform_analysis.n_aligned_spikes = 0;
                end
                
            catch ME
                if obj.verbose
                    fprintf('  ERROR: Failed to analyze spike waveforms: %s\n', ME.message);
                end
                
                waveform_analysis.success = false;
                waveform_analysis.error = ME;
                waveform_analysis.avg_width_ms = NaN;
                waveform_analysis.avg_spike_waveform = [];
                waveform_analysis.n_aligned_spikes = 0;
            end
        end
        
        function result = compileResults(obj, result, frequencyNode)
            % COMPILERESULTS - Compile all analysis results into final structure
            % Input:
            %   result - Analysis result structure
            %   frequencyNode - Original frequency node
            % Output:
            %   result - Compiled result structure
            
            % Add node information
            result.cell_id = frequencyNode.cell;
            result.protocol = frequencyNode.protocol;
            result.date = frequencyNode.date;
            result.group = frequencyNode.group;
            result.frequency = frequencyNode.frequency;
            result.selected_sd_value = result.selected_sd.value;
            result.n_epochs = result.selected_sd.max_epochs;
            
            % Add analysis metadata
            result.analysis_timestamp = datetime('now');
            result.analysis_version = '1.0';
            result.analysis_config = obj.analysis_config;
            result.spike_config = obj.spike_config;
            
            % Store results at the node level for later retrieval
            try
                result.selected_sd.node.custom.put('analysis_results', riekesuite.util.toJavaMap(result));
                if obj.verbose
                    fprintf('Results stored at node level\n');
                end
            catch ME
                if obj.verbose
                    fprintf('Warning: Failed to store results at node level: %s\n', ME.message);
                end
            end
        end
    end
end 