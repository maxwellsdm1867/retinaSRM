classdef FrequencyAnalyzer
    % FREQUENCYANALYZER - Class for frequency-specific analysis
    % This class handles analysis that is specific to frequency nodes,
    % including frequency-dependent parameter estimation and cross-frequency
    % comparisons.
    %
    % The class provides methods to:
    % - Analyze frequency-dependent responses
    % - Compare responses across different frequencies
    % - Extract frequency-specific parameters
    % - Generate frequency-dependent statistics
    %
    % Properties:
    %   verbose - Verbosity level for logging
    %
    % Author: Maxwell
    % Date: 2024
    
    properties
        verbose = true
    end
    
    methods
        function obj = FrequencyAnalyzer(verbose)
            % FREQUENCYANALYZER Constructor
            % Input:
            %   verbose - Verbosity level (default: true)
            % Output:
            %   obj - FrequencyAnalyzer instance
            
            if nargin >= 1
                obj.verbose = verbose;
            end
        end
        
        function frequency_analysis = analyzeFrequencyResponse(obj, frequencyNode, analysis_results)
            % ANALYZEFREQUENCYRESPONSE - Analyze frequency-specific response
            % Input:
            %   frequencyNode - Frequency node structure
            %   analysis_results - Results from basic analysis
            % Output:
            %   frequency_analysis - Structure containing frequency-specific analysis
            
            if obj.verbose
                fprintf('Performing frequency-specific analysis for %s Hz\n', frequencyNode.frequency);
            end
            
            frequency_analysis = struct();
            frequency_analysis.frequency = str2double(frequencyNode.frequency);
            frequency_analysis.timestamp = datetime('now');
            
            try
                % Extract frequency-specific parameters
                frequency_analysis.response_characteristics = obj.extractResponseCharacteristics(analysis_results);
                frequency_analysis.frequency_dependent_params = obj.extractFrequencyDependentParams(analysis_results);
                frequency_analysis.quality_metrics = obj.computeFrequencyQualityMetrics(analysis_results);
                
                frequency_analysis.success = true;
                
            catch ME
                if obj.verbose
                    fprintf('Frequency analysis failed: %s\n', ME.message);
                end
                
                frequency_analysis.error = ME;
                frequency_analysis.success = false;
            end
        end
        
        function response_characteristics = extractResponseCharacteristics(obj, analysis_results)
            % EXTRACTRESPONSECHARACTERISTICS - Extract response characteristics
            % Input:
            %   analysis_results - Analysis results structure
            % Output:
            %   response_characteristics - Structure containing response characteristics
            
            response_characteristics = struct();
            
            if ~analysis_results.success
                response_characteristics.success = false;
                return;
            end
            
            % Extract basic response characteristics
            spike_stats = analysis_results.spike_stats;
            response_characteristics.firing_rate = spike_stats.firing_rate;
            response_characteristics.n_spikes = spike_stats.n_spikes;
            response_characteristics.mean_isi = spike_stats.mean_isi;
            response_characteristics.cv_isi = spike_stats.cv_isi;
            
            % Extract filter characteristics
            if isfield(analysis_results, 'filter_correlation')
                response_characteristics.filter_correlation = analysis_results.filter_correlation;
            end
            
            % Extract waveform characteristics
            if isfield(analysis_results, 'spike_waveform_analysis') && ...
                    analysis_results.spike_waveform_analysis.success
                waveform_analysis = analysis_results.spike_waveform_analysis;
                response_characteristics.spike_width = waveform_analysis.avg_width_ms;
                response_characteristics.n_aligned_spikes = waveform_analysis.n_aligned_spikes;
            end
            
            response_characteristics.success = true;
        end
        
        function frequency_params = extractFrequencyDependentParams(obj, analysis_results)
            % EXTRACTFREQUENCYDEPENDENTPARAMS - Extract frequency-dependent parameters
            % Input:
            %   analysis_results - Analysis results structure
            % Output:
            %   frequency_params - Structure containing frequency-dependent parameters
            
            frequency_params = struct();
            
            if ~analysis_results.success
                frequency_params.success = false;
                return;
            end
            
            % Extract STA characteristics
            if isfield(analysis_results, 'sta_current') && ~isempty(analysis_results.sta_current)
                sta_current = analysis_results.sta_current;
                frequency_params.sta_amplitude = max(abs(sta_current));
                frequency_params.sta_peak_time = find(abs(sta_current) == max(abs(sta_current)), 1);
            end
            
            % Extract linear filter characteristics
            if isfield(analysis_results, 'linear_filter') && ~isempty(analysis_results.linear_filter)
                linear_filter = analysis_results.linear_filter;
                frequency_params.filter_amplitude = max(abs(linear_filter));
                frequency_params.filter_peak_time = find(abs(linear_filter) == max(abs(linear_filter)), 1);
            end
            
            frequency_params.success = true;
        end
        
        function quality_metrics = computeFrequencyQualityMetrics(obj, analysis_results)
            % COMPUTEFREQUENCYQUALITYMETRICS - Compute quality metrics for frequency analysis
            % Input:
            %   analysis_results - Analysis results structure
            % Output:
            %   quality_metrics - Structure containing quality metrics
            
            quality_metrics = struct();
            
            if ~analysis_results.success
                quality_metrics.success = false;
                return;
            end
            
            % Compute signal-to-noise ratio (simplified)
            if isfield(analysis_results, 'spike_stats')
                spike_stats = analysis_results.spike_stats;
                quality_metrics.snr_estimate = spike_stats.n_spikes / max(1, spike_stats.firing_rate);
            end
            
            % Compute regularity metric
            if isfield(analysis_results, 'spike_stats') && ~isnan(analysis_results.spike_stats.cv_isi)
                quality_metrics.regularity = 1 / (1 + analysis_results.spike_stats.cv_isi);
            else
                quality_metrics.regularity = NaN;
            end
            
            % Compute completeness metric
            quality_metrics.completeness = 0;
            if isfield(analysis_results, 'spike_stats')
                quality_metrics.completeness = quality_metrics.completeness + 1;
            end
            if isfield(analysis_results, 'sta_current')
                quality_metrics.completeness = quality_metrics.completeness + 1;
            end
            if isfield(analysis_results, 'linear_filter')
                quality_metrics.completeness = quality_metrics.completeness + 1;
            end
            if isfield(analysis_results, 'spike_waveform_analysis')
                quality_metrics.completeness = quality_metrics.completeness + 1;
            end
            quality_metrics.completeness = quality_metrics.completeness / 4; % Normalize to 0-1
            
            quality_metrics.success = true;
        end
        
        function comparison_result = compareFrequencies(obj, frequency_results)
            % COMPAREFREQUENCIES - Compare results across different frequencies
            % Input:
            %   frequency_results - Cell array of frequency analysis results
            % Output:
            %   comparison_result - Structure containing comparison results
            
            if obj.verbose
                fprintf('Comparing results across %d frequencies\n', length(frequency_results));
            end
            
            comparison_result = struct();
            comparison_result.timestamp = datetime('now');
            
            try
                % Extract frequencies and firing rates
                frequencies = [];
                firing_rates = [];
                filter_correlations = [];
                spike_widths = [];
                
                for i = 1:length(frequency_results)
                    if frequency_results{i}.success
                        frequencies(end+1) = frequency_results{i}.frequency;
                        
                        if isfield(frequency_results{i}, 'response_characteristics')
                            rc = frequency_results{i}.response_characteristics;
                            firing_rates(end+1) = rc.firing_rate;
                            
                            if isfield(rc, 'filter_correlation')
                                filter_correlations(end+1) = rc.filter_correlation;
                            end
                            
                            if isfield(rc, 'spike_width')
                                spike_widths(end+1) = rc.spike_width;
                            end
                        end
                    end
                end
                
                % Compute frequency-dependent trends
                if length(frequencies) >= 2
                    comparison_result.frequency_trends = obj.computeFrequencyTrends(...
                        frequencies, firing_rates, filter_correlations, spike_widths);
                end
                
                comparison_result.n_frequencies = length(frequencies);
                comparison_result.success = true;
                
            catch ME
                if obj.verbose
                    fprintf('Frequency comparison failed: %s\n', ME.message);
                end
                
                comparison_result.error = ME;
                comparison_result.success = false;
            end
        end
        
        function trends = computeFrequencyTrends(obj, frequencies, firing_rates, filter_correlations, spike_widths)
            % COMPUTEFREQUENCYTRENDS - Compute trends across frequencies
            % Input:
            %   frequencies - Array of frequencies
            %   firing_rates - Array of firing rates
            %   filter_correlations - Array of filter correlations
            %   spike_widths - Array of spike widths
            % Output:
            %   trends - Structure containing trend analysis
            
            trends = struct();
            
            % Firing rate vs frequency
            if length(frequencies) == length(firing_rates) && length(frequencies) >= 2
                [trends.firing_rate_slope, trends.firing_rate_intercept, trends.firing_rate_r2] = ...
                    obj.computeLinearTrend(frequencies, firing_rates);
            end
            
            % Filter correlation vs frequency
            if length(frequencies) == length(filter_correlations) && length(frequencies) >= 2
                [trends.filter_corr_slope, trends.filter_corr_intercept, trends.filter_corr_r2] = ...
                    obj.computeLinearTrend(frequencies, filter_correlations);
            end
            
            % Spike width vs frequency
            if length(frequencies) == length(spike_widths) && length(frequencies) >= 2
                [trends.spike_width_slope, trends.spike_width_intercept, trends.spike_width_r2] = ...
                    obj.computeLinearTrend(frequencies, spike_widths);
            end
        end
        
        function [slope, intercept, r2] = computeLinearTrend(obj, x, y)
            % COMPUTELINEARTREND - Compute linear trend between two variables
            % Input:
            %   x - Independent variable
            %   y - Dependent variable
            % Output:
            %   slope - Slope of linear fit
            %   intercept - Intercept of linear fit
            %   r2 - R-squared value
            
            % Remove NaN values
            valid_idx = ~isnan(x) & ~isnan(y);
            x_valid = x(valid_idx);
            y_valid = y(valid_idx);
            
            if length(x_valid) < 2
                slope = NaN;
                intercept = NaN;
                r2 = NaN;
                return;
            end
            
            % Compute linear fit
            p = polyfit(x_valid, y_valid, 1);
            slope = p(1);
            intercept = p(2);
            
            % Compute R-squared
            y_fit = polyval(p, x_valid);
            ss_res = sum((y_valid - y_fit).^2);
            ss_tot = sum((y_valid - mean(y_valid)).^2);
            r2 = 1 - (ss_res / ss_tot);
        end
        
        function summary = generateFrequencySummary(obj, frequency_results)
            % GENERATEFREQUENCYSUMMARY - Generate summary of frequency analysis
            % Input:
            %   frequency_results - Cell array of frequency analysis results
            % Output:
            %   summary - Structure containing summary information
            
            if obj.verbose
                fprintf('Generating frequency analysis summary\n');
            end
            
            summary = struct();
            summary.timestamp = datetime('now');
            summary.n_frequencies = length(frequency_results);
            
            % Count successful analyses
            successful_count = sum(cellfun(@(x) x.success, frequency_results));
            summary.successful_analyses = successful_count;
            summary.success_rate = successful_count / length(frequency_results);
            
            % Extract frequency range
            frequencies = [];
            for i = 1:length(frequency_results)
                if frequency_results{i}.success
                    frequencies(end+1) = frequency_results{i}.frequency;
                end
            end
            
            if ~isempty(frequencies)
                summary.frequency_range = [min(frequencies), max(frequencies)];
                summary.n_unique_frequencies = length(unique(frequencies));
            else
                summary.frequency_range = [NaN, NaN];
                summary.n_unique_frequencies = 0;
            end
            
            % Generate summary statistics
            if successful_count > 0
                summary = obj.computeSummaryStatistics(summary, frequency_results);
            end
            
            summary.success = true;
        end
        
        function summary = computeSummaryStatistics(obj, summary, frequency_results)
            % COMPUTESUMMARYSTATISTICS - Compute summary statistics
            % Input:
            %   summary - Summary structure to populate
            %   frequency_results - Frequency analysis results
            % Output:
            %   summary - Updated summary structure
            
            % Extract firing rates
            firing_rates = [];
            filter_correlations = [];
            spike_widths = [];
            
            for i = 1:length(frequency_results)
                if frequency_results{i}.success && isfield(frequency_results{i}, 'response_characteristics')
                    rc = frequency_results{i}.response_characteristics;
                    firing_rates(end+1) = rc.firing_rate;
                    
                    if isfield(rc, 'filter_correlation')
                        filter_correlations(end+1) = rc.filter_correlation;
                    end
                    
                    if isfield(rc, 'spike_width')
                        spike_widths(end+1) = rc.spike_width;
                    end
                end
            end
            
            % Compute statistics
            if ~isempty(firing_rates)
                summary.firing_rate_stats = struct(...
                    'mean', mean(firing_rates), ...
                    'std', std(firing_rates), ...
                    'min', min(firing_rates), ...
                    'max', max(firing_rates) ...
                );
            end
            
            if ~isempty(filter_correlations)
                summary.filter_correlation_stats = struct(...
                    'mean', mean(filter_correlations), ...
                    'std', std(filter_correlations), ...
                    'min', min(filter_correlations), ...
                    'max', max(filter_correlations) ...
                );
            end
            
            if ~isempty(spike_widths)
                summary.spike_width_stats = struct(...
                    'mean', mean(spike_widths), ...
                    'std', std(spike_widths), ...
                    'min', min(spike_widths), ...
                    'max', max(spike_widths) ...
                );
            end
        end
    end
end 