classdef AnalysisReporter
    % ANALYSISREPORTER - Class for generating comprehensive analysis reports
    % This class handles the generation of detailed reports from analysis
    % results, including summary statistics, quality metrics, and
    % performance indicators.
    %
    % The class provides methods to:
    % - Generate summary reports
    % - Create detailed analysis reports
    % - Export results to various formats
    % - Generate visualizations
    % - Track analysis performance
    %
    % Properties:
    %   analysis_results - Structure containing analysis results
    %   verbose - Verbosity level for logging
    %
    % Author: Maxwell
    % Date: 2024
    
    properties
        analysis_results
        verbose = true
    end
    
    methods
        function obj = AnalysisReporter(analysis_results, verbose)
            % ANALYSISREPORTER Constructor
            % Input:
            %   analysis_results - Structure containing analysis results
            %   verbose - Verbosity level (default: true)
            % Output:
            %   obj - AnalysisReporter instance
            
            obj.analysis_results = analysis_results;
            
            if nargin >= 2
                obj.verbose = verbose;
            end
        end
        
        function generateSummary(obj)
            % GENERATESUMMARY - Generate comprehensive analysis summary
            % Creates a detailed summary of all analysis results including
            % success rates, performance metrics, and key findings.
            
            if obj.verbose
                fprintf('\n=== GENERATING ANALYSIS SUMMARY ===\n');
            end
            
            try
                % Extract basic statistics
                summary_stats = obj.computeSummaryStatistics();
                
                % Display summary
                obj.displaySummary(summary_stats);
                
                % Generate detailed report
                detailed_report = obj.generateDetailedReport();
                
                % Save report to file
                obj.saveReport(detailed_report);
                
                if obj.verbose
                    fprintf('Analysis summary generated successfully\n');
                end
                
            catch ME
                if obj.verbose
                    fprintf('Failed to generate summary: %s\n', ME.message);
                end
            end
        end
        
        function summary_stats = computeSummaryStatistics(obj)
            % COMPUTESUMMARYSTATISTICS - Compute comprehensive summary statistics
            % Output:
            %   summary_stats - Structure containing summary statistics
            
            summary_stats = struct();
            
            % Basic counts
            successful_analyses = obj.analysis_results.successful;
            failed_analyses = obj.analysis_results.failed;
            
            summary_stats.total_analyses = length(successful_analyses) + length(failed_analyses);
            summary_stats.successful_count = length(successful_analyses);
            summary_stats.failed_count = length(failed_analyses);
            summary_stats.success_rate = summary_stats.successful_count / summary_stats.total_analyses;
            
            % Extract metadata
            if isfield(obj.analysis_results, 'metadata')
                summary_stats.metadata = obj.analysis_results.metadata;
            end
            
            % Compute performance metrics
            if summary_stats.successful_count > 0
                summary_stats.performance_metrics = obj.computePerformanceMetrics(successful_analyses);
            end
            
            % Compute quality metrics
            summary_stats.quality_metrics = obj.computeQualityMetrics(successful_analyses);
            
            % Compute error analysis
            if summary_stats.failed_count > 0
                summary_stats.error_analysis = obj.analyzeErrors(failed_analyses);
            end
            
            summary_stats.timestamp = datetime('now');
        end
        
        function performance_metrics = computePerformanceMetrics(obj, successful_analyses)
            % COMPUTEPERFORMANCEMETRICS - Compute performance metrics
            % Input:
            %   successful_analyses - Cell array of successful analysis results
            % Output:
            %   performance_metrics - Structure containing performance metrics
            
            performance_metrics = struct();
            
            % Extract firing rates
            firing_rates = [];
            filter_correlations = [];
            spike_counts = [];
            spike_widths = [];
            
            for i = 1:length(successful_analyses)
                result = successful_analyses{i};
                
                if isfield(result, 'spike_stats')
                    firing_rates(end+1) = result.spike_stats.firing_rate;
                    spike_counts(end+1) = result.spike_stats.n_spikes;
                end
                
                if isfield(result, 'filter_correlation')
                    filter_correlations(end+1) = result.filter_correlation;
                end
                
                if isfield(result, 'spike_waveform_analysis') && ...
                        result.spike_waveform_analysis.success
                    spike_widths(end+1) = result.spike_waveform_analysis.avg_width_ms;
                end
            end
            
            % Compute statistics
            if ~isempty(firing_rates)
                performance_metrics.firing_rate_stats = struct(...
                    'mean', mean(firing_rates), ...
                    'std', std(firing_rates), ...
                    'min', min(firing_rates), ...
                    'max', max(firing_rates), ...
                    'median', median(firing_rates) ...
                );
            end
            
            if ~isempty(filter_correlations)
                performance_metrics.filter_correlation_stats = struct(...
                    'mean', mean(filter_correlations), ...
                    'std', std(filter_correlations), ...
                    'min', min(filter_correlations), ...
                    'max', max(filter_correlations), ...
                    'median', median(filter_correlations) ...
                );
            end
            
            if ~isempty(spike_counts)
                performance_metrics.spike_count_stats = struct(...
                    'mean', mean(spike_counts), ...
                    'std', std(spike_counts), ...
                    'min', min(spike_counts), ...
                    'max', max(spike_counts), ...
                    'median', median(spike_counts) ...
                );
            end
            
            if ~isempty(spike_widths)
                performance_metrics.spike_width_stats = struct(...
                    'mean', mean(spike_widths), ...
                    'std', std(spike_widths), ...
                    'min', min(spike_widths), ...
                    'max', max(spike_widths), ...
                    'median', median(spike_widths) ...
                );
            end
        end
        
        function quality_metrics = computeQualityMetrics(obj, successful_analyses)
            % COMPUTEQUALITYMETRICS - Compute quality metrics
            % Input:
            %   successful_analyses - Cell array of successful analysis results
            % Output:
            %   quality_metrics - Structure containing quality metrics
            
            quality_metrics = struct();
            
            % Count quality warnings
            total_warnings = 0;
            warning_types = {};
            
            for i = 1:length(successful_analyses)
                result = successful_analyses{i};
                
                if isfield(result, 'quality_result') && isfield(result.quality_result, 'warnings')
                    warnings = result.quality_result.warnings;
                    total_warnings = total_warnings + length(warnings);
                    warning_types = [warning_types, warnings];
                end
            end
            
            quality_metrics.total_warnings = total_warnings;
            if length(successful_analyses) > 0
                quality_metrics.avg_warnings_per_analysis = total_warnings / length(successful_analyses);
            else
                quality_metrics.avg_warnings_per_analysis = 0;
            end
            
            % Analyze warning types
            if ~isempty(warning_types)
                unique_warnings = unique(warning_types);
                warning_counts = zeros(length(unique_warnings), 1);
                
                for i = 1:length(unique_warnings)
                    warning_counts(i) = sum(strcmp(warning_types, unique_warnings{i}));
                end
                
                quality_metrics.warning_analysis = struct(...
                    'types', unique_warnings, ...
                    'counts', warning_counts, ...
                    'frequencies', warning_counts / total_warnings ...
                );
            end
            
            % Compute data quality metrics
            quality_metrics.data_quality = obj.computeDataQualityMetrics(successful_analyses);
        end
        
        function data_quality = computeDataQualityMetrics(obj, successful_analyses)
            % COMPUTEDATAQUALITYMETRICS - Compute data quality metrics
            % Input:
            %   successful_analyses - Cell array of successful analysis results
            % Output:
            %   data_quality - Structure containing data quality metrics
            
            data_quality = struct();
            
            % Extract data characteristics
            durations = [];
            trial_counts = [];
            timepoint_counts = [];
            
            for i = 1:length(successful_analyses)
                result = successful_analyses{i};
                
                if isfield(result, 'data_metadata')
                    metadata = result.data_metadata;
                    durations(end+1) = metadata.total_duration;
                    trial_counts(end+1) = metadata.n_trials;
                    timepoint_counts(end+1) = metadata.n_timepoints;
                end
            end
            
            % Compute statistics
            if ~isempty(durations)
                data_quality.duration_stats = struct(...
                    'mean', mean(durations), ...
                    'std', std(durations), ...
                    'min', min(durations), ...
                    'max', max(durations) ...
                );
            end
            
            if ~isempty(trial_counts)
                data_quality.trial_count_stats = struct(...
                    'mean', mean(trial_counts), ...
                    'std', std(trial_counts), ...
                    'min', min(trial_counts), ...
                    'max', max(trial_counts) ...
                );
            end
            
            if ~isempty(timepoint_counts)
                data_quality.timepoint_count_stats = struct(...
                    'mean', mean(timepoint_counts), ...
                    'std', std(timepoint_counts), ...
                    'min', min(timepoint_counts), ...
                    'max', max(timepoint_counts) ...
                );
            end
        end
        
        function error_analysis = analyzeErrors(obj, failed_analyses)
            % ANALYZEERRORS - Analyze failed analyses
            % Input:
            %   failed_analyses - Cell array of failed analysis results
            % Output:
            %   error_analysis - Structure containing error analysis
            
            error_analysis = struct();
            
            % Extract error messages
            error_messages = {};
            error_types = {};
            
            for i = 1:length(failed_analyses)
                failed = failed_analyses{i};
                
                if isfield(failed, 'error')
                    error_messages{end+1} = failed.error.message;
                    
                    % Extract error type (first part of error message)
                    error_parts = strsplit(failed.error.message, ':');
                    if length(error_parts) > 0
                        error_types{end+1} = strtrim(error_parts{1});
                    end
                end
            end
            
            error_analysis.total_errors = length(error_messages);
            error_analysis.error_messages = error_messages;
            
            % Analyze error types
            if ~isempty(error_types)
                unique_error_types = unique(error_types);
                error_type_counts = zeros(length(unique_error_types), 1);
                
                for i = 1:length(unique_error_types)
                    error_type_counts(i) = sum(strcmp(error_types, unique_error_types{i}));
                end
                
                error_analysis.error_type_analysis = struct(...
                    'types', unique_error_types, ...
                    'counts', error_type_counts, ...
                    'frequencies', error_type_counts / length(error_types) ...
                );
            end
        end
        
        function displaySummary(obj, summary_stats)
            % DISPLAYSUMMARY - Display analysis summary
            % Input:
            %   summary_stats - Summary statistics structure
            
            fprintf('\n===============================================\n');
            fprintf('ANALYSIS SUMMARY REPORT\n');
            fprintf('===============================================\n');
            fprintf('Generated: %s\n', char(summary_stats.timestamp));
            
            % Basic statistics
            fprintf('\nBasic Statistics:\n');
            fprintf('  Total Analyses: %d\n', summary_stats.total_analyses);
            fprintf('  Successful: %d (%.1f%%)\n', ...
                summary_stats.successful_count, summary_stats.success_rate * 100);
            fprintf('  Failed: %d (%.1f%%)\n', ...
                summary_stats.failed_count, (1 - summary_stats.success_rate) * 100);
            
            % Performance metrics
            if isfield(summary_stats, 'performance_metrics')
                pm = summary_stats.performance_metrics;
                
                fprintf('\nPerformance Metrics:\n');
                if isfield(pm, 'firing_rate_stats')
                    fr = pm.firing_rate_stats;
                    fprintf('  Firing Rate: %.2f ± %.2f Hz (range: %.2f - %.2f)\n', ...
                        fr.mean, fr.std, fr.min, fr.max);
                end
                
                if isfield(pm, 'filter_correlation_stats')
                    fc = pm.filter_correlation_stats;
                    fprintf('  Filter Correlation: %.3f ± %.3f (range: %.3f - %.3f)\n', ...
                        fc.mean, fc.std, fc.min, fc.max);
                end
                
                if isfield(pm, 'spike_count_stats')
                    sc = pm.spike_count_stats;
                    fprintf('  Spike Count: %.1f ± %.1f (range: %d - %d)\n', ...
                        sc.mean, sc.std, sc.min, sc.max);
                end
            end
            
            % Quality metrics
            if isfield(summary_stats, 'quality_metrics')
                qm = summary_stats.quality_metrics;
                fprintf('\nQuality Metrics:\n');
                fprintf('  Total Warnings: %d\n', qm.total_warnings);
                fprintf('  Avg Warnings per Analysis: %.2f\n', qm.avg_warnings_per_analysis);
            end
            
            % Error analysis
            if isfield(summary_stats, 'error_analysis')
                ea = summary_stats.error_analysis;
                fprintf('\nError Analysis:\n');
                fprintf('  Total Errors: %d\n', ea.total_errors);
                
                if isfield(ea, 'error_type_analysis')
                    eta = ea.error_type_analysis;
                    fprintf('  Most Common Error Types:\n');
                    for i = 1:min(3, length(eta.types))
                        fprintf('    %s: %d (%.1f%%)\n', ...
                            eta.types{i}, eta.counts(i), eta.frequencies(i) * 100);
                    end
                end
            end
            
            fprintf('\n===============================================\n');
        end
        
        function detailed_report = generateDetailedReport(obj)
            % GENERATEDETAILEDREPORT - Generate detailed analysis report
            % Output:
            %   detailed_report - Structure containing detailed report
            
            detailed_report = struct();
            detailed_report.timestamp = datetime('now');
            detailed_report.analysis_results = obj.analysis_results;
            
            % Add summary statistics
            detailed_report.summary = obj.computeSummaryStatistics();
            
            % Add individual analysis details
            detailed_report.individual_analyses = obj.generateIndividualReports();
            
            % Add recommendations
            detailed_report.recommendations = obj.generateRecommendations();
            
            detailed_report.success = true;
        end
        
        function individual_reports = generateIndividualReports(obj)
            % GENERATEINDIVIDUALREPORTS - Generate individual analysis reports
            % Output:
            %   individual_reports - Cell array of individual reports
            
            individual_reports = {};
            
            % Process successful analyses
            for i = 1:length(obj.analysis_results.successful)
                result = obj.analysis_results.successful{i};
                report = obj.createIndividualReport(result, i, 'successful');
                individual_reports{end+1} = report;
            end
            
            % Process failed analyses
            for i = 1:length(obj.analysis_results.failed)
                failed = obj.analysis_results.failed{i};
                report = obj.createIndividualReport(failed, i, 'failed');
                individual_reports{end+1} = report;
            end
        end
        
        function report = createIndividualReport(obj, result, index, status)
            % CREATEINDIVIDUALREPORT - Create individual analysis report
            % Input:
            %   result - Analysis result or failed analysis
            %   index - Index of the analysis
            %   status - Status of the analysis ('successful' or 'failed')
            % Output:
            %   report - Individual report structure
            
            report = struct();
            report.index = index;
            report.status = status;
            report.timestamp = datetime('now');
            
            if strcmp(status, 'successful')
                report.cell_id = result.cell_id;
                report.protocol = result.protocol;
                report.frequency = result.frequency;
                report.selected_sd = result.selected_sd_value;
                report.n_epochs = result.n_epochs;
                
                if isfield(result, 'spike_stats')
                    report.n_spikes = result.spike_stats.n_spikes;
                    report.firing_rate = result.spike_stats.firing_rate;
                end
                
                if isfield(result, 'filter_correlation')
                    report.filter_correlation = result.filter_correlation;
                end
                
                if isfield(result, 'spike_waveform_analysis') && ...
                        result.spike_waveform_analysis.success
                    report.spike_width = result.spike_waveform_analysis.avg_width_ms;
                end
            else
                report.node_info = result.node;
                if isfield(result, 'error')
                    report.error_message = result.error.message;
                    report.error_id = result.error.identifier;
                end
            end
        end
        
        function recommendations = generateRecommendations(obj)
            % GENERATERECOMMENDATIONS - Generate recommendations based on analysis
            % Output:
            %   recommendations - Structure containing recommendations
            
            recommendations = struct();
            recommendations.timestamp = datetime('now');
            recommendations.items = {};
            
            % Analyze success rate
            success_rate = length(obj.analysis_results.successful) / ...
                (length(obj.analysis_results.successful) + length(obj.analysis_results.failed));
            
            if success_rate < 0.8
                recommendations.items{end+1} = 'Consider reviewing analysis parameters to improve success rate';
            end
            
            % Analyze quality warnings
            total_warnings = 0;
            for i = 1:length(obj.analysis_results.successful)
                result = obj.analysis_results.successful{i};
                if isfield(result, 'quality_result') && isfield(result.quality_result, 'warnings')
                    total_warnings = total_warnings + length(result.quality_result.warnings);
                end
            end
            
            if total_warnings > 0
                recommendations.items{end+1} = sprintf('Review quality warnings (%d total) to improve data quality', total_warnings);
            end
            
            % Analyze error patterns
            if length(obj.analysis_results.failed) > 0
                recommendations.items{end+1} = 'Investigate common error patterns in failed analyses';
            end
            
            recommendations.n_recommendations = length(recommendations.items);
        end
        
        function saveReport(obj, detailed_report)
            % SAVEREPORT - Save detailed report to file
            % Input:
            %   detailed_report - Detailed report structure
            
            try
                % Create filename with timestamp
                timestamp_str = datestr(detailed_report.timestamp, 'yyyymmdd_HHMMSS');
                filename = sprintf('analysis_report_%s.mat', timestamp_str);
                
                % Save report
                save(filename, 'detailed_report');
                
                if obj.verbose
                    fprintf('Detailed report saved to: %s\n', filename);
                end
                
            catch ME
                if obj.verbose
                    fprintf('Failed to save report: %s\n', ME.message);
                end
            end
        end
    end
end 