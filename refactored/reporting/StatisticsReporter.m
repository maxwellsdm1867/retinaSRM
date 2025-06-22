classdef StatisticsReporter
    % STATISTICSREPORTER - Class for generating statistical reports
    % This class handles the generation of statistical analysis reports,
    % including descriptive statistics, trend analysis, and statistical
    % summaries.
    %
    % The class provides methods to:
    % - Generate descriptive statistics
    % - Perform trend analysis
    % - Create statistical summaries
    % - Export statistical data
    % - Generate statistical visualizations
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
        function obj = StatisticsReporter(analysis_results, verbose)
            % STATISTICSREPORTER Constructor
            % Input:
            %   analysis_results - Structure containing analysis results
            %   verbose - Verbosity level (default: true)
            % Output:
            %   obj - StatisticsReporter instance
            
            obj.analysis_results = analysis_results;
            
            if nargin >= 2
                obj.verbose = verbose;
            end
        end
        
        function generateStatistics(obj)
            % GENERATESTATISTICS - Generate comprehensive statistical report
            % Creates a detailed statistical analysis of all results including
            % descriptive statistics, distributions, and trend analysis.
            
            if obj.verbose
                fprintf('\n=== GENERATING STATISTICAL REPORT ===\n');
            end
            
            try
                % Extract statistical data
                statistical_data = obj.extractStatisticalData();
                
                % Compute descriptive statistics
                descriptive_stats = obj.computeDescriptiveStatistics(statistical_data);
                
                % Perform trend analysis
                trend_analysis = obj.performTrendAnalysis(statistical_data);
                
                % Generate distribution analysis
                distribution_analysis = obj.analyzeDistributions(statistical_data);
                
                % Create correlation analysis
                correlation_analysis = obj.performCorrelationAnalysis(statistical_data);
                
                % Display statistics
                obj.displayStatistics(descriptive_stats, trend_analysis, distribution_analysis, correlation_analysis);
                
                % Save statistical report
                statistical_report = obj.createStatisticalReport(descriptive_stats, trend_analysis, distribution_analysis, correlation_analysis);
                obj.saveStatisticalReport(statistical_report);
                
                if obj.verbose
                    fprintf('Statistical report generated successfully\n');
                end
                
            catch ME
                if obj.verbose
                    fprintf('Failed to generate statistics: %s\n', ME.message);
                end
            end
        end
        
        function statistical_data = extractStatisticalData(obj)
            % EXTRACTSTATISTICALDATA - Extract data for statistical analysis
            % Output:
            %   statistical_data - Structure containing extracted data
            
            statistical_data = struct();
            
            successful_analyses = obj.analysis_results.successful;
            
            % Initialize data arrays
            firing_rates = [];
            filter_correlations = [];
            spike_counts = [];
            spike_widths = [];
            durations = [];
            trial_counts = [];
            frequencies = [];
            cell_ids = {};
            protocols = {};
            
            % Extract data from successful analyses
            for i = 1:length(successful_analyses)
                result = successful_analyses{i};
                
                % Basic metrics
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
                
                if isfield(result, 'data_metadata')
                    durations(end+1) = result.data_metadata.total_duration;
                    trial_counts(end+1) = result.data_metadata.n_trials;
                end
                
                % Categorical data
                if isfield(result, 'frequency')
                    frequencies(end+1) = str2double(result.frequency);
                end
                
                if isfield(result, 'cell_id')
                    cell_ids{end+1} = result.cell_id;
                end
                
                if isfield(result, 'protocol')
                    protocols{end+1} = result.protocol;
                end
            end
            
            % Store extracted data
            statistical_data.firing_rates = firing_rates;
            statistical_data.filter_correlations = filter_correlations;
            statistical_data.spike_counts = spike_counts;
            statistical_data.spike_widths = spike_widths;
            statistical_data.durations = durations;
            statistical_data.trial_counts = trial_counts;
            statistical_data.frequencies = frequencies;
            statistical_data.cell_ids = cell_ids;
            statistical_data.protocols = protocols;
            
            statistical_data.n_samples = length(firing_rates);
        end
        
        function descriptive_stats = computeDescriptiveStatistics(obj, statistical_data)
            % COMPUTEDESCRIPTIVESTATISTICS - Compute descriptive statistics
            % Input:
            %   statistical_data - Structure containing statistical data
            % Output:
            %   descriptive_stats - Structure containing descriptive statistics
            
            descriptive_stats = struct();
            
            % Firing rate statistics
            if ~isempty(statistical_data.firing_rates)
                fr = statistical_data.firing_rates;
                descriptive_stats.firing_rate = struct(...
                    'mean', mean(fr), ...
                    'std', std(fr), ...
                    'median', median(fr), ...
                    'min', min(fr), ...
                    'max', max(fr), ...
                    'range', range(fr), ...
                    'skewness', skewness(fr), ...
                    'kurtosis', kurtosis(fr), ...
                    'n', length(fr) ...
                );
            end
            
            % Filter correlation statistics
            if ~isempty(statistical_data.filter_correlations)
                fc = statistical_data.filter_correlations;
                descriptive_stats.filter_correlation = struct(...
                    'mean', mean(fc), ...
                    'std', std(fc), ...
                    'median', median(fc), ...
                    'min', min(fc), ...
                    'max', max(fc), ...
                    'range', range(fc), ...
                    'skewness', skewness(fc), ...
                    'kurtosis', kurtosis(fc), ...
                    'n', length(fc) ...
                );
            end
            
            % Spike count statistics
            if ~isempty(statistical_data.spike_counts)
                sc = statistical_data.spike_counts;
                descriptive_stats.spike_count = struct(...
                    'mean', mean(sc), ...
                    'std', std(sc), ...
                    'median', median(sc), ...
                    'min', min(sc), ...
                    'max', max(sc), ...
                    'range', range(sc), ...
                    'skewness', skewness(sc), ...
                    'kurtosis', kurtosis(sc), ...
                    'n', length(sc) ...
                );
            end
            
            % Spike width statistics
            if ~isempty(statistical_data.spike_widths)
                sw = statistical_data.spike_widths;
                descriptive_stats.spike_width = struct(...
                    'mean', mean(sw), ...
                    'std', std(sw), ...
                    'median', median(sw), ...
                    'min', min(sw), ...
                    'max', max(sw), ...
                    'range', range(sw), ...
                    'skewness', skewness(sw), ...
                    'kurtosis', kurtosis(sw), ...
                    'n', length(sw) ...
                );
            end
            
            % Duration statistics
            if ~isempty(statistical_data.durations)
                dur = statistical_data.durations;
                descriptive_stats.duration = struct(...
                    'mean', mean(dur), ...
                    'std', std(dur), ...
                    'median', median(dur), ...
                    'min', min(dur), ...
                    'max', max(dur), ...
                    'range', range(dur), ...
                    'n', length(dur) ...
                );
            end
            
            % Trial count statistics
            if ~isempty(statistical_data.trial_counts)
                tc = statistical_data.trial_counts;
                descriptive_stats.trial_count = struct(...
                    'mean', mean(tc), ...
                    'std', std(tc), ...
                    'median', median(tc), ...
                    'min', min(tc), ...
                    'max', max(tc), ...
                    'range', range(tc), ...
                    'n', length(tc) ...
                );
            end
            
            % Frequency statistics
            if ~isempty(statistical_data.frequencies)
                freq = statistical_data.frequencies;
                descriptive_stats.frequency = struct(...
                    'mean', mean(freq), ...
                    'std', std(freq), ...
                    'median', median(freq), ...
                    'min', min(freq), ...
                    'max', max(freq), ...
                    'range', range(freq), ...
                    'unique_values', unique(freq), ...
                    'n', length(freq) ...
                );
            end
            
            % Categorical statistics
            if ~isempty(statistical_data.cell_ids)
                descriptive_stats.cell_analysis = obj.analyzeCategoricalData(statistical_data.cell_ids);
            end
            
            if ~isempty(statistical_data.protocols)
                descriptive_stats.protocol_analysis = obj.analyzeCategoricalData(statistical_data.protocols);
            end
        end
        
        function categorical_analysis = analyzeCategoricalData(obj, categorical_data)
            % ANALYZECATEGORICALDATA - Analyze categorical data
            % Input:
            %   categorical_data - Cell array of categorical data
            % Output:
            %   categorical_analysis - Structure containing categorical analysis
            
            categorical_analysis = struct();
            
            % Get unique values and counts
            unique_values = unique(categorical_data);
            counts = zeros(length(unique_values), 1);
            
            for i = 1:length(unique_values)
                counts(i) = sum(strcmp(categorical_data, unique_values{i}));
            end
            
            categorical_analysis.unique_values = unique_values;
            categorical_analysis.counts = counts;
            categorical_analysis.frequencies = counts / length(categorical_data);
            categorical_analysis.n_categories = length(unique_values);
            categorical_analysis.total_count = length(categorical_data);
        end
        
        function trend_analysis = performTrendAnalysis(obj, statistical_data)
            % PERFORMTRENDANALYSIS - Perform trend analysis
            % Input:
            %   statistical_data - Structure containing statistical data
            % Output:
            %   trend_analysis - Structure containing trend analysis
            
            trend_analysis = struct();
            
            % Frequency vs firing rate trend
            if length(statistical_data.frequencies) == length(statistical_data.firing_rates) && ...
                    length(statistical_data.frequencies) >= 2
                [slope, intercept, r2] = obj.computeLinearTrend(...
                    statistical_data.frequencies, statistical_data.firing_rates);
                
                trend_analysis.frequency_vs_firing_rate = struct(...
                    'slope', slope, ...
                    'intercept', intercept, ...
                    'r2', r2, ...
                    'n_points', length(statistical_data.frequencies) ...
                );
            end
            
            % Frequency vs filter correlation trend
            if length(statistical_data.frequencies) == length(statistical_data.filter_correlations) && ...
                    length(statistical_data.filter_correlations) >= 2
                [slope, intercept, r2] = obj.computeLinearTrend(...
                    statistical_data.frequencies, statistical_data.filter_correlations);
                
                trend_analysis.frequency_vs_filter_correlation = struct(...
                    'slope', slope, ...
                    'intercept', intercept, ...
                    'r2', r2, ...
                    'n_points', length(statistical_data.filter_correlations) ...
                );
            end
            
            % Duration vs firing rate trend
            if length(statistical_data.durations) == length(statistical_data.firing_rates) && ...
                    length(statistical_data.durations) >= 2
                [slope, intercept, r2] = obj.computeLinearTrend(...
                    statistical_data.durations, statistical_data.firing_rates);
                
                trend_analysis.duration_vs_firing_rate = struct(...
                    'slope', slope, ...
                    'intercept', intercept, ...
                    'r2', r2, ...
                    'n_points', length(statistical_data.durations) ...
                );
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
        
        function distribution_analysis = analyzeDistributions(obj, statistical_data)
            % ANALYZEDISTRIBUTIONS - Analyze data distributions
            % Input:
            %   statistical_data - Structure containing statistical data
            % Output:
            %   distribution_analysis - Structure containing distribution analysis
            
            distribution_analysis = struct();
            
            % Firing rate distribution
            if ~isempty(statistical_data.firing_rates)
                fr = statistical_data.firing_rates;
                distribution_analysis.firing_rate = struct(...
                    'percentiles', prctile(fr, [5, 25, 50, 75, 95]), ...
                    'iqr', iqr(fr), ...
                    'outliers', obj.detectOutliers(fr), ...
                    'normality_test', obj.testNormality(fr) ...
                );
            end
            
            % Filter correlation distribution
            if ~isempty(statistical_data.filter_correlations)
                fc = statistical_data.filter_correlations;
                distribution_analysis.filter_correlation = struct(...
                    'percentiles', prctile(fc, [5, 25, 50, 75, 95]), ...
                    'iqr', iqr(fc), ...
                    'outliers', obj.detectOutliers(fc), ...
                    'normality_test', obj.testNormality(fc) ...
                );
            end
            
            % Spike count distribution
            if ~isempty(statistical_data.spike_counts)
                sc = statistical_data.spike_counts;
                distribution_analysis.spike_count = struct(...
                    'percentiles', prctile(sc, [5, 25, 50, 75, 95]), ...
                    'iqr', iqr(sc), ...
                    'outliers', obj.detectOutliers(sc), ...
                    'normality_test', obj.testNormality(sc) ...
                );
            end
        end
        
        function outliers = detectOutliers(obj, data)
            % DETECTOUTLIERS - Detect outliers using IQR method
            % Input:
            %   data - Data array
            % Output:
            %   outliers - Structure containing outlier information
            
            outliers = struct();
            
            q1 = prctile(data, 25);
            q3 = prctile(data, 75);
            iqr_val = q3 - q1;
            
            lower_bound = q1 - 1.5 * iqr_val;
            upper_bound = q3 + 1.5 * iqr_val;
            
            outlier_idx = data < lower_bound | data > upper_bound;
            outlier_values = data(outlier_idx);
            
            outliers.lower_bound = lower_bound;
            outliers.upper_bound = upper_bound;
            outliers.outlier_values = outlier_values;
            outliers.outlier_indices = find(outlier_idx);
            outliers.n_outliers = length(outlier_values);
            outliers.outlier_percentage = length(outlier_values) / length(data) * 100;
        end
        
        function normality_test = testNormality(obj, data)
            % TESTNORMALITY - Test for normality using Lilliefors test
            % Input:
            %   data - Data array
            % Output:
            %   normality_test - Structure containing normality test results
            
            normality_test = struct();
            
            try
                [h, p_value, k_statistic, critical_value] = lillietest(data);
                
                normality_test.hypothesis_test = h;
                normality_test.p_value = p_value;
                normality_test.k_statistic = k_statistic;
                normality_test.critical_value = critical_value;
                normality_test.is_normal = ~h; % h=0 means normal distribution
                normality_test.test_name = 'Lilliefors test';
                
            catch ME
                normality_test.error = ME.message;
                normality_test.is_normal = NaN;
            end
        end
        
        function correlation_analysis = performCorrelationAnalysis(obj, statistical_data)
            % PERFORMCORRELATIONANALYSIS - Perform correlation analysis
            % Input:
            %   statistical_data - Structure containing statistical data
            % Output:
            %   correlation_analysis - Structure containing correlation analysis
            
            correlation_analysis = struct();
            
            % Create correlation matrix
            variables = {};
            data_matrix = [];
            
            if ~isempty(statistical_data.firing_rates)
                variables{end+1} = 'firing_rate';
                data_matrix = [data_matrix, statistical_data.firing_rates(:)];
            end
            
            if ~isempty(statistical_data.filter_correlations)
                variables{end+1} = 'filter_correlation';
                data_matrix = [data_matrix, statistical_data.filter_correlations(:)];
            end
            
            if ~isempty(statistical_data.spike_counts)
                variables{end+1} = 'spike_count';
                data_matrix = [data_matrix, statistical_data.spike_counts(:)];
            end
            
            if ~isempty(statistical_data.spike_widths)
                variables{end+1} = 'spike_width';
                data_matrix = [data_matrix, statistical_data.spike_widths(:)];
            end
            
            if ~isempty(statistical_data.durations)
                variables{end+1} = 'duration';
                data_matrix = [data_matrix, statistical_data.durations(:)];
            end
            
            if size(data_matrix, 2) >= 2
                % Compute correlation matrix
                correlation_matrix = corrcoef(data_matrix, 'rows', 'complete');
                
                correlation_analysis.variables = variables;
                correlation_analysis.correlation_matrix = correlation_matrix;
                correlation_analysis.n_variables = length(variables);
                
                % Extract significant correlations
                significant_correlations = obj.findSignificantCorrelations(correlation_matrix, variables);
                correlation_analysis.significant_correlations = significant_correlations;
            end
        end
        
        function significant_correlations = findSignificantCorrelations(obj, correlation_matrix, variables)
            % FINDSIGNIFICANTCORRELATIONS - Find significant correlations
            % Input:
            %   correlation_matrix - Correlation matrix
            %   variables - Variable names
            % Output:
            %   significant_correlations - Structure containing significant correlations
            
            significant_correlations = {};
            correlation_idx = 1;
            
            for i = 1:size(correlation_matrix, 1)
                for j = i+1:size(correlation_matrix, 2)
                    correlation_value = correlation_matrix(i, j);
                    
                    % Consider correlations with absolute value > 0.3 as significant
                    if abs(correlation_value) > 0.3
                        significant_correlations{correlation_idx} = struct(...
                            'variable1', variables{i}, ...
                            'variable2', variables{j}, ...
                            'correlation', correlation_value, ...
                            'strength', obj.categorizeCorrelationStrength(correlation_value) ...
                        );
                        correlation_idx = correlation_idx + 1;
                    end
                end
            end
        end
        
        function strength = categorizeCorrelationStrength(obj, correlation_value)
            % CATEGORIZECORRELATIONSTRENGTH - Categorize correlation strength
            % Input:
            %   correlation_value - Correlation coefficient
            % Output:
            %   strength - Strength category string
            
            abs_corr = abs(correlation_value);
            
            if abs_corr >= 0.7
                strength = 'strong';
            elseif abs_corr >= 0.5
                strength = 'moderate';
            elseif abs_corr >= 0.3
                strength = 'weak';
            else
                strength = 'negligible';
            end
        end
        
        function displayStatistics(obj, descriptive_stats, trend_analysis, distribution_analysis, correlation_analysis)
            % DISPLAYSTATISTICS - Display statistical results
            % Input:
            %   descriptive_stats - Descriptive statistics structure
            %   trend_analysis - Trend analysis structure
            %   distribution_analysis - Distribution analysis structure
            %   correlation_analysis - Correlation analysis structure
            
            fprintf('\n===============================================\n');
            fprintf('STATISTICAL ANALYSIS REPORT\n');
            fprintf('===============================================\n');
            
            % Display descriptive statistics
            fprintf('\nDescriptive Statistics:\n');
            obj.displayDescriptiveStatistics(descriptive_stats);
            
            % Display trend analysis
            fprintf('\nTrend Analysis:\n');
            obj.displayTrendAnalysis(trend_analysis);
            
            % Display distribution analysis
            fprintf('\nDistribution Analysis:\n');
            obj.displayDistributionAnalysis(distribution_analysis);
            
            % Display correlation analysis
            fprintf('\nCorrelation Analysis:\n');
            obj.displayCorrelationAnalysis(correlation_analysis);
            
            fprintf('\n===============================================\n');
        end
        
        function displayDescriptiveStatistics(obj, descriptive_stats)
            % DISPLAYDESCRIPTIVESTATISTICS - Display descriptive statistics
            % Input:
            %   descriptive_stats - Descriptive statistics structure
            
            fields = fieldnames(descriptive_stats);
            
            for i = 1:length(fields)
                field = fields{i};
                if isstruct(descriptive_stats.(field))
                    stats = descriptive_stats.(field);
                    
                    fprintf('  %s:\n', field);
                    fprintf('    Mean: %.3f ± %.3f\n', stats.mean, stats.std);
                    fprintf('    Median: %.3f\n', stats.median);
                    fprintf('    Range: [%.3f, %.3f]\n', stats.min, stats.max);
                    fprintf('    N: %d\n', stats.n);
                end
            end
        end
        
        function displayTrendAnalysis(obj, trend_analysis)
            % DISPLAYTRENDANALYSIS - Display trend analysis
            % Input:
            %   trend_analysis - Trend analysis structure
            
            fields = fieldnames(trend_analysis);
            
            for i = 1:length(fields)
                field = fields{i};
                if isstruct(trend_analysis.(field))
                    trend = trend_analysis.(field);
                    
                    fprintf('  %s:\n', field);
                    fprintf('    Slope: %.3f\n', trend.slope);
                    fprintf('    Intercept: %.3f\n', trend.intercept);
                    fprintf('    R²: %.3f\n', trend.r2);
                    fprintf('    N points: %d\n', trend.n_points);
                end
            end
        end
        
        function displayDistributionAnalysis(obj, distribution_analysis)
            % DISPLAYDISTRIBUTIONANALYSIS - Display distribution analysis
            % Input:
            %   distribution_analysis - Distribution analysis structure
            
            fields = fieldnames(distribution_analysis);
            
            for i = 1:length(fields)
                field = fields{i};
                if isstruct(distribution_analysis.(field))
                    dist = distribution_analysis.(field);
                    
                    fprintf('  %s:\n', field);
                    fprintf('    IQR: %.3f\n', dist.iqr);
                    fprintf('    Outliers: %d (%.1f%%)\n', dist.outliers.n_outliers, dist.outliers.outlier_percentage);
                    fprintf('    Normal distribution: %s\n', mat2str(dist.normality_test.is_normal));
                end
            end
        end
        
        function displayCorrelationAnalysis(obj, correlation_analysis)
            % DISPLAYCORRELATIONANALYSIS - Display correlation analysis
            % Input:
            %   correlation_analysis - Correlation analysis structure
            
            if isfield(correlation_analysis, 'significant_correlations')
                significant_correlations = correlation_analysis.significant_correlations;
                
                if ~isempty(significant_correlations)
                    fprintf('  Significant correlations:\n');
                    for i = 1:length(significant_correlations)
                        corr = significant_correlations{i};
                        fprintf('    %s vs %s: %.3f (%s)\n', ...
                            corr.variable1, corr.variable2, corr.correlation, corr.strength);
                    end
                else
                    fprintf('  No significant correlations found\n');
                end
            end
        end
        
        function statistical_report = createStatisticalReport(obj, descriptive_stats, trend_analysis, distribution_analysis, correlation_analysis)
            % CREATESTATISTICALREPORT - Create comprehensive statistical report
            % Input:
            %   descriptive_stats - Descriptive statistics structure
            %   trend_analysis - Trend analysis structure
            %   distribution_analysis - Distribution analysis structure
            %   correlation_analysis - Correlation analysis structure
            % Output:
            %   statistical_report - Structure containing statistical report
            
            statistical_report = struct();
            statistical_report.timestamp = datetime('now');
            statistical_report.descriptive_statistics = descriptive_stats;
            statistical_report.trend_analysis = trend_analysis;
            statistical_report.distribution_analysis = distribution_analysis;
            statistical_report.correlation_analysis = correlation_analysis;
            statistical_report.success = true;
        end
        
        function saveStatisticalReport(obj, statistical_report)
            % SAVESTATISTICALREPORT - Save statistical report to file
            % Input:
            %   statistical_report - Statistical report structure
            
            try
                % Create filename with timestamp
                timestamp_str = datestr(statistical_report.timestamp, 'yyyymmdd_HHMMSS');
                filename = sprintf('statistical_report_%s.mat', timestamp_str);
                
                % Save report
                save(filename, 'statistical_report');
                
                if obj.verbose
                    fprintf('Statistical report saved to: %s\n', filename);
                end
                
            catch ME
                if obj.verbose
                    fprintf('Failed to save statistical report: %s\n', ME.message);
                end
            end
        end
    end
end 