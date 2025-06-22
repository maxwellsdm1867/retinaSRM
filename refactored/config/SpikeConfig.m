classdef SpikeConfig
    % SPIKECONFIG - Configuration class for spike detection parameters
    % This class contains all parameters related to spike detection,
    % including voltage thresholds, derivative thresholds, and search
    % parameters for the enhanced elbow method.
    %
    % Properties:
    %   vm_thresh - Voltage threshold for spike detection (mV)
    %   d2v_thresh - Second derivative threshold
    %   search_back_ms - Search back window for spike initiation (ms)
    %   plot_flag - Whether to generate plots during detection
    %   elbow_thresh - Elbow detection threshold (mV)
    %   spike_thresh - Spike detection threshold (mV)
    %   min_dv_thresh - Minimum derivative threshold (mV/ms)
    %   time_to_peak_thresh - Maximum time to peak (ms)
    %
    % Author: Maxwell
    % Date: 2024
    
    properties
        % Basic spike detection parameters
        vm_thresh = -20              % Voltage threshold (mV)
        d2v_thresh = 50              % Second derivative threshold
        search_back_ms = 2           % Search back window (ms)
        plot_flag = false            % Suppress plots during batch processing
        
        % Enhanced elbow method parameters
        elbow_thresh = -65           % Elbow detection threshold (mV)
        spike_thresh = -10           % Spike detection threshold (mV)
        min_dv_thresh = 0.1          % Minimum derivative threshold (mV/ms)
        time_to_peak_thresh = 1.5    % Maximum time to peak (ms)
        
        % Quality control parameters
        min_spikes = 50              % Minimum spike count for analysis
        min_firing_rate = 0.5        % Minimum firing rate (Hz)
        max_firing_rate = 50         % Maximum firing rate (Hz)
        min_filter_correlation = 0.3 % Minimum linear filter correlation
        min_duration = 30            % Minimum recording duration (s)
    end
    
    methods
        function obj = SpikeConfig()
            % SPIKECONFIG Constructor
            % Creates a new SpikeConfig instance with default values
            % No input parameters required - uses default values
        end
        
        function search_back_samples = getSearchBackSamples(obj, dt)
            % GETSEARCHBACKSAMPLES - Convert search back time to samples
            % Input:
            %   dt - Sampling interval (seconds)
            % Output:
            %   search_back_samples - Number of samples for search back
            search_back_samples = round(obj.search_back_ms / 1000 / dt);
        end
        
        function time_to_peak_samples = getTimeToPeakSamples(obj, dt)
            % GETTIMETOPEAKSAMPLES - Convert time to peak to samples
            % Input:
            %   dt - Sampling interval (seconds)
            % Output:
            %   time_to_peak_samples - Number of samples for time to peak
            time_to_peak_samples = round(obj.time_to_peak_thresh / 1000 / dt);
        end
        
        function is_valid = validateSpikeCount(obj, spike_count, duration)
            % VALIDATESPIKECOUNT - Check if spike count meets quality criteria
            % Input:
            %   spike_count - Number of detected spikes
            %   duration - Recording duration (seconds)
            % Output:
            %   is_valid - Boolean indicating if spike count is valid
            
            if spike_count < obj.min_spikes
                is_valid = false;
                return;
            end
            
            firing_rate = spike_count / duration;
            if firing_rate < obj.min_firing_rate || firing_rate > obj.max_firing_rate
                is_valid = false;
                return;
            end
            
            is_valid = true;
        end
        
        function quality_warnings = getQualityWarnings(obj, spike_count, firing_rate, filter_correlation, duration)
            % GETQUALITYWARNINGS - Get quality warnings for analysis
            % Input:
            %   spike_count - Number of detected spikes
            %   firing_rate - Firing rate (Hz)
            %   filter_correlation - Linear filter correlation
            %   duration - Recording duration (seconds)
            % Output:
            %   quality_warnings - Cell array of warning messages
            
            quality_warnings = {};
            
            if spike_count < obj.min_spikes
                quality_warnings{end+1} = sprintf('low_spikes(%d)', spike_count);
            end
            
            if firing_rate < obj.min_firing_rate
                quality_warnings{end+1} = sprintf('low_rate(%.2f)', firing_rate);
            end
            
            if firing_rate > obj.max_firing_rate
                quality_warnings{end+1} = sprintf('high_rate(%.2f)', firing_rate);
            end
            
            if filter_correlation < obj.min_filter_correlation
                quality_warnings{end+1} = sprintf('low_corr(%.3f)', filter_correlation);
            end
            
            if duration < obj.min_duration
                quality_warnings{end+1} = sprintf('short_dur(%.1f)', duration);
            end
        end
        
        function passes_quality = passesQualityCheck(obj, spike_count, firing_rate, filter_correlation, duration)
            % PASSESQUALITYCHECK - Check if analysis passes quality criteria
            % Input:
            %   spike_count - Number of detected spikes
            %   firing_rate - Firing rate (Hz)
            %   filter_correlation - Linear filter correlation
            %   duration - Recording duration (seconds)
            % Output:
            %   passes_quality - Boolean indicating if quality check passes
            
            % Only filter correlation is a hard failure condition
            passes_quality = filter_correlation >= obj.min_filter_correlation;
        end
        
        function validate(obj)
            % VALIDATE - Validate spike detection parameters
            % Checks that all parameters are within reasonable bounds
            % Throws error if validation fails
            
            if obj.vm_thresh >= 0
                error('vm_thresh must be negative');
            end
            
            if obj.d2v_thresh <= 0
                error('d2v_thresh must be positive');
            end
            
            if obj.search_back_ms <= 0
                error('search_back_ms must be positive');
            end
            
            if obj.elbow_thresh >= obj.spike_thresh
                error('elbow_thresh must be less than spike_thresh');
            end
            
            if obj.min_dv_thresh <= 0
                error('min_dv_thresh must be positive');
            end
            
            if obj.time_to_peak_thresh <= 0
                error('time_to_peak_thresh must be positive');
            end
            
            if obj.min_spikes <= 0
                error('min_spikes must be positive');
            end
            
            if obj.min_firing_rate <= 0
                error('min_firing_rate must be positive');
            end
            
            if obj.max_firing_rate <= obj.min_firing_rate
                error('max_firing_rate must be greater than min_firing_rate');
            end
            
            if obj.min_filter_correlation < 0 || obj.min_filter_correlation > 1
                error('min_filter_correlation must be between 0 and 1');
            end
            
            if obj.min_duration <= 0
                error('min_duration must be positive');
            end
        end
        
        function display(obj)
            % DISPLAY - Display spike detection configuration
            % Shows all spike detection parameters in a formatted way
            fprintf('\n=== SPIKE DETECTION CONFIGURATION ===\n');
            fprintf('Basic Detection Parameters:\n');
            fprintf('  Voltage Threshold: %.1f mV\n', obj.vm_thresh);
            fprintf('  Second Derivative Threshold: %.1f\n', obj.d2v_thresh);
            fprintf('  Search Back Window: %.1f ms\n', obj.search_back_ms);
            fprintf('  Plot Flag: %s\n', mat2str(obj.plot_flag));
            
            fprintf('\nEnhanced Elbow Method Parameters:\n');
            fprintf('  Elbow Threshold: %.1f mV\n', obj.elbow_thresh);
            fprintf('  Spike Threshold: %.1f mV\n', obj.spike_thresh);
            fprintf('  Min Derivative Threshold: %.2f mV/ms\n', obj.min_dv_thresh);
            fprintf('  Time to Peak Threshold: %.1f ms\n', obj.time_to_peak_thresh);
            
            fprintf('\nQuality Control Parameters:\n');
            fprintf('  Min Spikes: %d\n', obj.min_spikes);
            fprintf('  Min Firing Rate: %.1f Hz\n', obj.min_firing_rate);
            fprintf('  Max Firing Rate: %.1f Hz\n', obj.max_firing_rate);
            fprintf('  Min Filter Correlation: %.2f\n', obj.min_filter_correlation);
            fprintf('  Min Duration: %.1f s\n', obj.min_duration);
            fprintf('=====================================\n\n');
        end
    end
end 