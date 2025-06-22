classdef AnalysisConfig
    % ANALYSISCONFIG - Configuration class for analysis parameters
    % This class contains all the parameters needed for the sodium channel
    % dynamics analysis, including sampling rates, filtering parameters,
    % and analysis thresholds.
    %
    % Properties:
    %   Amp - Amplifier channel name
    %   SamplingInterval - Time between samples (seconds)
    %   Verbose - Verbosity level for logging
    %   cutoff_freq_Hz - Low-pass filter cutoff frequency
    %   n_trials_filter - Number of trials for filter estimation
    %   regularization - Regularization parameter for filter estimation
    %   max_lag_ms - Maximum filter lag in milliseconds
    %
    % Author: Maxwell
    % Date: 2024
    
    properties
        Amp = 'Amp1'                    % Amplifier channel name
        SamplingInterval = 0.0001       % 10 kHz sampling (seconds)
        Verbose = 1                     % Verbosity level
        
        % Filter parameters
        cutoff_freq_Hz = 90             % Low-pass cutoff for cleaned voltage
        n_trials_filter = 50            % Number of trials for filter estimation
        regularization = 1e-4           % Regularization parameter
        max_lag_ms = 5                  % Maximum filter lag (ms)
        
        % Analysis windows
        sta_window_before_ms = 100      % STA window before spike (ms)
        sta_window_after_ms = 100       % STA window after spike (ms)
        spike_pre_window_ms = 10        % Pre-spike window for analysis (ms)
        spike_post_window_ms = 10       % Post-spike window for analysis (ms)
    end
    
    methods
        function obj = AnalysisConfig()
            % ANALYSISCONFIG Constructor
            % Creates a new AnalysisConfig instance with default values
            % No input parameters required - uses default values
        end
        
        function sampling_rate = getSamplingRate(obj)
            % GETSAMPLINGRATE - Get sampling rate in Hz
            % Returns the sampling rate calculated from the sampling interval
            sampling_rate = 1 / obj.SamplingInterval;
        end
        
        function normalized_cutoff = getNormalizedCutoff(obj)
            % GETNORMALIZEDCUTOFF - Get normalized cutoff frequency
            % Returns the normalized cutoff frequency for filter design
            sampling_rate = obj.getSamplingRate();
            normalized_cutoff = obj.cutoff_freq_Hz / (sampling_rate / 2);
        end
        
        function validate(obj)
            % VALIDATE - Validate configuration parameters
            % Checks that all parameters are within reasonable bounds
            % Throws error if validation fails
            
            if obj.SamplingInterval <= 0
                error('SamplingInterval must be positive');
            end
            
            if obj.cutoff_freq_Hz <= 0
                error('cutoff_freq_Hz must be positive');
            end
            
            if obj.n_trials_filter <= 0
                error('n_trials_filter must be positive');
            end
            
            if obj.regularization < 0
                error('regularization must be non-negative');
            end
            
            if obj.max_lag_ms <= 0
                error('max_lag_ms must be positive');
            end
            
            % Check that cutoff frequency is less than Nyquist
            sampling_rate = obj.getSamplingRate();
            nyquist_freq = sampling_rate / 2;
            if obj.cutoff_freq_Hz >= nyquist_freq
                error('cutoff_freq_Hz must be less than Nyquist frequency (%.1f Hz)', nyquist_freq);
            end
        end
        
        function display(obj)
            % DISPLAY - Display configuration parameters
            % Shows all configuration parameters in a formatted way
            fprintf('\n=== ANALYSIS CONFIGURATION ===\n');
            fprintf('Amplifier Channel: %s\n', obj.Amp);
            fprintf('Sampling Interval: %.1f μs (%.1f kHz)\n', ...
                obj.SamplingInterval * 1e6, obj.getSamplingRate() / 1000);
            fprintf('Verbosity Level: %d\n', obj.Verbose);
            fprintf('\nFilter Parameters:\n');
            fprintf('  Cutoff Frequency: %.1f Hz\n', obj.cutoff_freq_Hz);
            fprintf('  Normalized Cutoff: %.3f\n', obj.getNormalizedCutoff());
            fprintf('  Trials for Filter: %d\n', obj.n_trials_filter);
            fprintf('  Regularization: %.1e\n', obj.regularization);
            fprintf('  Max Lag: %.1f ms\n', obj.max_lag_ms);
            fprintf('\nAnalysis Windows:\n');
            fprintf('  STA Before: %.1f ms\n', obj.sta_window_before_ms);
            fprintf('  STA After: %.1f ms\n', obj.sta_window_after_ms);
            fprintf('  Spike Pre: %.1f ms\n', obj.spike_pre_window_ms);
            fprintf('  Spike Post: %.1f ms\n', obj.spike_post_window_ms);
            fprintf('================================\n\n');
        end
    end
end 