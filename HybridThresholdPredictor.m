%% HYBRID DATA-DRIVEN THRESHOLD PREDICTION FOR SPIKE RESPONSE MODEL
% This class inherits from SpikeResponseModel and adds data-driven threshold prediction using:
% 1. Ridge regression with firing rate history
% 2. Ridge regression with subthreshold membrane voltage
% 3. Hybrid approach combining both
% 4. Integration with detect_spike_initiation_elbow_v2 for spike detection

classdef HybridThresholdPredictor < SpikeResponseModel

    properties
        % Feature extraction parameters
        history_window_ms = 100   % ms of history to include
        history_bins      = 20    % Number of history bins
        voltage_window_ms = 20    % ms of voltage history
        voltage_samples             % Number of voltage samples

        % Ridge regression parameters
        alpha_rate = 1.0          % Ridge regularization for firing rate
        alpha_voltage = 0.1       % Ridge regularization for voltage
        alpha_hybrid = 0.5        % Ridge regularization for hybrid

        % Spike detection parameters (for detect_spike_initiation_elbow_v2)
        vm_thresh = -20           % Voltage threshold for peak detection (mV)
        d2v_thresh = 50           % Second derivative threshold for elbow detection
        search_back_ms = 2        % Search back window in ms
        elbow_thresh = -65        % Minimum elbow voltage threshold
        spike_thresh = -10        % Minimum spike peak threshold
        min_dv_thresh = 0.1       % Minimum dV/dt threshold
        time_to_peak_thresh = 1.5 % Maximum time to peak threshold

        % Fitted models
        rate_model        % Firing rate regression model
        voltage_model     % Voltage regression model
        hybrid_model      % Hybrid regression model

        % Performance metrics (ADD THESE LINES)
        rate_r2 = []      % R² for rate model
        voltage_r2 = []   % R² for voltage model
        hybrid_r2 = []    % R² for hybrid model

        % Feature normalization parameters (stored for prediction)
        rate_features_mean
        rate_features_std
        voltage_features_mean
        voltage_features_std
        hybrid_features_mean
        hybrid_features_std
    end

    methods
        function obj = HybridThresholdPredictor(Vm_data, Vm_cleaned, dt, varargin)
            % Constructor that automatically detects spikes using detect_spike_initiation_elbow_v2
            % Inputs:
            %   Vm_data: Raw voltage data with spikes
            %   Vm_cleaned: Subthreshold voltage data (spikes removed)
            %   dt: Time step
            %   varargin: Optional parameters for spike detection

            % Parse optional arguments for spike detection FIRST
            p = inputParser;
            addParameter(p, 'vm_thresh', -20, @isnumeric);
            addParameter(p, 'd2v_thresh', 50, @isnumeric);
            addParameter(p, 'search_back_ms', 2, @isnumeric);
            addParameter(p, 'plot_detection', false, @islogical);
            addParameter(p, 'cell_id', 'Hybrid-Cell', @ischar);
            addParameter(p, 'cell_type', 'Unknown', @ischar);
            addParameter(p, 'elbow_thresh', -65, @isnumeric);
            addParameter(p, 'spike_thresh', -10, @isnumeric);
            addParameter(p, 'min_dv_thresh', 0.1, @isnumeric);
            addParameter(p, 'time_to_peak_thresh', 1.5, @isnumeric);
            parse(p, varargin{:});

            % Detect spikes using detect_spike_initiation_elbow_v2 BEFORE calling superclass
            fprintf('=== INITIALIZING HYBRID THRESHOLD PREDICTOR ===\n');
            fprintf('Running spike detection with detect_spike_initiation_elbow_v2...\n');

            [elbow_indices, spike_peaks, isi, avg_spike, diagnostic_info] = ...
                detect_spike_initiation_elbow_v2(Vm_data, dt, p.Results.vm_thresh, p.Results.d2v_thresh, ...
                p.Results.search_back_ms, p.Results.plot_detection, ...
                'elbow_thresh', p.Results.elbow_thresh, 'spike_thresh', p.Results.spike_thresh, ...
                'min_dv_thresh', p.Results.min_dv_thresh, 'time_to_peak_thresh', p.Results.time_to_peak_thresh);

            % Extract threshold values at spike initiation points
            threshold_values = Vm_data(elbow_indices);

            % Set refractory period
            tau_ref_ms = 2.0;  % 2 ms absolute refractory period

            % Call parent constructor (SpikeResponseModel) FIRST
            obj@SpikeResponseModel(Vm_cleaned, Vm_data, dt, avg_spike, tau_ref_ms, ...
                elbow_indices, threshold_values, p.Results.cell_id, p.Results.cell_type);

            % NOW set the additional properties after calling superclass constructor
            obj.vm_thresh = p.Results.vm_thresh;
            obj.d2v_thresh = p.Results.d2v_thresh;
            obj.search_back_ms = p.Results.search_back_ms;
            obj.elbow_thresh = p.Results.elbow_thresh;
            obj.spike_thresh = p.Results.spike_thresh;
            obj.min_dv_thresh = p.Results.min_dv_thresh;
            obj.time_to_peak_thresh = p.Results.time_to_peak_thresh;

            % Calculate voltage window samples
            obj.voltage_samples = round(obj.voltage_window_ms / 1000 / dt);

            fprintf('HybridThresholdPredictor initialized:\n');
            fprintf('  Data length: %d samples (%.1f s)\n', length(Vm_data), length(Vm_data)*dt);
            fprintf('  Spikes detected: %d (%.1f Hz)\n', length(elbow_indices), length(elbow_indices)/(length(Vm_data)*dt));
            fprintf('  Threshold range: %.1f to %.1f mV\n', min(threshold_values), max(threshold_values));
            fprintf('  History window: %.1f ms (%d bins)\n', obj.history_window_ms, obj.history_bins);
            fprintf('  Voltage window: %.1f ms (%d samples)\n', obj.voltage_window_ms, obj.voltage_samples);
        end

        function [X_rate, y] = extract_firing_rate_features(obj)
            % Extract firing rate history features for ridge regression
            % Returns: X_rate (n_spikes × history_bins), y (n_spikes × 1)

            fprintf('\n=== EXTRACTING FIRING RATE FEATURES ===\n');

            % Time parameters
            history_samples = round(obj.history_window_ms / 1000 / obj.dt);
            bin_size = history_samples / obj.history_bins;

            % Initialize features and targets
            n_spikes = length(obj.elbow_indices);
            X_rate = zeros(n_spikes, obj.history_bins);
            y = obj.threshold_values;

            % Create binary spike train
            spike_train = zeros(size(obj.Vm_recorded));
            spike_train(obj.elbow_indices) = 1;

            % Extract features for each spike
            valid_idx = [];
            for i = 1:n_spikes
                spike_idx = obj.elbow_indices(i);

                % Check if we have enough history
                if spike_idx > history_samples
                    % Extract history window
                    history_start = spike_idx - history_samples + 1;
                    history_end = spike_idx - 1; % Exclude current spike
                    history_spikes = spike_train(history_start:history_end);

                    % Bin the firing rate history with proper bounds checking
                    for bin = 1:obj.history_bins
                        bin_start = round((bin-1) * bin_size) + 1;
                        bin_end = round(bin * bin_size);

                        % Ensure indices are within bounds
                        bin_start = max(1, bin_start);
                        bin_end = min(length(history_spikes), bin_end);

                        % Calculate firing rate for this bin
                        if bin_end >= bin_start
                            actual_bin_size = bin_end - bin_start + 1;
                            X_rate(i, bin) = sum(history_spikes(bin_start:bin_end)) / (actual_bin_size * obj.dt); % Hz
                        else
                            X_rate(i, bin) = 0;  % Empty bin
                        end
                    end

                    valid_idx = [valid_idx; i];
                end
            end

            % Keep only valid samples
            X_rate = X_rate(valid_idx, :);
            y = y(valid_idx);

            fprintf('Firing rate features extracted:\n');
            fprintf('  Valid spikes: %d/%d\n', length(valid_idx), n_spikes);
            fprintf('  Feature matrix: %d × %d\n', size(X_rate));
            fprintf('  History samples: %d, bin size: %.1f\n', history_samples, bin_size);
            if ~isempty(X_rate)
                fprintf('  Rate range: %.1f - %.1f Hz\n', min(X_rate(:)), max(X_rate(:)));
            end
        end

        function [X_voltage, y] = extract_voltage_features(obj)
            % Extract subthreshold voltage history features
            % Returns: X_voltage (n_spikes × voltage_samples), y (n_spikes × 1)

            fprintf('\n=== EXTRACTING VOLTAGE FEATURES ===\n');

            % Initialize features and targets
            n_spikes = length(obj.elbow_indices);
            X_voltage = zeros(n_spikes, obj.voltage_samples);
            y = obj.threshold_values;

            % Extract features for each spike
            valid_idx = [];
            for i = 1:n_spikes
                spike_idx = obj.elbow_indices(i);

                % Check if we have enough history
                if spike_idx > obj.voltage_samples
                    % Extract voltage history (subthreshold)
                    history_start = spike_idx - obj.voltage_samples + 1;
                    history_end = spike_idx;

                    X_voltage(i, :) = obj.Vm(history_start:history_end);  % Use Vm (subthreshold)
                    valid_idx = [valid_idx; i];
                end
            end

            % Keep only valid samples
            X_voltage = X_voltage(valid_idx, :);
            y = y(valid_idx);

            fprintf('Voltage features extracted:\n');
            fprintf('  Valid spikes: %d/%d\n', length(valid_idx), n_spikes);
            fprintf('  Feature matrix: %d × %d\n', size(X_voltage));
            fprintf('  Voltage range: %.1f - %.1f mV\n', min(X_voltage(:)), max(X_voltage(:)));
        end

        function [X_hybrid, y] = extract_hybrid_features(obj)
            % Extract combined firing rate + voltage features
            % Returns: X_hybrid (n_spikes × (history_bins + voltage_samples)), y (n_spikes × 1)

            fprintf('\n=== EXTRACTING HYBRID FEATURES ===\n');

            % Time parameters
            history_samples = round(obj.history_window_ms / 1000 / obj.dt);

            n_spikes = length(obj.elbow_indices);
            X_hybrid = [];
            y = [];

            % Create spike train
            spike_train = zeros(size(obj.Vm_recorded));
            spike_train(obj.elbow_indices) = 1;

            for i = 1:n_spikes
                spike_idx = obj.elbow_indices(i);

                % Check if we have enough history for both features
                if spike_idx > max(history_samples, obj.voltage_samples)
                    % Extract rate features
                    history_start = spike_idx - history_samples + 1;
                    history_end = spike_idx - 1;
                    history_spikes = spike_train(history_start:history_end);

                    rate_features = zeros(1, obj.history_bins);
                    bin_size = history_samples / obj.history_bins;
                    for bin = 1:obj.history_bins
                        bin_start = round((bin-1) * bin_size) + 1;
                        bin_end = round(bin * bin_size);

                        % Ensure indices are within bounds
                        bin_start = max(1, bin_start);
                        bin_end = min(length(history_spikes), bin_end);

                        % Calculate firing rate for this bin
                        if bin_end >= bin_start
                            actual_bin_size = bin_end - bin_start + 1;
                            rate_features(bin) = sum(history_spikes(bin_start:bin_end)) / (actual_bin_size * obj.dt);
                        else
                            rate_features(bin) = 0;  % Empty bin
                        end
                    end

                    % Extract voltage features
                    volt_start = spike_idx - obj.voltage_samples + 1;
                    volt_end = spike_idx;
                    voltage_features = obj.Vm(volt_start:volt_end)';  % Use Vm (subthreshold)

                    % Combine features
                    X_hybrid = [X_hybrid; rate_features, voltage_features];
                    y = [y; obj.threshold_values(i)];
                end
            end

            fprintf('Hybrid features extracted:\n');
            fprintf('  Valid spikes: %d/%d\n', size(X_hybrid, 1), n_spikes);
            fprintf('  Feature matrix: %d × %d (%d rate + %d voltage)\n', ...
                size(X_hybrid), obj.history_bins, obj.voltage_samples);
        end

        function obj = fit_ridge_models(obj)  % CHANGE: Return obj
            % Fit ridge regression models for all three approaches

            fprintf('\n=== FITTING RIDGE REGRESSION MODELS ===\n');

            % Extract features
            [X_rate, y_rate] = obj.extract_firing_rate_features();
            [X_voltage, y_voltage] = obj.extract_voltage_features();
            [X_hybrid, y_hybrid] = obj.extract_hybrid_features();

            % Standardize features and store normalization parameters
            [X_rate_std, obj.rate_features_mean, obj.rate_features_std] = standardize_features(X_rate);
            [X_voltage_std, obj.voltage_features_mean, obj.voltage_features_std] = standardize_features(X_voltage);
            [X_hybrid_std, obj.hybrid_features_mean, obj.hybrid_features_std] = standardize_features(X_hybrid);

            % Fit ridge regression models
            obj.rate_model = ridge_regression(X_rate_std, y_rate, obj.alpha_rate);
            obj.voltage_model = ridge_regression(X_voltage_std, y_voltage, obj.alpha_voltage);
            obj.hybrid_model = ridge_regression(X_hybrid_std, y_hybrid, obj.alpha_hybrid);

            % Calculate R²
            obj.rate_r2 = calculate_r2(y_rate, X_rate_std * obj.rate_model.beta + obj.rate_model.intercept);
            obj.voltage_r2 = calculate_r2(y_voltage, X_voltage_std * obj.voltage_model.beta + obj.voltage_model.intercept);
            obj.hybrid_r2 = calculate_r2(y_hybrid, X_hybrid_std * obj.hybrid_model.beta + obj.hybrid_model.intercept);

            fprintf('\nModel fitting results:\n');
            fprintf('  Firing rate model: R² = %.3f (n=%d)\n', obj.rate_r2, length(y_rate));
            fprintf('  Voltage model: R² = %.3f (n=%d)\n', obj.voltage_r2, length(y_voltage));
            fprintf('  Hybrid model: R² = %.3f (n=%d)\n', obj.hybrid_r2, length(y_hybrid));

            % Display best model
            [best_r2, best_idx] = max([obj.rate_r2, obj.voltage_r2, obj.hybrid_r2]);
            model_names = {'Firing Rate', 'Voltage', 'Hybrid'};
            fprintf(' Best model: %s (R² = %.3f)\n', model_names{best_idx}, best_r2);
        end

        function threshold_trace = predict_threshold_trace(obj, model_type)
            % Predict threshold trace for entire recording using fitted model
            % model_type: 'rate', 'voltage', or 'hybrid'

            fprintf('\n=== PREDICTING THRESHOLD TRACE (%s model) ===\n', upper(model_type));

            threshold_trace = nan(size(obj.Vm_recorded));

            % Get appropriate model and normalization parameters
            switch lower(model_type)
                case 'rate'
                    model = obj.rate_model;
                    features_mean = obj.rate_features_mean;
                    features_std_dev = obj.rate_features_std;
                    n_features = obj.history_bins;
                    history_samples = round(obj.history_window_ms / 1000 / obj.dt);
                case 'voltage'
                    model = obj.voltage_model;
                    features_mean = obj.voltage_features_mean;
                    features_std_dev = obj.voltage_features_std;
                    n_features = obj.voltage_samples;
                    history_samples = obj.voltage_samples;
                case 'hybrid'
                    model = obj.hybrid_model;
                    features_mean = obj.hybrid_features_mean;
                    features_std_dev = obj.hybrid_features_std;
                    n_features = obj.history_bins + obj.voltage_samples;
                    history_samples = round(obj.history_window_ms / 1000 / obj.dt);
                otherwise
                    error('Unknown model type: %s', model_type);
            end

            % Check if model exists and normalization parameters are available
            % UPDATED VALIDATION: Check if model is a struct with required fields
            if ~isstruct(model) || ~isfield(model, 'beta') || ~isfield(model, 'intercept')
                error('Model for %s has not been fitted yet. Call fit_ridge_models() first.', model_type);
            end
            if isempty(features_mean) || isempty(features_std_dev)
                error('Normalization parameters for %s model are not available.', model_type);
            end
            % Create spike train
            spike_train = zeros(size(obj.Vm_recorded));
            spike_train(obj.elbow_indices) = 1;

            % Predict threshold for each time point
            for t = (history_samples + 1):length(obj.Vm_recorded)
                features = [];

                switch lower(model_type)
                    case 'rate'
                        % Extract firing rate features
                        history_start = t - history_samples + 1;
                        history_end = t - 1;
                        history_spikes = spike_train(history_start:history_end);

                        bin_size = history_samples / obj.history_bins;
                        features = zeros(1, obj.history_bins);
                        for bin = 1:obj.history_bins
                            bin_start = round((bin-1) * bin_size) + 1;
                            bin_end = round(bin * bin_size);

                            % Ensure indices are within bounds
                            bin_start = max(1, bin_start);
                            bin_end = min(length(history_spikes), bin_end);

                            if bin_end >= bin_start
                                actual_bin_size = bin_end - bin_start + 1;
                                features(bin) = sum(history_spikes(bin_start:bin_end)) / (actual_bin_size * obj.dt);
                            else
                                features(bin) = 0;
                            end
                        end

                    case 'voltage'
                        % Extract voltage features
                        volt_start = t - obj.voltage_samples + 1;
                        volt_end = t;
                        features = obj.Vm(volt_start:volt_end)';  % Use Vm (subthreshold)

                    case 'hybrid'
                        % Extract both rate and voltage features
                        % Rate features
                        history_start = t - history_samples + 1;
                        history_end = t - 1;
                        history_spikes = spike_train(history_start:history_end);

                        rate_features = zeros(1, obj.history_bins);
                        bin_size = history_samples / obj.history_bins;
                        for bin = 1:obj.history_bins
                            bin_start = round((bin-1) * bin_size) + 1;
                            bin_end = round(bin * bin_size);

                            % Ensure indices are within bounds
                            bin_start = max(1, bin_start);
                            bin_end = min(length(history_spikes), bin_end);

                            if bin_end >= bin_start
                                actual_bin_size = bin_end - bin_start + 1;
                                rate_features(bin) = sum(history_spikes(bin_start:bin_end)) / (actual_bin_size * obj.dt);
                            else
                                rate_features(bin) = 0;
                            end
                        end

                        % Voltage features
                        volt_start = t - obj.voltage_samples + 1;
                        volt_end = t;
                        voltage_features = obj.Vm(volt_start:volt_end)';  % Use Vm (subthreshold)

                        features = [rate_features, voltage_features];
                end

                % Check feature dimensions
                if length(features) ~= length(features_mean)
                    error('Feature dimension mismatch: got %d features, expected %d', ...
                        length(features), length(features_mean));
                end

                % Standardize features using stored normalization parameters
                % Avoid division by zero
                safe_std = features_std_dev;
                safe_std(safe_std == 0) = 1;

                features_normalized = (features - features_mean) ./ safe_std;

                % Handle NaN/Inf from standardization
                features_normalized(~isfinite(features_normalized)) = 0;

                % Predict threshold
                threshold_trace(t) = features_normalized * model.beta + model.intercept;
            end

            fprintf('Threshold trace predicted:\n');
            fprintf('  Valid predictions: %d/%d (%.1f%%)\n', ...
                sum(~isnan(threshold_trace)), length(threshold_trace), ...
                100 * sum(~isnan(threshold_trace)) / length(threshold_trace));
        end
        function visualize_model_insights(obj, model_type)
            % Visualize model weights, filters, and predictions
            % model_type: 'rate', 'voltage', 'hybrid', or 'all'

            if nargin < 2
                model_type = 'all';
            end

            fprintf('\n=== VISUALIZING MODEL INSIGHTS ===\n');

            % Create figure
            figure('Position', [50, 50, 1800, 1000]);

            if strcmpi(model_type, 'all')
                % Compare all three models
                obj.visualize_all_models();
            else
                % Visualize specific model
                obj.visualize_single_model(model_type);
            end
        end

        function visualize_all_models(obj)
            % Comprehensive visualization comparing all models

            % Row 1: Model weights/filters
            % Rate model weights
            subplot(3, 4, 1);
            bar(obj.rate_model.beta, 'FaceColor', [0.8, 0.3, 0.3]);
            xlabel('History Bin');
            ylabel('Weight');
            title('Rate Model Weights');
            grid on;

            % Voltage model weights as image
            subplot(3, 4, 2);
            imagesc(reshape(obj.voltage_model.beta, [], 1)');
            colorbar;
            xlabel('Time Sample');
            ylabel('');
            title('Voltage Model Weights');
            set(gca, 'YTick', []);

            % Hybrid model weights
            subplot(3, 4, 3);
            n_rate = obj.history_bins;
            n_voltage = obj.voltage_samples;
            beta_hybrid = obj.hybrid_model.beta;

            % Split weights
            rate_weights = beta_hybrid(1:n_rate);
            voltage_weights = beta_hybrid((n_rate+1):end);

            yyaxis left;
            bar(1:n_rate, rate_weights, 'FaceColor', [0.8, 0.3, 0.3]);
            ylabel('Rate Weights');

            yyaxis right;
            plot(1:n_voltage, voltage_weights, 'b-', 'LineWidth', 2);
            ylabel('Voltage Weights');
            xlabel('Feature Index');
            title('Hybrid Model Weights');
            grid on;

            % Row 1, Col 4: Weight magnitudes comparison
            subplot(3, 4, 4);
            weight_norms = [norm(obj.rate_model.beta), ...
                norm(obj.voltage_model.beta), ...
                norm(rate_weights), norm(voltage_weights)];
            bar(weight_norms, 'FaceColor', [0.3, 0.7, 0.9]);
            set(gca, 'XTickLabel', {'Rate', 'Voltage', 'Hybrid-R', 'Hybrid-V'});
            ylabel('L2 Norm');
            title('Weight Magnitudes');
            grid on;

            % Row 2: Threshold predictions
            t_ms = (1:length(obj.Vm_recorded)) * obj.dt * 1000; % Convert to ms

            % Rate model predictions
            subplot(3, 4, 5);
            threshold_rate = obj.predict_threshold_trace('rate');
            valid_idx = ~isnan(threshold_rate);
            plot(t_ms(valid_idx), threshold_rate(valid_idx), 'r-', 'LineWidth', 1);
            hold on;
            spike_times_ms = obj.elbow_indices * obj.dt * 1000;
            scatter(spike_times_ms, obj.threshold_values, 20, 'ko', 'filled');
            xlabel('Time (ms)');
            ylabel('Threshold (mV)');
            title(sprintf('Rate Model (R² = %.3f)', obj.rate_r2));
            legend('Predicted', 'True');
            xlim([0, 500]); % First 500ms
            grid on;

            % Voltage model predictions
            subplot(3, 4, 6);
            threshold_voltage = obj.predict_threshold_trace('voltage');
            valid_idx = ~isnan(threshold_voltage);
            plot(t_ms(valid_idx), threshold_voltage(valid_idx), 'g-', 'LineWidth', 1);
            hold on;
            scatter(spike_times_ms, obj.threshold_values, 20, 'ko', 'filled');
            xlabel('Time (ms)');
            ylabel('Threshold (mV)');
            title(sprintf('Voltage Model (R² = %.3f)', obj.voltage_r2));
            xlim([0, 500]);
            grid on;

            % Hybrid model predictions
            subplot(3, 4, 7);
            threshold_hybrid = obj.predict_threshold_trace('hybrid');
            valid_idx = ~isnan(threshold_hybrid);
            plot(t_ms(valid_idx), threshold_hybrid(valid_idx), 'b-', 'LineWidth', 1);
            hold on;
            scatter(spike_times_ms, obj.threshold_values, 20, 'ko', 'filled');
            xlabel('Time (ms)');
            ylabel('Threshold (mV)');
            title(sprintf('Hybrid Model (R² = %.3f)', obj.hybrid_r2));
            xlim([0, 500]);
            grid on;

            % Prediction statistics
            subplot(3, 4, 8);
            models = {'Rate', 'Voltage', 'Hybrid'};
            thresholds = {threshold_rate, threshold_voltage, threshold_hybrid};
            stats_data = zeros(3, 4); % mean, std, min, max

            for i = 1:3
                valid = ~isnan(thresholds{i});
                stats_data(i, :) = [mean(thresholds{i}(valid)), ...
                    std(thresholds{i}(valid)), ...
                    min(thresholds{i}(valid)), ...
                    max(thresholds{i}(valid))];
            end

            bar(stats_data);
            set(gca, 'XTickLabel', models);
            ylabel('Threshold (mV)');
            legend('Mean', 'Std', 'Min', 'Max', 'Location', 'best');
            title('Prediction Statistics');
            grid on;

            % Row 3: Feature importance and temporal filters

            % Rate model: temporal filter
            subplot(3, 4, 9);
            bin_centers_ms = (0.5:obj.history_bins) * (obj.history_window_ms / obj.history_bins);
            bar(bin_centers_ms, flip(obj.rate_model.beta), 'FaceColor', [0.8, 0.3, 0.3]);
            xlabel('Time before spike (ms)');
            ylabel('Weight');
            title('Rate Temporal Filter');
            grid on;

            % Voltage model: temporal filter
            subplot(3, 4, 10);
            voltage_time_ms = (0:obj.voltage_samples-1) * obj.dt * 1000;
            plot(voltage_time_ms, flip(obj.voltage_model.beta), 'g-', 'LineWidth', 2);
            xlabel('Time before spike (ms)');
            ylabel('Weight');
            title('Voltage Temporal Filter');
            grid on;

            % Feature correlation matrix
            subplot(3, 4, 11);
            % Extract features for visualization
            [X_hybrid, ~] = obj.extract_hybrid_features();
            if size(X_hybrid, 1) > 1000
                % Subsample for visualization
                idx = randperm(size(X_hybrid, 1), 1000);
                X_subset = X_hybrid(idx, :);
            else
                X_subset = X_hybrid;
            end

            % Compute correlation for rate and voltage features separately
            n_rate = obj.history_bins;
            corr_rate = corr(X_subset(:, 1:n_rate));
            corr_voltage = corr(X_subset(:, (n_rate+1):end));

            % Display rate correlation
            imagesc(corr_rate);
            colorbar;
            xlabel('Rate Feature');
            ylabel('Rate Feature');
            title('Rate Feature Correlations');
            axis square;

            % Summary text
            subplot(3, 4, 12);
            axis off;

            % Calculate some key metrics
            true_range = [min(obj.threshold_values), max(obj.threshold_values)];
            pred_ranges = cell(3, 1);
            for i = 1:3
                valid = ~isnan(thresholds{i});
                if any(valid)
                    pred_ranges{i} = [min(thresholds{i}(valid)), max(thresholds{i}(valid))];
                else
                    pred_ranges{i} = [NaN, NaN];
                end
            end

            summary_text = {
                'MODEL INSIGHTS SUMMARY';
                '';
                sprintf('True threshold range: [%.1f, %.1f] mV', true_range);
                sprintf('Vm recorded range: [%.1f, %.1f] mV', ...
                min(obj.Vm_recorded), max(obj.Vm_recorded));
                '';
                'PREDICTED RANGES:';
                sprintf('Rate: [%.1f, %.1f] mV', pred_ranges{1});
                sprintf('Voltage: [%.1f, %.1f] mV', pred_ranges{2});
                sprintf('Hybrid: [%.1f, %.1f] mV', pred_ranges{3});
                '';
                'FEATURE WINDOWS:';
                sprintf('Rate: %d bins over %d ms', obj.history_bins, obj.history_window_ms);
                sprintf('Voltage: %d samples over %.1f ms', ...
                obj.voltage_samples, obj.voltage_window_ms);
                };

            text(0.05, 0.95, summary_text, 'Units', 'normalized', ...
                'VerticalAlignment', 'top', 'FontSize', 10, 'FontName', 'FixedWidth');

            sgtitle('Model Weights, Filters, and Threshold Predictions', 'FontSize', 14, 'FontWeight', 'bold');
        end

        function visualize_single_model(obj, model_type)
            % Detailed visualization for a single model

            switch lower(model_type)
                case 'rate'
                    model = obj.rate_model;
                    r2 = obj.rate_r2;
                    threshold_trace = obj.predict_threshold_trace('rate');
                case 'voltage'
                    model = obj.voltage_model;
                    r2 = obj.voltage_r2;
                    threshold_trace = obj.predict_threshold_trace('voltage');
                case 'hybrid'
                    model = obj.hybrid_model;
                    r2 = obj.hybrid_r2;
                    threshold_trace = obj.predict_threshold_trace('hybrid');
            end

            % Subplot 1: Weights
            subplot(2, 3, 1);
            if strcmpi(model_type, 'hybrid')
                n_rate = obj.history_bins;
                rate_weights = model.beta(1:n_rate);
                voltage_weights = model.beta((n_rate+1):end);

                plot(1:n_rate, rate_weights, 'ro-', 'LineWidth', 2, 'DisplayName', 'Rate');
                hold on;
                plot(1:length(voltage_weights), voltage_weights, 'bo-', 'LineWidth', 2, 'DisplayName', 'Voltage');
                xlabel('Feature Index');
                ylabel('Weight');
                legend();
            else
                bar(model.beta, 'FaceColor', [0.3, 0.7, 0.9]);
                xlabel('Feature Index');
                ylabel('Weight');
            end
            title(sprintf('%s Model Weights', upper(model_type)));
            grid on;

            % Subplot 2: Temporal filter interpretation
            subplot(2, 3, 2);
            if strcmpi(model_type, 'rate')
                bin_centers_ms = (0.5:obj.history_bins) * (obj.history_window_ms / obj.history_bins);
                bar(bin_centers_ms, flip(model.beta), 'FaceColor', [0.8, 0.3, 0.3]);
                xlabel('Time before spike (ms)');
            elseif strcmpi(model_type, 'voltage')
                voltage_time_ms = (0:obj.voltage_samples-1) * obj.dt * 1000;
                plot(voltage_time_ms, flip(model.beta), 'g-', 'LineWidth', 2);
                xlabel('Time before spike (ms)');
            else
                % Hybrid: show both
                n_rate = obj.history_bins;
                rate_weights = model.beta(1:n_rate);
                voltage_weights = model.beta((n_rate+1):end);

                bin_centers_ms = (0.5:obj.history_bins) * (obj.history_window_ms / obj.history_bins);
                voltage_time_ms = (0:obj.voltage_samples-1) * obj.dt * 1000;

                yyaxis left;
                bar(bin_centers_ms, flip(rate_weights), 'FaceColor', [0.8, 0.3, 0.3]);
                ylabel('Rate Weight');

                yyaxis right;
                plot(voltage_time_ms, flip(voltage_weights), 'b-', 'LineWidth', 2);
                ylabel('Voltage Weight');
                xlabel('Time before spike (ms)');
            end
            ylabel('Weight');
            title('Temporal Filter');
            grid on;

            % Subplot 3: Threshold trace (full)
            subplot(2, 3, 3);
            t_sec = (1:length(obj.Vm_recorded)) * obj.dt;
            valid_idx = ~isnan(threshold_trace);
            plot(t_sec(valid_idx), threshold_trace(valid_idx), 'b-', 'LineWidth', 1);
            hold on;
            spike_times = obj.elbow_indices * obj.dt;
            scatter(spike_times, obj.threshold_values, 20, 'ro', 'filled');
            xlabel('Time (s)');
            ylabel('Threshold (mV)');
            title(sprintf('Full Threshold Trace (R² = %.3f)', r2));
            legend('Predicted', 'True');
            grid on;

            % Subplot 4: Threshold trace (zoom)
            subplot(2, 3, 4);
            zoom_start = 10; % seconds
            zoom_end = 15;
            zoom_idx = t_sec >= zoom_start & t_sec <= zoom_end;
            plot(t_sec(zoom_idx & valid_idx), threshold_trace(zoom_idx & valid_idx), 'b-', 'LineWidth', 2);
            hold on;

            % Add Vm trace for context
            yyaxis right;
            plot(t_sec(zoom_idx), obj.Vm_recorded(zoom_idx), 'k-', 'Alpha', 0.5);
            ylabel('Vm (mV)');

            yyaxis left;
            spike_mask = spike_times >= zoom_start & spike_times <= zoom_end;
            scatter(spike_times(spike_mask), obj.threshold_values(spike_mask), 40, 'ro', 'filled');
            xlabel('Time (s)');
            ylabel('Threshold (mV)');
            title('Zoomed View with Vm');
            grid on;

            % Subplot 5: Prediction vs Truth scatter
            subplot(2, 3, 5);
            pred_at_spikes = threshold_trace(obj.elbow_indices);
            valid_pred = ~isnan(pred_at_spikes);
            scatter(obj.threshold_values(valid_pred), pred_at_spikes(valid_pred), 30, 'filled');
            hold on;
            plot([min(obj.threshold_values), max(obj.threshold_values)], ...
                [min(obj.threshold_values), max(obj.threshold_values)], 'r--');
            xlabel('True Threshold (mV)');
            ylabel('Predicted Threshold (mV)');
            title(sprintf('Prediction Accuracy (r = %.3f)', ...
                corr(obj.threshold_values(valid_pred), pred_at_spikes(valid_pred))));
            axis equal;
            grid on;

            % Subplot 6: Residuals
            subplot(2, 3, 6);
            residuals = pred_at_spikes(valid_pred) - obj.threshold_values(valid_pred);
            histogram(residuals, 30, 'FaceColor', [0.3, 0.7, 0.9]);
            xlabel('Residual (mV)');
            ylabel('Count');
            title(sprintf('Residuals (mean = %.2f, std = %.2f)', ...
                mean(residuals), std(residuals)));
            grid on;

            sgtitle(sprintf('%s Model Analysis', upper(model_type)), 'FontSize', 14, 'FontWeight', 'bold');
        end



        function [spike_train, Vm_sim] = generate_spikes_from_filtered_vm(obj, filter_kernel, threshold, delay_ms)
            % Generate spike train by filtering Vm and thresholding
            % Inputs:
            %   filter_kernel: temporal filter weights
            %   threshold: scalar threshold for spike initiation
            %   delay_ms: delay after spike before injecting average spike
            % Outputs:
            %   spike_train: binary vector of generated spikes
            %   Vm_sim: simulated voltage trace with average spikes injected

            delay_samples = round(delay_ms / 1000 / obj.dt);
            refractory_samples = round(obj.tau_ref_ms / 1000 / obj.dt);
            Vm_filtered = conv(obj.Vm, filter_kernel, 'same');

            Vm_sim = obj.Vm;  % Start from clean voltage trace
            spike_train = zeros(size(obj.Vm));

            t = 1;
            while t <= length(Vm_filtered)
                if Vm_filtered(t) >= threshold
                    % Inject average spike waveform
                    spike_train(t) = 1;
                    inject_start = t + delay_samples;
                    inject_end = inject_start + length(obj.avg_spike) - 1;
                     if inject_end <= length(Vm_sim)
                        Vm_sim(inject_start:inject_end) = Vm_sim(inject_start:inject_end) + obj.avg_spike(:);  % Add spike waveform
                    end
                    % Enforce refractory period
                    t = t + refractory_samples;
                else
                    t = t + 1;
                end
            end
        end

        function [spike_times, spike_indices] = generate_spikes_hybrid(obj, threshold_trace, theta0_baseline)
            % Generate spikes using predicted threshold trace + SRM mechanism
            % This uses a simple threshold crossing approach instead of simulateFast

            fprintf('\n=== GENERATING SPIKES WITH HYBRID THRESHOLD ===\n');

            % Handle NaN values in threshold trace
            valid_indices = ~isnan(threshold_trace);
            if sum(valid_indices) == 0
                error('Threshold trace contains only NaN values');
            end

            % Fill NaN values at the beginning with the first valid value
            first_valid_idx = find(valid_indices, 1, 'first');
            threshold_trace_clean = threshold_trace;
            threshold_trace_clean(1:first_valid_idx-1) = threshold_trace(first_valid_idx);

            % Combine baseline + predicted adaptive component
            total_threshold = theta0_baseline + threshold_trace_clean;

            % DEBUG: Print threshold statistics
            fprintf('DEBUG: Threshold statistics:\n');
            fprintf('  Baseline theta0: %.1f mV\n', theta0_baseline);
            fprintf('  Adaptive threshold range: [%.1f, %.1f] mV\n', ...
                min(threshold_trace_clean(valid_indices)), max(threshold_trace_clean(valid_indices)));
            fprintf('  Total threshold range: [%.1f, %.1f] mV\n', ...
                min(total_threshold(valid_indices)), max(total_threshold(valid_indices)));
            fprintf('  Vm_recorded range: [%.1f, %.1f] mV\n', ...
                min(obj.Vm_recorded), max(obj.Vm_recorded));

            % Check how many times Vm crosses threshold
            crossings = 0;
            for t = 2:length(obj.Vm_recorded)
                if obj.Vm_recorded(t) >= total_threshold(t) && obj.Vm_recorded(t-1) < total_threshold(t-1)
                    crossings = crossings + 1;
                end
            end
            fprintf('  Potential threshold crossings (before refractory): %d\n', crossings);

            % Simple threshold crossing detection with refractory period
            spike_indices = [];
            refractory_samples = round(obj.tau_ref_ms / 1000 / obj.dt); % Convert ms to samples

            for t = 2:length(obj.Vm_recorded)
                % Check for upward threshold crossing
                if obj.Vm_recorded(t) >= total_threshold(t) && obj.Vm_recorded(t-1) < total_threshold(t-1)
                    % Check refractory period
                    if isempty(spike_indices) || (t - spike_indices(end)) > refractory_samples
                        spike_indices = [spike_indices; t];
                    end
                end
            end

            spike_times = spike_indices * obj.dt;

            fprintf('Hybrid spike generation results:\n');
            fprintf('  Generated spikes: %d\n', length(spike_indices));
            fprintf('  True spikes: %d\n', length(obj.elbow_indices));
            fprintf('  Predicted rate: %.1f Hz\n', length(spike_indices) / (length(obj.Vm_recorded) * obj.dt));
            fprintf('  True rate: %.1f Hz\n', length(obj.elbow_indices) / (length(obj.Vm_recorded) * obj.dt));
        end

        function results = compare_approaches(obj)
            % Compare all three approaches without validation split (simpler version)

            fprintf('\n=== COMPARING ALL APPROACHES (NO VALIDATION SPLIT) ===\n');

            % Always fit models to ensure they exist and are up to date
            fprintf('Fitting ridge regression models...\n');
            obj.fit_ridge_models();

            results = struct();
            model_types = {'rate', 'voltage', 'hybrid'};

            for i = 1:length(model_types)
                model_type = model_types{i};

                % Predict threshold trace
                threshold_trace = obj.predict_threshold_trace(model_type);

                % Find optimal baseline threshold using Victor-Purpura
                baseline_range = linspace(-70, -30, 21);
                best_vp = inf;
                best_baseline = nan;

                for baseline = baseline_range
                    [spike_times, ~] = obj.generate_spikes_hybrid(threshold_trace, baseline);
                    true_spike_times = obj.elbow_indices * obj.dt;

                    % Calculate Victor-Purpura distance
                    vp_dist = spkd_c(spike_times, true_spike_times, ...
                        length(spike_times), length(true_spike_times), 4);

                    if vp_dist < best_vp
                        best_vp = vp_dist;
                        best_baseline = baseline;
                    end
                end

                % Store results
                results.(model_type).r2 = obj.([model_type '_r2']);
                results.(model_type).best_baseline = best_baseline;
                results.(model_type).best_vp = best_vp;
                results.(model_type).threshold_trace = threshold_trace;

                fprintf('%s model: R² = %.3f, Best baseline = %.1f mV, VP = %.3f\n', ...
                    upper(model_type), results.(model_type).r2, best_baseline, best_vp);
            end

            % Find overall best approach based on VP distance
            vp_scores = [results.rate.best_vp, results.voltage.best_vp, results.hybrid.best_vp];
            [~, best_idx] = min(vp_scores);

            fprintf('\n������ BEST APPROACH: %s (VP = %.3f)\n', upper(model_types{best_idx}), vp_scores(best_idx));

            results.best_approach = model_types{best_idx};
        end
        function diagnostics_filtered_vm(obj, Vm_sim, spike_train, zoom_xlim)
            % Wrapper to call standard diagnostics on filtered Vm simulation

            vp_dummy = NaN;

            obj.diagnostics(Vm_sim, threshold_trace, spike_train, vp_dummy, zoom_xlim);
        end

        function diagnostics(obj, V_pred, theta_trace, spike_vec, vp_dist_final, zoom_xlim, save_path, kernel_params, spike_times_sec, spike_V_values)
            t = (0:length(obj.Vm)-1) * obj.dt;
            threshold_interp = obj.interp_threshold();
            elbow_times = obj.elbow_indices * obj.dt;
            threshold_values = obj.threshold_values;

            figure('Position', [100, 100, 1200, 800]);

            % Row 1: Vm recorded & prediction (full width)
            subplot(3,3,[1 2 3]);
            plot(t, obj.Vm_recorded, 'b'); hold on;
            plot(t, V_pred, 'r');
            plot(t, theta_trace, 'k--', 'LineWidth', 1.2);
            spike_times_idx = find(spike_vec);
            plot(t(spike_times_idx), V_pred(spike_times_idx), 'mx', 'MarkerSize', 8, 'LineWidth', 1.2);
            plot(t(obj.elbow_indices), obj.Vm_recorded(obj.elbow_indices), 'bo', 'MarkerSize', 4, 'MarkerFaceColor', 'b');
            if nargin >= 10 && ~isempty(spike_times_sec) && ~isempty(spike_V_values)
                plot(spike_times_sec, spike_V_values, 'rx', 'MarkerSize', 8, 'LineWidth', 1.2);
            end
            ylabel('Voltage (mV)');
            legend('Vm\_recorded', 'V\_pred', '\theta', 'Generated Spikes', 'Elbow Spikes');
            title(sprintf('%s | %s | VP = %.2f', obj.cell_id, obj.cell_type, vp_dist_final));
            xlim(zoom_xlim);
            grid on;

            % Row 2: Threshold comparison (full width)
            subplot(3,3,[4 5 6]);
            plot(t, threshold_interp, 'b'); hold on;
            plot(t, theta_trace, 'r--', 'LineWidth', 1.2);
            mask = elbow_times >= zoom_xlim(1) & elbow_times <= zoom_xlim(2);
            plot(elbow_times(mask), threshold_values(mask), 'go', 'MarkerSize', 6, 'MarkerFaceColor', 'g');
            ylabel('Threshold (mV)');
            legend('\theta (data interp)', '\theta (model)', 'Spike Initiations');
            title('Threshold Comparison');
            xlim(zoom_xlim);
            grid on;

            % Row 3 left: ISI Histogram
            isi_true = diff(obj.elbow_indices) * obj.dt * 1000;
            isi_pred = diff(find(spike_vec)) * obj.dt * 1000;
            edges = 0:1:15;
            subplot(3,3,7);
            histogram(isi_true, edges, 'FaceColor', 'b', 'FaceAlpha', 0.5); hold on;
            histogram(isi_pred, edges, 'FaceColor', 'r', 'FaceAlpha', 0.5);
            xlabel('ISI (ms)'); ylabel('Count');
            title('ISI Distribution (0–15 ms)');
            legend('True ISI', 'Predicted ISI');
            grid on;

            % Row 3 middle: Scatter spike threshold comparison
            elbow_idx = obj.elbow_indices;
            elbow_theta = theta_trace(elbow_idx);
            subplot(3,3,8);
            scatter(threshold_values, elbow_theta, 30, 'filled');
            hold on;
            plot(xlim, xlim, 'k:');
            axis square;
            xlabel('Data Threshold'); ylabel('Model Threshold');
            r = corr(threshold_values(:), elbow_theta(:));
            title(sprintf('Threshold Corr: r = %.2f', r));
            hold off;
            grid on;

            % Row 3 right: Post-spike kernel
            if nargin >= 8 && ~isempty(kernel_params)
                subplot(3,3,9);
                t_kernel = 0:obj.dt:0.1;
                if length(kernel_params) == 2
                    A = kernel_params(1);
                    tau = kernel_params(2);
                    kernel_vals = A * exp(-t_kernel / tau);
                    param_title = sprintf('A = %.2f, \tau = %.3f', A, tau);
                else
                    A = kernel_params(1);
                    T_rise = kernel_params(2);
                    tau_decay = kernel_params(3);
                    kernel_vals = (t_kernel < T_rise) .* (A / T_rise .* t_kernel) + ...
                        (t_kernel >= T_rise) .* (A * exp(-(t_kernel - T_rise) / tau_decay));
                    param_title = sprintf('A = %.2f\nT_{rise} = %.3f\n\tau_{decay} = %.3f', A, T_rise, tau_decay);
                end
                plot(t_kernel*1000, kernel_vals, 'k', 'LineWidth', 1.5);
                xlabel('Time (ms)'); ylabel('\Delta \theta (mV)');
                title('Post-Spike Threshold');
                text(0.98, 0.98, param_title, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
                    'HorizontalAlignment', 'right', 'BackgroundColor', 'none', 'FontSize', 10);
                grid on;
            end

            sgtitle(sprintf('Diagnostics for %s (%s)', obj.cell_id, obj.cell_type));

            if nargin > 6 && ~isempty(save_path)
                saveas(gcf, save_path);
                fprintf('Diagnostic plot saved to: %s\n', save_path);
            end
        end
        function results = compare_approaches_with_validation(obj, validation_fraction)
            % Compare all approaches with proper train/validation split
            % validation_fraction: fraction of data to use for validation (e.g., 0.3)

            if nargin < 2
                validation_fraction = 0.3;
            end

            fprintf('\n=== COMPARING APPROACHES WITH VALIDATION ===\n');

            % Split spikes into training and validation sets
            n_spikes = length(obj.elbow_indices);
            n_validation = round(validation_fraction * n_spikes);

            % Use temporal split (last part for validation)
            train_indices = 1:(n_spikes - n_validation);
            val_indices = (n_spikes - n_validation + 1):n_spikes;

            fprintf('Data split: %d training, %d validation spikes\n', length(train_indices), length(val_indices));

            % Temporarily store original data
            orig_elbow_indices = obj.elbow_indices;
            orig_threshold_values = obj.threshold_values;

            % Set training data
            obj.elbow_indices = obj.elbow_indices(train_indices);
            obj.threshold_values = obj.threshold_values(train_indices);

            % Fit models on training data
            obj.fit_ridge_models();

            % Restore full data for prediction
            obj.elbow_indices = orig_elbow_indices;
            obj.threshold_values = orig_threshold_values;

            results = struct();
            model_types = {'rate', 'voltage', 'hybrid'};

            for i = 1:length(model_types)
                model_type = model_types{i};

                % Predict threshold trace
                threshold_trace = obj.predict_threshold_trace(model_type);

                % Evaluate on validation spikes
                val_spike_indices = obj.elbow_indices(val_indices);
                val_true_thresholds = obj.threshold_values(val_indices);
                val_pred_thresholds = threshold_trace(val_spike_indices);

                % Calculate validation R²
                val_r2 = calculate_r2(val_true_thresholds, val_pred_thresholds);

                % Find optimal baseline threshold using Victor-Purpura
                baseline_range = linspace(-70, -30, 21);
                best_vp = inf;
                best_baseline = nan;

                for baseline = baseline_range
                    [spike_times, ~] = obj.generate_spikes_hybrid(threshold_trace, baseline);
                    true_spike_times = obj.elbow_indices * obj.dt;

                    % Calculate Victor-Purpura distance
                    vp_dist = spkd_c(spike_times, true_spike_times, ...
                        length(spike_times), length(true_spike_times), 4);

                    if vp_dist < best_vp
                        best_vp = vp_dist;
                        best_baseline = baseline;
                    end
                end

                % Store results
                results.(model_type).train_r2 = obj.([model_type '_r2']);
                results.(model_type).val_r2 = val_r2;
                results.(model_type).best_baseline = best_baseline;
                results.(model_type).best_vp = best_vp;
                results.(model_type).threshold_trace = threshold_trace;

                fprintf('%s model: Train R² = %.3f, Val R² = %.3f, Baseline = %.1f mV, VP = %.3f\n', ...
                    upper(model_type), results.(model_type).train_r2, val_r2, best_baseline, best_vp);
            end

            % Find overall best approach based on validation performance
            val_r2_scores = [results.rate.val_r2, results.voltage.val_r2, results.hybrid.val_r2];
            [~, best_idx] = max(val_r2_scores);

            fprintf('\n BEST APPROACH (validation R²): %s (R² = %.3f)\n', ...
                upper(model_types{best_idx}), val_r2_scores(best_idx));

            results.best_approach = model_types{best_idx};
            results.validation_fraction = validation_fraction;
        end

        function plot_comparison(obj, results)
            % Plot comparison of all approaches (basic version)

            figure('Position', [50, 50, 1600, 800]);

            % Plot 1: R² comparison
            subplot(2, 4, 1);
            r2_scores = [results.rate.r2, results.voltage.r2, results.hybrid.r2];
            bar(r2_scores, 'FaceColor', [0.3 0.7 0.9]);
            set(gca, 'XTickLabel', {'Rate', 'Voltage', 'Hybrid'});
            ylabel('R² Score');
            title('Threshold Prediction R²');
            grid on;

            % Plot 2: VP distance comparison
            subplot(2, 4, 2);
            vp_scores = [results.rate.best_vp, results.voltage.best_vp, results.hybrid.best_vp];
            bar(vp_scores, 'FaceColor', [0.9 0.7 0.3]);
            set(gca, 'XTickLabel', {'Rate', 'Voltage', 'Hybrid'});
            ylabel('Victor-Purpura Distance');
            title('Spike Train Quality');
            grid on;

            % Plot 3-4: Example threshold traces
            subplot(2, 4, [3, 4]);
            t_plot = (1:length(obj.Vm_recorded)) * obj.dt;
            plot(t_plot, results.rate.threshold_trace, 'r-', 'LineWidth', 1, 'DisplayName', 'Rate Model');
            hold on;
            plot(t_plot, results.voltage.threshold_trace, 'g-', 'LineWidth', 1, 'DisplayName', 'Voltage Model');
            plot(t_plot, results.hybrid.threshold_trace, 'b-', 'LineWidth', 1, 'DisplayName', 'Hybrid Model');

            % Mark true spikes
            spike_times = obj.elbow_indices * obj.dt;
            scatter(spike_times, obj.threshold_values, 20, 'ko', 'filled', 'DisplayName', 'True Thresholds');

            xlabel('Time (s)');
            ylabel('Threshold (mV)');
            title('Predicted Threshold Traces');
            legend('Location', 'best');
            grid on;

            % Plot 5: Feature importance (hybrid model)
            subplot(2, 4, 5);
            if ~isempty(obj.hybrid_model)
                beta_weights = obj.hybrid_model.beta;
                rate_weights = beta_weights(1:obj.history_bins);
                voltage_weights = beta_weights((obj.history_bins+1):end);

                plot(1:obj.history_bins, abs(rate_weights), 'ro-', 'DisplayName', 'Rate Features');
                hold on;
                plot(1:length(voltage_weights), abs(voltage_weights), 'go-', 'DisplayName', 'Voltage Features');
                xlabel('Feature Index');
                ylabel('|Weight|');
                title('Feature Importance (Hybrid Model)');
                legend();
                grid on;
            end

            % Plot 6: Residuals analysis
            subplot(2, 4, 6);
            if ~isempty(obj.hybrid_model)
                [X_hybrid, y_hybrid] = obj.extract_hybrid_features();
                X_hybrid_std = zscore(X_hybrid);
                y_pred = X_hybrid_std * obj.hybrid_model.beta + obj.hybrid_model.intercept;
                residuals = y_hybrid - y_pred;

                scatter(y_pred, residuals, 30, 'filled');
                xlabel('Predicted Threshold (mV)');
                ylabel('Residuals (mV)');
                title('Residuals Analysis');
                yline(0, 'r--');
                grid on;
            end

            % Plot 7: Spike generation example
            subplot(2, 4, 7);
            best_approach = results.best_approach;
            best_threshold = results.(best_approach).threshold_trace;
            best_baseline = results.(best_approach).best_baseline;

            % Generate spikes with best model
            [spike_times_pred, spike_indices_pred] = obj.generate_spikes_hybrid(best_threshold, best_baseline);

            % Show spike raster for a zoom window
            zoom_start_time = 30;  % seconds
            zoom_duration = 10;    % seconds
            zoom_end_time = zoom_start_time + zoom_duration;

            % True spikes in zoom window
            true_spike_times = obj.elbow_indices * obj.dt;
            true_zoom_spikes = true_spike_times(true_spike_times >= zoom_start_time & true_spike_times <= zoom_end_time);

            % Predicted spikes in zoom window
            pred_zoom_spikes = spike_times_pred(spike_times_pred >= zoom_start_time & spike_times_pred <= zoom_end_time);

            % Plot raster
            plot(true_zoom_spikes, ones(size(true_zoom_spikes)), 'bo', 'MarkerSize', 8, 'DisplayName', 'True Spikes');
            hold on;
            plot(pred_zoom_spikes, 2*ones(size(pred_zoom_spikes)), 'r^', 'MarkerSize', 8, 'DisplayName', 'Predicted Spikes');

            ylim([0.5, 2.5]);
            xlim([zoom_start_time, zoom_end_time]);
            xlabel('Time (s)');
            set(gca, 'YTick', [1, 2], 'YTickLabel', {'True', 'Predicted'});
            title(sprintf('Spike Timing (%s Model)', upper(best_approach)));
            legend();
            grid on;

            % Plot 8: Summary statistics
            subplot(2, 4, 8);
            axis off;
            summary_text = {
                'COMPARISON SUMMARY';
                '';
                sprintf('Best Approach: %s', upper(results.best_approach));
                sprintf('Best R²: %.3f', results.(results.best_approach).r2);
                sprintf('Best VP: %.3f', results.(results.best_approach).best_vp);
                sprintf('Best Baseline: %.1f mV', results.(results.best_approach).best_baseline);
                '';
                'SPIKE COUNTS:';
                sprintf('True spikes: %d', length(obj.elbow_indices));
                sprintf('Predicted spikes: %d', length(spike_indices_pred));
                sprintf('Rate accuracy: %.1f%%', ...
                100 * min(length(spike_indices_pred), length(obj.elbow_indices)) / ...
                max(length(spike_indices_pred), length(obj.elbow_indices)));
                '';
                'MODEL PERFORMANCE:';
                sprintf('Rate R²: %.3f', results.rate.r2);
                sprintf('Voltage R²: %.3f', results.voltage.r2);
                sprintf('Hybrid R²: %.3f', results.hybrid.r2);
                };

            text(0.05, 0.95, summary_text, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
                'FontSize', 10, 'FontName', 'FixedWidth');

            sgtitle(sprintf('Hybrid Threshold Prediction Comparison: %s (%s)', obj.cell_id, obj.cell_type), ...
                'FontSize', 14, 'FontWeight', 'bold');
        end

        function plot_comprehensive_analysis(obj, results)
            % Enhanced plotting with validation results

            figure('Position', [50, 50, 1800, 1200]);

            % Plot 1: R² comparison (train vs validation)
            subplot(3, 4, 1);
            model_names = {'Rate', 'Voltage', 'Hybrid'};
            train_r2 = [results.rate.train_r2, results.voltage.train_r2, results.hybrid.train_r2];
            val_r2 = [results.rate.val_r2, results.voltage.val_r2, results.hybrid.val_r2];

            x = 1:3;
            width = 0.35;
            bar(x - width/2, train_r2, width, 'FaceColor', [0.3 0.7 0.9], 'DisplayName', 'Training');
            hold on;
            bar(x + width/2, val_r2, width, 'FaceColor', [0.9 0.7 0.3], 'DisplayName', 'Validation');
            set(gca, 'XTickLabel', model_names);
            ylabel('R² Score');
            title('Model Performance');
            legend();
            grid on;

            % Plot 2: VP distance comparison
            subplot(3, 4, 2);
            vp_scores = [results.rate.best_vp, results.voltage.best_vp, results.hybrid.best_vp];
            bar(vp_scores, 'FaceColor', [0.7 0.9 0.3]);
            set(gca, 'XTickLabel', model_names);
            ylabel('Victor-Purpura Distance');
            title('Spike Train Quality');
            grid on;

            % Plot 3-4: Threshold traces comparison
            subplot(3, 4, [3, 4]);
            t_plot = (1:length(obj.Vm_recorded)) * obj.dt;
            plot(t_plot, results.rate.threshold_trace, 'r-', 'LineWidth', 1, 'DisplayName', 'Rate Model');
            hold on;
            plot(t_plot, results.voltage.threshold_trace, 'g-', 'LineWidth', 1, 'DisplayName', 'Voltage Model');
            plot(t_plot, results.hybrid.threshold_trace, 'b-', 'LineWidth', 1, 'DisplayName', 'Hybrid Model');

            % Mark true spikes
            spike_times = obj.elbow_indices * obj.dt;
            scatter(spike_times, obj.threshold_values, 20, 'ko', 'filled', 'DisplayName', 'True Thresholds');

            xlabel('Time (s)');
            ylabel('Threshold (mV)');
            title('Predicted Threshold Traces');
            legend('Location', 'best');
            grid on;

            % Plot 5: Feature importance (hybrid model)
            subplot(3, 4, 5);
            if ~isempty(obj.hybrid_model)
                beta_weights = obj.hybrid_model.beta;
                rate_weights = beta_weights(1:obj.history_bins);
                voltage_weights = beta_weights((obj.history_bins+1):end);

                plot(1:obj.history_bins, abs(rate_weights), 'ro-', 'DisplayName', 'Rate Features');
                hold on;
                plot(1:length(voltage_weights), abs(voltage_weights), 'go-', 'DisplayName', 'Voltage Features');
                xlabel('Feature Index');
                ylabel('|Weight|');
                title('Feature Importance (Hybrid)');
                legend();
                grid on;
            end

            % Plot 6: Validation scatter plot
            subplot(3, 4, 6);
            if ~isempty(obj.hybrid_model)
                % Get validation predictions
                val_indices = round(0.7 * length(obj.elbow_indices)):length(obj.elbow_indices);
                val_spike_indices = obj.elbow_indices(val_indices);
                val_true = obj.threshold_values(val_indices);
                val_pred = results.hybrid.threshold_trace(val_spike_indices);

                scatter(val_true, val_pred, 30, 'filled');
                hold on;
                plot([min(val_true), max(val_true)], [min(val_true), max(val_true)], 'r--');
                xlabel('True Threshold (mV)');
                ylabel('Predicted Threshold (mV)');
                title(sprintf('Validation (R² = %.3f)', results.hybrid.val_r2));
                grid on;
            end

            % Plot 7-8: Spike train comparison (zoom window)
            subplot(3, 4, [7, 8]);
            zoom_start = round(0.3 * length(obj.Vm_recorded));
            zoom_end = round(0.4 * length(obj.Vm_recorded));
            zoom_indices = zoom_start:zoom_end;
            t_zoom = t_plot(zoom_indices);

            plot(t_zoom, obj.Vm_recorded(zoom_indices), 'k-', 'LineWidth', 1, 'DisplayName', 'Recorded');
            hold on;

            % Generate spikes with best model
            best_threshold = results.(results.best_approach).threshold_trace;
            best_baseline = results.(results.best_approach).best_baseline;
            [spike_times_pred, spike_indices_pred] = obj.generate_spikes_hybrid(best_threshold, best_baseline);

            % Plot predicted spikes in zoom window
            zoom_spike_mask = spike_times_pred >= t_zoom(1) & spike_times_pred <= t_zoom(end);
            zoom_spike_times = spike_times_pred(zoom_spike_mask);
            if ~isempty(zoom_spike_times)
                plot(zoom_spike_times, best_baseline * ones(size(zoom_spike_times)), ...
                    'r^', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'DisplayName', 'Predicted Spikes');
            end

            % Plot true spikes in zoom window
            true_spike_times = obj.elbow_indices * obj.dt;
            zoom_true_mask = true_spike_times >= t_zoom(1) & true_spike_times <= t_zoom(end);
            zoom_true_times = true_spike_times(zoom_true_mask);
            if ~isempty(zoom_true_times)
                plot(zoom_true_times, obj.threshold_values(zoom_true_mask), ...
                    'bo', 'MarkerSize', 6, 'MarkerFaceColor', 'b', 'DisplayName', 'True Spikes');
            end

            % Plot threshold trace
            plot(t_zoom, best_threshold(zoom_indices) + best_baseline, 'g--', ...
                'LineWidth', 2, 'DisplayName', 'Predicted Threshold');

            xlabel('Time (s)');
            ylabel('Voltage (mV)');
            title(sprintf('Spike Prediction (%s Model)', upper(results.best_approach)));
            legend('Location', 'best');
            grid on;

            % Plot 9: Cross-validation performance
            subplot(3, 4, 9);
            generalization = [results.rate.val_r2/results.rate.train_r2, ...
                results.voltage.val_r2/results.voltage.train_r2, ...
                results.hybrid.val_r2/results.hybrid.train_r2];
            bar(generalization, 'FaceColor', [0.6 0.4 0.8]);
            set(gca, 'XTickLabel', model_names);
            ylabel('Validation/Training R²');
            title('Generalization Performance');
            yline(1.0, 'r--', 'Perfect Generalization');
            grid on;

            % Plot 10: ISI comparison
            subplot(3, 4, 10);
            true_isi = diff(obj.elbow_indices) * obj.dt * 1000;  % ms
            pred_isi = diff(spike_indices_pred) * obj.dt * 1000;  % ms

            edges = 0:5:100;
            histogram(true_isi, edges, 'FaceColor', 'b', 'FaceAlpha', 0.5, 'DisplayName', 'True ISI');
            hold on;
            histogram(pred_isi, edges, 'FaceColor', 'r', 'FaceAlpha', 0.5, 'DisplayName', 'Predicted ISI');
            xlabel('ISI (ms)');
            ylabel('Count');
            title('ISI Distribution');
            legend();
            grid on;

            % Plot 11: Threshold prediction accuracy over time
            subplot(3, 4, 11);
            window_size = round(0.1 / obj.dt);  % 100ms windows
            n_windows = floor(length(obj.Vm_recorded) / window_size);
            time_windows = [];
            threshold_errors = [];

            for w = 1:n_windows
                window_start = (w-1) * window_size + 1;
                window_end = w * window_size;

                % Find spikes in this window
                window_spikes = obj.elbow_indices(obj.elbow_indices >= window_start & ...
                    obj.elbow_indices <= window_end);

                if ~isempty(window_spikes)
                    true_thresh = obj.threshold_values(ismember(obj.elbow_indices, window_spikes));
                    pred_thresh = best_threshold(window_spikes);

                    window_error = sqrt(mean((true_thresh - pred_thresh).^2));
                    time_windows(end+1) = (window_start + window_end) / 2 * obj.dt;
                    threshold_errors(end+1) = window_error;
                end
            end

            if ~isempty(time_windows)
                plot(time_windows, threshold_errors, 'b-o', 'MarkerSize', 4);
                xlabel('Time (s)');
                ylabel('RMSE (mV)');
                title('Threshold Prediction Error');
                grid on;
            end

            % Plot 12: Summary statistics
            subplot(3, 4, 12);
            axis off;
            summary_text = {
                'HYBRID MODEL SUMMARY';
                '';
                sprintf('Best Approach: %s', upper(results.best_approach));
                sprintf('Validation R²: %.3f', results.(results.best_approach).val_r2);
                sprintf('VP Distance: %.3f', results.(results.best_approach).best_vp);
                sprintf('Baseline θ₀: %.1f mV', results.(results.best_approach).best_baseline);
                '';
                'SPIKE STATISTICS:';
                sprintf('True spikes: %d', length(obj.elbow_indices));
                sprintf('Predicted spikes: %d', length(spike_indices_pred));
                sprintf('Rate accuracy: %.1f%%', ...
                100 * min(length(spike_indices_pred), length(obj.elbow_indices)) / ...
                max(length(spike_indices_pred), length(obj.elbow_indices)));
                '';
                'FEATURE WINDOWS:';
                sprintf('Rate history: %.0f ms', obj.history_window_ms);
                sprintf('Voltage history: %.0f ms', obj.voltage_window_ms);
                };

            text(0.05, 0.95, summary_text, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
                'FontSize', 10, 'FontName', 'FixedWidth');

            sgtitle(sprintf('Hybrid Threshold Prediction Analysis: %s (%s)', obj.cell_id, obj.cell_type), ...
                'FontSize', 14, 'FontWeight', 'bold');
        end
    end
end


%% UTILITY FUNCTIONS

function model = ridge_regression(X, y, alpha)
% Ridge regression with L2 regularization
% X: feature matrix (n_samples × n_features)
% y: target vector (n_samples × 1)
% alpha: regularization parameter

[n_samples, n_features] = size(X);

% Add regularization to normal equation
I = eye(n_features);
beta = (X' * X + alpha * I) \ (X' * y);

% Calculate intercept
intercept = mean(y) - mean(X) * beta;

model.beta = beta;
model.intercept = intercept;
model.alpha = alpha;
end

function r2 = calculate_r2(y_true, y_pred)
% Calculate R² coefficient of determination
ss_res = sum((y_true - y_pred).^2);
ss_tot = sum((y_true - mean(y_true)).^2);
r2 = 1 - (ss_res / ss_tot);
end

function [X_std, X_mean, X_std_dev] = standardize_features(X)
% Standardize features (z-score normalization) and return parameters
X_mean = mean(X, 1);
X_std_dev = std(X, 0, 1);

% Avoid division by zero
X_std_dev(X_std_dev == 0) = 1;

X_std = (X - X_mean) ./ X_std_dev;
end

%% EXAMPLE USAGE SCRIPT WITH INHERITANCE

function demo_hybrid_threshold_predictor()
% Example usage demonstrating the hybrid threshold predictor

fprintf('=== HYBRID THRESHOLD PREDICTOR DEMO ===\n');

% Check if data is loaded
if ~exist('Vm_all', 'var') || ~exist('Vm_cleaned', 'var')
    error(['Please load your experimental data first:\n' ...
        '  - Vm_all: Raw voltage trace with spikes\n' ...
        '  - Vm_cleaned: Subthreshold voltage (spikes removed)\n' ...
        '  - dt: Time step\n']);
end

% Initialize hybrid predictor (inherits from SpikeResponseModel)
predictor = HybridThresholdPredictor(Vm_all, Vm_cleaned, dt, ...
    'vm_thresh', -20, 'd2v_thresh', 50, 'search_back_ms', 2, ...
    'cell_id', 'Demo-Cell', 'cell_type', 'Ganglion');

% Customize parameters if needed
predictor.history_window_ms = 150;  % 150ms firing rate history
predictor.history_bins = 25;        % 25 temporal bins
predictor.voltage_window_ms = 30;   % 30ms voltage history
predictor.alpha_hybrid = 0.3;       % Lower regularization for hybrid

% Compare approaches with validation
results = predictor.compare_approaches_with_validation(0.3);  % 30% validation

% Create comprehensive analysis plots
predictor.plot_comprehensive_analysis(results);

% Generate final spike train with best model
best_approach = results.best_approach;
best_threshold_trace = results.(best_approach).threshold_trace;
best_baseline = results.(best_approach).best_baseline;

fprintf('\n=== FINAL SPIKE GENERATION ===\n');
[final_spike_times, final_spike_indices] = predictor.generate_spikes_hybrid(...
    best_threshold_trace, best_baseline);

% Calculate final Victor-Purpura distance
true_spike_times = predictor.elbow_indices * predictor.dt;
final_vp = spkd_c(final_spike_times, true_spike_times, ...
    length(final_spike_times), length(true_spike_times), 4);

% Run diagnostics using inherited method
fprintf('\n=== RUNNING INHERITED SRM DIAGNOSTICS ===\n');

% Create custom kernel for diagnostics
kernel_fn = @(t) interp1(0:predictor.dt:(length(best_threshold_trace)-1)*predictor.dt, ...
    best_threshold_trace - best_threshold_trace(1), t, 'linear', 'extrap');

% Use inherited simulate method for full diagnostics
[spike_vec, V_pred, threshold_trace_full, spike_times_diag, spike_V_diag] = ...
    predictor.simulate(best_baseline, kernel_fn);

% Call inherited diagnostics method
zoom_xlim = [30, 35];  % 5-second zoom window
predictor.diagnostics(V_pred, threshold_trace_full, spike_vec, final_vp, zoom_xlim, ...
    [], [best_baseline; results.(best_approach).best_vp], spike_times_diag, spike_V_diag);

% Final summary
fprintf('\n=== FINAL RESULTS SUMMARY ===\n');
fprintf('Best approach: %s\n', upper(best_approach));
fprintf('Training R²: %.3f\n', results.(best_approach).train_r2);
fprintf('Validation R²: %.3f\n', results.(best_approach).val_r2);
fprintf('Victor-Purpura distance: %.3f\n', final_vp);
fprintf('Spike count accuracy: %.1f%%\n', ...
    100 * min(length(final_spike_times), length(true_spike_times)) / ...
    max(length(final_spike_times), length(true_spike_times)));
fprintf('Rate accuracy: %.1f%% (%.1f Hz pred vs %.1f Hz true)\n', ...
    100 * length(final_spike_times) / length(true_spike_times), ...
    length(final_spike_times) / (length(predictor.Vm_recorded) * predictor.dt), ...
    length(true_spike_times) / (length(predictor.Vm_recorded) * predictor.dt));

fprintf('\n✅ Hybrid threshold prediction analysis complete!\n');
fprintf('������ See figures for detailed analysis\n');
fprintf('������ Inherits all SRM functionality plus data-driven features\n');
end

% Uncomment to run demo:
% demo_hybrid_threshold_predictor();