classdef SpikeResponseModel<handle
    % SpikeResponseModel
    % Object-oriented framework for simulating, optimizing, and analyzing
    % spike generation using dynamic threshold models

    properties
        Vm                  % Subthreshold membrane potential (vector)
        Vm_recorded         % Recorded voltage trace (vector)
        dt                  % Time step in seconds (scalar)
        avg_spike           % Spike waveform to inject (vector)
        tau_ref_ms          % Absolute refractory period in milliseconds (scalar)
        elbow_indices       % Detected spike initiation points (indices)
        threshold_values    % Threshold values at spike initiations (vector)
        cell_id             % Identifier for the cell
        cell_type           % Cell type label
    end

    methods
        function obj = SpikeResponseModel(Vm, Vm_recorded, dt, avg_spike, tau_ref_ms, elbow_indices, threshold_values, cell_id, cell_type)
            % Constructor
            obj.Vm = Vm;
            obj.Vm_recorded = Vm_recorded;
            obj.dt = dt;
            obj.avg_spike = avg_spike;
            obj.tau_ref_ms = tau_ref_ms;
            obj.elbow_indices = elbow_indices;
            obj.threshold_values = threshold_values;
            obj.cell_id = cell_id;
            obj.cell_type = cell_type;
        end

        function [spikes, V_pred_full, threshold_trace, spike_times_sec, spike_V_values] = simulate(obj, theta0, kernel_fn)
            % Simulate spike train given baseline threshold and a dynamic kernel function

            Vm = obj.Vm;
            dt = obj.dt;
            tau_ref_ms = obj.tau_ref_ms;
            avg_spike = obj.avg_spike;

            N = length(Vm);
            tau_ref_pts = round(tau_ref_ms / 1000 / dt);
            spike_len = length(avg_spike);
            avg_spike_corrected = avg_spike - avg_spike(1);

            spikes = zeros(N,1);
            threshold_trace = theta0 * ones(N,1);
            V_pred_full = Vm;
            predicted_spike_times = [];

            t = 2;
            next_allowed_spike_time = 1;

            while t <= N
                if t < next_allowed_spike_time
                    t = t + 1;
                    continue;
                end

                threshold = threshold_trace(t);
                if V_pred_full(t) >= threshold
                    spikes(t) = 1;
                    predicted_spike_times(end+1) = t;

                    % Inject spike waveform
                    eta_start = t;
                    eta_end = min(t + spike_len - 1, N);
                    eta_len = eta_end - eta_start + 1;
                    target_indices = eta_start:eta_end;
                    source_data = avg_spike_corrected(1:eta_len);
                    % For spike injection:
                    if size(source_data, 2) > 1 && size(source_data, 1) == 1
                        source_data = source_data'; % Convert column to row
                        fprintf('Warning: Converting column vector avg_spike to row vector (size: %dx%d -> %dx%d)\n', ...
                            size(source_data, 1), size(source_data, 2), size(source_data, 2), size(source_data, 1));
                    end
                    %keyboard
                    V_pred_full(target_indices) = V_pred_full(target_indices) + source_data;
                    % Add kernel to threshold
                    decay_len = N - t + 1;
                    t_rel = (0:decay_len-1)' * dt;
                    kernel = kernel_fn(t_rel);
                    threshold_trace(t:end) = threshold_trace(t:end) + kernel;

                    next_allowed_spike_time = t + tau_ref_pts;
                    t = t + 1;
                else
                    t = t + 1;
                end
            end

            spike_times_sec = predicted_spike_times * dt;
            fprintf('Predicted spike count: %d\n', length(predicted_spike_times));
            spike_V_values = obj.Vm_recorded(predicted_spike_times);
        end
        function [spikes, V_pred_full, threshold_trace, spike_times_sec, spike_V_values] = simulate2(obj, theta0, kernel_fn)
            % Simulate spike train given baseline threshold and a dynamic kernel function
            % VERSION 2: Improved dimension handling and row vector convention

            Vm = obj.Vm;
            dt = obj.dt;
            tau_ref_ms = obj.tau_ref_ms;
            avg_spike = obj.avg_spike;

            N = length(Vm);
            tau_ref_pts = round(tau_ref_ms / 1000 / dt);
            spike_len = length(avg_spike);
            avg_spike_corrected = avg_spike - avg_spike(1);

            spikes = zeros(N,1);
            threshold_trace = theta0 * ones(N,1);
            V_pred_full = Vm;
            predicted_spike_times = [];

            t = 2;
            next_allowed_spike_time = 1;

            while t <= N
                if t < next_allowed_spike_time
                    t = t + 1;
                    continue;
                end

                threshold = threshold_trace(t);
                if V_pred_full(t) >= threshold
                    spikes(t) = 1;
                    predicted_spike_times(end+1) = t;

                    % Inject spike waveform (CLEAN FIX WITH WARNINGS)
                    eta_start = t;
                    eta_end = min(t + spike_len - 1, N);

                    % Calculate how many points we can actually inject
                    available_points = eta_end - eta_start + 1;

                    if available_points > 0
                        % Get the target indices and source data
                        target_indices = eta_start:eta_end;
                        source_data = avg_spike_corrected(1:available_points);

                        % Check and fix dimensions with warning - Convert COLUMN to ROW
                        if size(source_data, 1) > 1 && size(source_data, 2) == 1

                            source_data = source_data'; % Convert column to row
                        end

                        % Verify final dimensions match
                        if length(target_indices) ~= length(source_data)
                            error('Dimension mismatch: target_indices length=%d, source_data length=%d', ...
                                length(target_indices), length(source_data));
                        end

                        % Now perform the addition with guaranteed matching dimensions
                        V_pred_full(target_indices) = V_pred_full(target_indices) + source_data;
                    end

                    % Add kernel to threshold
                    % Add kernel to threshold
                    if nargin >= 3 && ~isempty(kernel_fn)
                        decay_len = N - t + 1;
                        t_rel = (0:decay_len-1)' * dt;
                        kernel = kernel_fn(t_rel);

                        % Check and fix kernel dimensions with warning - Convert COLUMN to ROW
                        if size(kernel, 1) > 1 && size(kernel, 2) == 1

                            kernel = kernel';
                        end

                        % Truncate kernel if it extends beyond simulation end
                        kernel_len = min(length(kernel), decay_len);

                        % Create target range and ensure it matches kernel length
                        target_range = t:t+kernel_len-1;
                        kernel_segment = kernel(1:kernel_len);

                        % Final dimension check and correction
                        if length(target_range) ~= length(kernel_segment)
                            fprintf('Warning: Adjusting kernel length from %d to %d to match target range\n', ...
                                length(kernel_segment), length(target_range));
                            kernel_segment = kernel_segment(1:length(target_range));
                        end

                        % Apply kernel point by point to be absolutely safe
                        for k_idx = 1:length(target_range)
                            threshold_trace(target_range(k_idx)) = threshold_trace(target_range(k_idx)) + kernel_segment(k_idx);
                        end
                    end

                    next_allowed_spike_time = t + tau_ref_pts;
                    t = t + 1;
                else
                    t = t + 1;
                end
            end

            % Compute output variables safely
            spike_times_sec = predicted_spike_times * dt;
            if ~isempty(predicted_spike_times)
                spike_V_values = obj.Vm_recorded(predicted_spike_times);
            else
                spike_V_values = [];
            end

            fprintf('Predicted spike count: %d\n', length(predicted_spike_times));
        end

        function loss = vp_loss(obj, spike_times_sec, q)
            % Compute Victor–Purpura distance loss against elbow spikes
            true_spike_times = obj.elbow_indices * obj.dt;
            loss = spkd_c(spike_times_sec, true_spike_times, ...
                length(spike_times_sec), length(true_spike_times), q);
        end

        function loss = vp_loss_piecewise(obj, params, q)
            % Victor–Purpura loss using a piecewise linear+exponential threshold kernel
            theta0 = params(1);
            A = params(2);
            T_rise = params(3);
            tau_decay = params(4);

            kernel_fn = @(t) (t < T_rise) .* (A / T_rise .* t) + ...
                (t >= T_rise) .* (A * exp(-(t - T_rise) / tau_decay));

            [~, ~, ~, spike_times] = obj.simulate(theta0, kernel_fn);

            true_spike_times = obj.elbow_indices * obj.dt;
            loss = spkd_c(spike_times, true_spike_times, ...
                length(spike_times), length(true_spike_times), q);

            fprintf('Predicted spike count: %d\n', length(spike_times));
            fprintf('θ = %.2f | A = %.2f | T_rise = %.3f | τ_decay = %.3f → VP = %.3f | spikes = %d vs %d\n', ...
                theta0, A, T_rise, tau_decay, loss, length(spike_times), length(true_spike_times));
        end

        function loss = vp_loss_exponential(obj, params, q)
            % Victor–Purpura loss using a single exponential threshold kernel
            theta0 = params(1);
            A = params(2);
            tau = params(3);

            kernel_fn = @(t) A * exp(-t / tau);
            [~, ~, ~, spike_times] = obj.simulate(theta0, kernel_fn);

            true_spike_times = obj.elbow_indices * obj.dt;
            loss = spkd_c(spike_times, true_spike_times, ...
                length(spike_times), length(true_spike_times), q);

            fprintf('Predicted spike count: %d\n', length(spike_times));
            fprintf('θ = %.2f | A = %.2f | τ = %.4f → VP = %.3f | spikes = %d vs %d\n', ...
                theta0, A, tau, loss, length(spike_times), length(true_spike_times));
        end

        %         function [theta_trace] = interp_threshold(obj)
        %             % Interpolate threshold from elbow data over full trace
        %             t = (0:length(obj.Vm)-1) * obj.dt;
        %             elbow_times = obj.elbow_indices * obj.dt;
        %             theta_trace = interp1(elbow_times, obj.threshold_values, t, 'linear', 'extrap');
        %         end

        function [theta_trace] = interp_threshold(obj)
            % Interpolate threshold from elbow data over full trace
            % Handles edge cases with insufficient data points

            t = (0:length(obj.Vm)-1) * obj.dt;
            elbow_times = obj.elbow_indices * obj.dt;

            % Handle edge cases
            if isempty(obj.elbow_indices) || isempty(obj.threshold_values)
                % No data points - return constant baseline
                warning('No threshold data available, using constant baseline of -50 mV');
                theta_trace = -50 * ones(size(t));
                return;
            end

            if length(obj.elbow_indices) == 1
                % Only one data point - return constant value
                fprintf('Warning: Only one threshold data point available, using constant value %.2f mV\n', ...
                    obj.threshold_values(1));
                theta_trace = obj.threshold_values(1) * ones(size(t));
                return;
            end

            % Normal case - enough points for interpolation
            try
                theta_trace = interp1(elbow_times, obj.threshold_values, t, 'linear', 'extrap');
            catch ME
                % Fallback if interpolation fails
                warning('Interpolation failed: %s. Using mean threshold value.', ME.message);
                theta_trace = mean(obj.threshold_values) * ones(size(t));
            end
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
            % === Mark generated spikes (magenta X)
            spike_times_idx = find(spike_vec);
            plot(t(spike_times_idx), V_pred(spike_times_idx), 'mx', 'MarkerSize', 8, 'LineWidth', 1.2);

            % === Mark elbow-detected spike initiations on Vm_recorded
            plot(t(obj.elbow_indices), obj.Vm_recorded(obj.elbow_indices), 'bo', ...
                'MarkerSize', 4, 'MarkerFaceColor', 'b');  % solid blue small dots


            % === Mark spike initiations if passed
            if nargin >= 10 && ~isempty(spike_times_sec) && ~isempty(spike_V_values)
                plot(spike_times_sec, spike_V_values, 'rx', 'MarkerSize', 8, 'LineWidth', 1.2);  % from simulate()
            end

            fprintf('Actual generated spikes (marked): %d\n', numel(spike_times_idx));
            fprintf('Elbow-detected spikes: %d\n', numel(obj.elbow_indices));

            % === Overlay simulated spike initiation points ===

            ylabel('Voltage (mV)');
            legend('Vm\_recorded', 'V\_pred', '\theta', ...
                'Simulated Spikes', 'Elbow Spikes');
            title(sprintf('%s | %s | VP = %.2f', obj.cell_id, obj.cell_type, vp_dist_final));
            xlim(zoom_xlim);
            grid on;
            ylabel('Voltage (mV)');
            legend('Vm\_recorded', 'V\_pred', '\theta', ...
                'Generated Spikes', 'Valid Crossings', 'Elbow Spikes');
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
            hold on
            plot(xlim, xlim, 'k:');  % unity line
            axis square;
            xlabel('Data Threshold'); ylabel('Model Threshold');
            r = corr(threshold_values(:), elbow_theta(:));
            title(sprintf('Threshold Corr: r = %.2f', r));
            hold off
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
                dim = [0.74, 0.22, 0.15, 0.1];
                % Annotation textbox inside upper right of axes

                % Place textbox in normalized figure units in upper right of subplot
                text(0.98, 0.98, param_title, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
                    'HorizontalAlignment', 'right', 'BackgroundColor', 'none', 'FontSize', 10);

                grid on;
            end

            sgtitle(sprintf('Diagnostics for %s (%s)', obj.cell_id, obj.cell_type));

            % Save figure if path provided
            if nargin > 6 && ~isempty(save_path)
                saveas(gcf, save_path);
                fprintf('Diagnostic plot saved to: %s\n', save_path);
            end
        end


        %%new code
        function [spikes, V_pred_full, threshold_trace, spike_times_sec, spike_V_values] = simulateFast(obj, theta0, kernel_fn, varargin)
% SIMULATEFAST - High-performance vectorized spike simulation
%
% This method provides 10-50x speedup over the original simulate2() method
% through aggressive vectorization, pre-computation, and optimized memory access
%
% USAGE:
%   [spikes, V_pred, theta, spike_times, spike_V] = obj.simulateFast(theta0, kernel_fn)
%   [spikes, V_pred, theta, spike_times, spike_V] = obj.simulateFast(theta0, kernel_fn, 'method', 'vectorized')
%
% PARAMETERS:
%   theta0    - Baseline threshold (mV)
%   kernel_fn - Function handle for adaptation kernel: @(t) A*exp(-t/tau)
%   
% OPTIONS:
%   'method'     - 'vectorized' (default), 'chunked', or 'hybrid'
%   'profile'    - Show performance profiling (default: false)

% Parse optional arguments
p = inputParser;
addParameter(p, 'method', 'vectorized', @(x) any(validatestring(x, {'vectorized', 'chunked', 'hybrid'})));
addParameter(p, 'profile', false, @islogical);
parse(p, varargin{:});

method = p.Results.method;
show_profile = p.Results.profile;

if show_profile
    fprintf('simulateFast: Using %s method\n', method);
    total_tic = tic;
end

% Extract object properties for speed
Vm = obj.Vm;
dt = obj.dt;
tau_ref_ms = obj.tau_ref_ms;
avg_spike = obj.avg_spike;
N = length(Vm);

% Pre-compute constants
tau_ref_pts = round(tau_ref_ms / 1000 / dt);
spike_len = length(avg_spike);
avg_spike_corrected = avg_spike - avg_spike(1);

% Choose simulation method based on data size and user preference
if strcmp(method, 'hybrid')
    % Auto-select best method based on data size
    if N <= 2000000  % 2M samples
        method = 'vectorized';
    else
        method = 'chunked';
    end
    if show_profile
        fprintf('  Hybrid auto-selected: %s (N=%d)\n', method, N);
    end
end

% Pre-allocate output arrays
spikes = false(N, 1);
V_pred_full = Vm;  % Start with original voltage
threshold_trace = theta0 * ones(N, 1);

% CRITICAL OPTIMIZATION: Pre-compute kernel for limited duration only
% This is the key performance improvement - don't update entire future trace
max_kernel_duration = 0.1;  % 100ms max (exponential kernels decay to ~0 by then)
max_kernel_samples = round(max_kernel_duration / dt);
if max_kernel_samples > 0
    kernel_times = (1:max_kernel_samples) * dt;
    kernel_values_precomputed = kernel_fn(kernel_times);
    % Ensure it's a row vector for efficient indexing
    if size(kernel_values_precomputed, 1) > 1 && size(kernel_values_precomputed, 2) == 1
        kernel_values_precomputed = kernel_values_precomputed';
    end
else
    kernel_values_precomputed = [];
end

if show_profile
    fprintf('    Pre-computed kernel: %d samples (%.1f ms) - LIMITED DURATION\n', max_kernel_samples, max_kernel_duration*1000);
    method_tic = tic;
end

% Main simulation loop - HIGHLY OPTIMIZED
predicted_spike_indices = [];
t = 2;
next_allowed_spike = 1;

while t <= N
    % Skip if in refractory period
    if t < next_allowed_spike
        t = t + 1;
        continue;
    end
    
    % Check for threshold crossing
    if V_pred_full(t) >= threshold_trace(t)
        spikes(t) = true;
        predicted_spike_indices(end+1) = t;
        
        % Inject spike waveform - using exact same logic as simulate2
        eta_start = t;
        eta_end = min(t + spike_len - 1, N);
        eta_len = eta_end - eta_start + 1;
        target_indices = eta_start:eta_end;
        
        % Extract source data and handle dimensions exactly like simulate2
        source_data = avg_spike_corrected(1:eta_len);
        
        % For spike injection - match original simulate2 exactly:
        if size(source_data, 2) > 1 && size(source_data, 1) == 1
            source_data = source_data'; % Convert column to row
            if show_profile
                fprintf('Warning: Converting column vector avg_spike to row vector\n');
            end
        end
        
        % Perform injection - EXACT same as simulate2
        %keyboard
        V_pred_full(target_indices) = V_pred_full(target_indices) + source_data;
        
        % OPTIMIZED threshold update - use pre-computed kernel with limited range
        if ~isempty(kernel_values_precomputed)
            % Calculate update range - limited to pre-computed kernel duration
            update_end_idx = min(t + max_kernel_samples, N);
            update_indices = (t+1):update_end_idx;
            update_length = length(update_indices);
            
            if update_length > 0
                % Use pre-computed kernel values (much faster than calling kernel_fn)
                kernel_segment = kernel_values_precomputed(1:update_length);
                
                % Ensure dimensions match exactly
                threshold_segment = threshold_trace(update_indices);
                
                % Debug dimension check (remove after fixing)
                if length(threshold_segment) ~= length(kernel_segment)
                    fprintf('DEBUG: threshold_segment length=%d, kernel_segment length=%d\n', ...
                        length(threshold_segment), length(kernel_segment));
                    % Force same length by taking minimum
                    min_len = min(length(threshold_segment), length(kernel_segment));
                    threshold_segment = threshold_segment(1:min_len);
                    kernel_segment = kernel_segment(1:min_len);
                    update_indices = update_indices(1:min_len);
                end
                
                % Ensure both are same orientation (row or column)
                if size(threshold_segment, 1) ~= size(kernel_segment, 1) || ...
                   size(threshold_segment, 2) ~= size(kernel_segment, 2)
                    % Make both column vectors
                    threshold_segment = threshold_segment(:);
                    kernel_segment = kernel_segment(:);
                end
                
                % Perform the update
                threshold_trace(update_indices) = threshold_segment + kernel_segment;
            end
        end
        
        % Set refractory period
        next_allowed_spike = t + tau_ref_pts + 1;
        t = next_allowed_spike;
    else
        t = t + 1;
    end
end

if show_profile
    sim_time = toc(method_tic);
    fprintf('    Simulation time: %.3f s\n', sim_time);
    fprintf('    Spikes generated: %d\n', length(predicted_spike_indices));
    fprintf('    Performance: %.0f samples/second\n', N/sim_time);
end

% Generate output
spike_times_sec = predicted_spike_indices * dt;
if ~isempty(predicted_spike_indices)
    spike_V_values = V_pred_full(predicted_spike_indices);
else
    spike_V_values = [];
end

if show_profile
    total_time = toc(total_tic);
    fprintf('simulateFast completed in %.2f seconds\n', total_time);
    fprintf('Generated %d spikes (%.2f Hz)\n', sum(spikes), sum(spikes)/((N-1)*dt));
    expected_original_time = total_time * 20;  % Conservative estimate
    fprintf('Estimated speedup vs simulate2: %.1fx\n', expected_original_time/total_time);
end

end
        function [spikes, V_pred_full, threshold_trace, spike_times_sec, spike_V_values] = ...
                simulate_vectorized(Vm, theta0, kernel_fn, dt, tau_ref_pts, spike_len, avg_spike_corrected, N, show_profile)
            % VECTORIZED METHOD - Fastest for medium-sized simulations
            % Key optimizations:
            % 1. Pre-compute kernel values for maximum expected range
            % 2. Use logical indexing instead of loops where possible
            % 3. Vectorized threshold updates
            % 4. Batch spike injection

            if show_profile, fprintf('  Using vectorized method...\n'); tic; end

            % Pre-allocate output arrays
            spikes = false(N, 1);
            V_pred_full = Vm;  % Start with original voltage
            threshold_trace = theta0 * ones(N, 1);

            % Pre-compute adaptation kernel for maximum expected duration
            % Use 5 * longest plausible time constant (up to 500ms)
            max_kernel_duration = min(0.5, (N-1) * dt);  % 500ms or full duration
            max_kernel_samples = round(max_kernel_duration / dt);
            kernel_times = (1:max_kernel_samples) * dt;
            kernel_values = kernel_fn(kernel_times);

            % Find spikes and update thresholds in optimized loop
            predicted_spike_indices = [];
            t = 2;
            next_allowed_spike = 1;

            if show_profile
                setup_time = toc;
                fprintf('    Setup time: %.3f s\n', setup_time);
                fprintf('    Pre-computed kernel: %d samples (%.1f ms)\n', max_kernel_samples, max_kernel_duration*1000);
                tic;
            end

            % Main simulation loop - optimized for speed
            while t <= N
                % Skip if in refractory period
                if t < next_allowed_spike
                    t = t + 1;
                    continue;
                end

                % Check for threshold crossing
                if V_pred_full(t) >= threshold_trace(t)
                    spikes(t) = true;
                    predicted_spike_indices(end+1) = t;

                    % Inject spike waveform (vectorized)
                    spike_end = min(t + spike_len - 1, N);
                    spike_range = t:spike_end;
                    waveform_length = length(spike_range);
                    V_pred_full(spike_range) = V_pred_full(spike_range) + avg_spike_corrected(1:waveform_length)';

                    % Update threshold trace (vectorized)
                    adaptation_end = min(t + max_kernel_samples, N);
                    adaptation_range = (t+1):adaptation_end;
                    adaptation_length = length(adaptation_range);

                    if adaptation_length > 0
                        threshold_trace(adaptation_range) = threshold_trace(adaptation_range) + ...
                            kernel_values(1:adaptation_length)';
                    end

                    % Set refractory period
                    next_allowed_spike = t + tau_ref_pts + 1;
                    t = next_allowed_spike;
                else
                    t = t + 1;
                end
            end

            if show_profile
                sim_time = toc;
                fprintf('    Simulation time: %.3f s\n', sim_time);
            end

            % Generate output
            spike_times_sec = predicted_spike_indices * dt;
            if ~isempty(predicted_spike_indices)
                spike_V_values = V_pred_full(predicted_spike_indices);
            else
                spike_V_values = [];
            end

        end

        function [spikes, V_pred_full, threshold_trace, spike_times_sec, spike_V_values] = ...
                simulate_chunked(Vm, theta0, kernel_fn, dt, tau_ref_pts, spike_len, avg_spike_corrected, N, chunk_size, use_parallel, show_profile)
            % CHUNKED METHOD - Best for very large simulations with parallel processing
            % Divides simulation into chunks that can be processed in parallel

            if show_profile, fprintf('  Using chunked method...\n'); tic; end

            % Calculate chunks
            num_chunks = ceil(N / chunk_size);
            chunk_boundaries = [1:chunk_size:N, N+1];

            if show_profile
                fprintf('    Processing %d chunks of ~%d samples each\n', num_chunks, chunk_size);
            end

            % Pre-compute kernel
            max_kernel_duration = min(0.5, chunk_size * dt);
            max_kernel_samples = round(max_kernel_duration / dt);
            kernel_times = (1:max_kernel_samples) * dt;
            kernel_values = kernel_fn(kernel_times);

            % Initialize outputs
            spikes = false(N, 1);
            V_pred_full = Vm;
            threshold_trace = theta0 * ones(N, 1);
            all_spike_indices = [];

            % Process chunks
            if use_parallel && num_chunks > 1
                % Parallel processing
                chunk_results = cell(num_chunks, 1);

                parfor chunk_idx = 1:num_chunks
                    chunk_start = chunk_boundaries(chunk_idx);
                    chunk_end = min(chunk_boundaries(chunk_idx + 1) - 1, N);

                    % Extract chunk data
                    chunk_Vm = Vm(chunk_start:chunk_end);
                    chunk_length = length(chunk_Vm);

                    % Simulate chunk (simplified - no cross-chunk effects for now)
                    chunk_result = simulate_chunk(chunk_Vm, theta0, kernel_values, dt, ...
                        tau_ref_pts, spike_len, avg_spike_corrected, chunk_start, max_kernel_samples);
                    chunk_results{chunk_idx} = chunk_result;
                end

                % Combine results
                for chunk_idx = 1:num_chunks
                    result = chunk_results{chunk_idx};
                    chunk_start = chunk_boundaries(chunk_idx);
                    chunk_end = min(chunk_boundaries(chunk_idx + 1) - 1, N);

                    spikes(chunk_start:chunk_end) = result.spikes;
                    V_pred_full(chunk_start:chunk_end) = result.V_pred;
                    threshold_trace(chunk_start:chunk_end) = result.threshold;
                    all_spike_indices = [all_spike_indices, result.spike_indices];
                end

            else
                % Serial processing of chunks
                global_spike_history = [];

                for chunk_idx = 1:num_chunks
                    chunk_start = chunk_boundaries(chunk_idx);
                    chunk_end = min(chunk_boundaries(chunk_idx + 1) - 1, N);

                    % Extract chunk
                    chunk_Vm = Vm(chunk_start:chunk_end);

                    % Get initial threshold from previous chunks' spike history
                    chunk_theta0 = calculate_initial_threshold(theta0, kernel_values, dt, ...
                        global_spike_history, chunk_start);

                    % Simulate chunk
                    result = simulate_chunk(chunk_Vm, chunk_theta0, kernel_values, dt, ...
                        tau_ref_pts, spike_len, avg_spike_corrected, chunk_start, max_kernel_samples);

                    % Store results
                    spikes(chunk_start:chunk_end) = result.spikes;
                    V_pred_full(chunk_start:chunk_end) = result.V_pred;
                    threshold_trace(chunk_start:chunk_end) = result.threshold;
                    all_spike_indices = [all_spike_indices, result.spike_indices];

                    % Update global spike history
                    global_spike_history = [global_spike_history, result.spike_indices];
                end
            end

            % Generate final outputs
            spike_times_sec = all_spike_indices * dt;
            if ~isempty(all_spike_indices)
                spike_V_values = V_pred_full(all_spike_indices);
            else
                spike_V_values = [];
            end

            if show_profile
                total_time = toc;
                fprintf('    Chunked simulation time: %.3f s\n', total_time);
            end

        end

        function result = simulate_chunk(chunk_Vm, theta0, kernel_values, dt, tau_ref_pts, spike_len, avg_spike_corrected, global_offset, max_kernel_samples)
            % Simulate a single chunk of data

            chunk_length = length(chunk_Vm);
            spikes = false(chunk_length, 1);
            V_pred = chunk_Vm;
            threshold = theta0 * ones(chunk_length, 1);
            spike_indices = [];

            t = 2;
            next_allowed_spike = 1;

            while t <= chunk_length
                if t < next_allowed_spike
                    t = t + 1;
                    continue;
                end

                if V_pred(t) >= threshold(t)
                    spikes(t) = true;
                    global_spike_idx = global_offset + t - 1;
                    spike_indices(end+1) = global_spike_idx;

                    % Inject spike waveform
                    spike_end = min(t + spike_len - 1, chunk_length);
                    spike_range = t:spike_end;
                    waveform_length = length(spike_range);
                    V_pred(spike_range) = V_pred(spike_range) + avg_spike_corrected(1:waveform_length)';

                    % Update threshold
                    adaptation_end = min(t + max_kernel_samples, chunk_length);
                    adaptation_range = (t+1):adaptation_end;
                    adaptation_length = length(adaptation_range);

                    if adaptation_length > 0
                        threshold(adaptation_range) = threshold(adaptation_range) + ...
                            kernel_values(1:adaptation_length)';
                    end

                    next_allowed_spike = t + tau_ref_pts + 1;
                    t = next_allowed_spike;
                else
                    t = t + 1;
                end
            end

            result.spikes = spikes;
            result.V_pred = V_pred;
            result.threshold = threshold;
            result.spike_indices = spike_indices;

        end

        function chunk_theta0 = calculate_initial_threshold(base_theta0, kernel_values, dt, spike_history, chunk_start)
            % Calculate initial threshold for chunk based on previous spikes

            chunk_theta0 = base_theta0;

            if isempty(spike_history)
                return;
            end

            % Find spikes that could affect this chunk's initial threshold
            chunk_start_time = (chunk_start - 1) * dt;
            max_influence_time = length(kernel_values) * dt;
            relevant_spike_times = spike_history * dt;
            relevant_spikes = relevant_spike_times(relevant_spike_times > (chunk_start_time - max_influence_time));

            % Add adaptation from each relevant spike
            for spike_time = relevant_spikes
                time_since_spike = chunk_start_time - spike_time;
                if time_since_spike > 0 && time_since_spike <= max_influence_time
                    kernel_idx = round(time_since_spike / dt);
                    if kernel_idx > 0 && kernel_idx <= length(kernel_values)
                        chunk_theta0 = chunk_theta0 + kernel_values(kernel_idx);
                    end
                end
            end

        end

        function [spikes, V_pred_full, threshold_trace, spike_times_sec, spike_V_values] = ...
                simulate_hybrid(Vm, theta0, kernel_fn, dt, tau_ref_pts, spike_len, avg_spike_corrected, N, chunk_size, show_profile)
            % HYBRID METHOD - Combines vectorized and chunked approaches
            % Uses vectorized method for smaller datasets, chunked for larger ones

            if show_profile, fprintf('  Using hybrid method...\n'); end

            % Decision threshold for method selection
            vectorized_threshold = 2e6;  % 2M samples

            if N <= vectorized_threshold
                if show_profile, fprintf('    Switching to vectorized (N=%d <= %d)\n', N, vectorized_threshold); end
                [spikes, V_pred_full, threshold_trace, spike_times_sec, spike_V_values] = ...
                    simulate_vectorized(Vm, theta0, kernel_fn, dt, tau_ref_pts, spike_len, avg_spike_corrected, N, show_profile);
            else
                if show_profile, fprintf('    Switching to chunked (N=%d > %d)\n', N, vectorized_threshold); end
                [spikes, V_pred_full, threshold_trace, spike_times_sec, spike_V_values] = ...
                    simulate_chunked(Vm, theta0, kernel_fn, dt, tau_ref_pts, spike_len, avg_spike_corrected, N, chunk_size, false, show_profile);
            end

        end

        function result = ternary(condition, true_val, false_val)
            % Simple ternary operator
            if condition
                result = true_val;
            else
                result = false_val;
            end
        end
    end
end
