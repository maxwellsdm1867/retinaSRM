classdef SpikeResponseModel
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
                    V_pred_full(eta_start:eta_end) = ...
                        V_pred_full(eta_start:eta_end) + avg_spike_corrected(1:eta_len)';

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

            fprintf('✅ Predicted spike count: %d\n', length(spike_times));
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

        function [theta_trace] = interp_threshold(obj)
            % Interpolate threshold from elbow data over full trace
            t = (0:length(obj.Vm)-1) * obj.dt;
            elbow_times = obj.elbow_indices * obj.dt;
            theta_trace = interp1(elbow_times, obj.threshold_values, t, 'linear', 'extrap');
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

    end
end
