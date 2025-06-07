function [spikes, threshold_trace, V_pred_full, predicted_spike_times_sec] = ...
    simulate_dynamic_threshold_spikes(Vm, theta0, A, tau, dt, avg_spike, tau_ref_ms)

    % === Parameters ===
    N = length(Vm);
    tau_ref_pts = round(tau_ref_ms / 1000 / dt);   % refractory period in samples
    spike_len = length(avg_spike);
    avg_spike_corrected = avg_spike - avg_spike(1);  % baseline at 0

    % === Outputs ===
    spikes = zeros(N,1);
    threshold_trace = theta0 * ones(N,1);
    V_pred_full = Vm;
    predicted_spike_times = [];

    % === Initialize
    t = 2;
    next_allowed_spike_time = 1;

    while t <= N
        % --- Enforce absolute refractory ---
        if t < next_allowed_spike_time
            t = t + 1;
            continue;
        end

        % --- Evaluate dynamic threshold ---
        threshold = threshold_trace(t);

        % --- Spike condition ---
        if V_pred_full(t) >= threshold && V_pred_full(t-1) < threshold
            % Register spike
            spikes(t) = 1;
            predicted_spike_times(end+1) = t;

            % Add spike waveform
            eta_start = t;
            eta_end = min(t + spike_len - 1, N);
            eta_len = eta_end - eta_start + 1;
            V_pred_full(eta_start:eta_end) = ...
                V_pred_full(eta_start:eta_end) + avg_spike_corrected(1:eta_len)';

            % Add threshold exponential
            decay_len = N - t + 1;
            t_rel = (0:decay_len-1)' * dt;
            threshold_trace(t:end) = threshold_trace(t:end) + A * exp(-t_rel / tau);

            % Update refractory window
            next_allowed_spike_time = t + tau_ref_pts;
            t = t + 1;
        else
            t = t + 1;
        end
    end

    % === Final output ===
    predicted_spike_times_sec = predicted_spike_times * dt;
    fprintf('✅ Predicted spike count: %d\n', length(predicted_spike_times));
end
