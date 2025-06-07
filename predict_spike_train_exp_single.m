function [Vm_model, predicted_spike_times, theta_trace] = predict_spike_train_exp_single( ...
    Vm, spike_waveform, theta0, A, tau_ms, tau_ref_ms, dt)
% PREDICT_SPIKE_TRAIN_EXP_SINGLE
% Generates spike train using dynamic threshold θ(t) = θ₀ + A·exp(-(t - t_spike)/τ)
%
% Inputs:
%   Vm             - subthreshold membrane potential (vector)
%   spike_waveform - waveform to insert on spike (e.g., avg_spike)
%   theta0         - baseline threshold (mV)
%   A              - jump after spike (mV)
%   tau_ms         - decay time constant of threshold (ms)
%   tau_ref_ms     - absolute refractory period (ms)
%   dt             - sampling interval (s)

% Preprocess
Vm_model = Vm;
spike_waveform = spike_waveform(:) - spike_waveform(1);  % force baseline to 0
spike_len = length(spike_waveform);
refract_pts = round(tau_ref_ms / 1000 / dt);
tau = tau_ms / 1000;  % convert to seconds

% Initialize
predicted_spike_times = [];
theta_trace = zeros(size(Vm_model));
t = 2;

while t <= length(Vm_model)
    % Compute dynamic threshold θ(t)
    if isempty(predicted_spike_times)
        theta_t = theta0;
    else
        t_last = predicted_spike_times(end);
        delay = (t - t_last) * dt;
        theta_t = theta0 + A * exp(-delay / tau);
    end
    theta_trace(t) = theta_t;

    % Spike condition
    if Vm_model(t) >= theta_t
        predicted_spike_times(end+1) = t;

        % Insert spike waveform
        eta_start = t;
        eta_end = min(t + spike_len - 1, length(Vm_model));
        eta_len = eta_end - eta_start + 1;

        Vm_model(eta_start:eta_end) = Vm_model(eta_start:eta_end) + spike_waveform(1:eta_len);

        % Refractory period
        t = eta_end + refract_pts;
    else
        t = t + 1;
    end
end

% Convert spike times to seconds
predicted_spike_times = predicted_spike_times * dt;
end
