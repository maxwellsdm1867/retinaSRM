% === Prepare Inputs ===
theta0_init = -50;    % Initial baseline threshold (mV)
A_init = 2;           % Initial jump in threshold (mV)
tau_init = 0.01;      % Initial time constant (s)
init = [theta0_init, A_init, tau_init];

% === Victor–Purpura Cost
vp_q = 4;

% === Create Model Object (replace with your data)
model = SpikeResponseModel( ...
    Vm_cleaned, ...
    Vm_all, ...
    dt, ...
    avg_spike_short, ...
    tau_ref_ms, ...
    elbow_indices, ...
    threshold_values, ...
    "Cell-01", ...
    "ON-Parasol");

% === Optimization
loss_fn = @(params) model.vp_loss_exponential(params, vp_q);
options = optimset('Display', 'iter', 'MaxFunEvals', 1000, 'MaxIter', 500);
[opt_params, vp_val] = fminsearch(loss_fn, init, options);

% === Extract Best Parameters
theta0_fit = opt_params(1);
A_fit = opt_params(2);
tau_fit = opt_params(3);
kernel_fn = @(t) A_fit * exp(-t / tau_fit);

% === Simulate with Best Parameters
[spike_vec, V_pred, theta_trace, spike_times_sec] = ...
    model.simulate(theta0_fit, kernel_fn);

% === Plot Diagnostics
zoom_xlim = [1 3];  % Adjust as needed
kernel_params = [A_fit, tau_fit];
model.diagnostics(V_pred, theta_trace, spike_vec, vp_val, zoom_xlim, '', kernel_params);
% === Set parameters from your optimized fit
theta0_fit = -51.006;
A_fit = 2.071;
tau_fit = 0.0489;
vp_q = 4;  % Victor–Purpura cost

% === Build kernel function
kernel_fn = @(t) A_fit * exp(-t / tau_fit);

% === Initialize your model (replace variables with actual workspace data)
model = SpikeResponseModel( ...
    Vm_cleaned, ...        % subthreshold voltage trace
    Vm_all, ...            % recorded full Vm
    dt, ...
    avg_spike_short, ...
    tau_ref_ms, ...
    elbow_indices, ...
    threshold_values, ...
    "Cell-01", ...
    "ON-Parasol" ...
);

% === Simulate using the provided parameters
[spike_vec, V_pred, theta_trace, spike_times_sec] = ...
    model.simulate(theta0_fit, kernel_fn);

% === Compute VP loss
true_spike_times_sec = model.elbow_indices * model.dt;
vp_dist = spkd_c(spike_times_sec, true_spike_times_sec, ...
                 length(spike_times_sec), length(true_spike_times_sec), vp_q);
fprintf('Final VP Distance = %.3f (q = %d)\n', vp_dist, vp_q);

% === Visualize
zoom_xlim = [1 3];  % adjust as needed
model.diagnostics(V_pred, theta_trace, spike_vec, vp_dist, zoom_xlim, '', [A_fit, tau_fit]);
