function analysis_results = analyzeSpikeInitiation4Factors(Vm_all, spike_indices, dt, cell_info, save_path, varargin)
% ANALYZESPIKEINITIATION4FACTORS - Comprehensive spike initiation analysis
%
% USAGE:
%   results = analyzeSpikeInitiation4Factors(Vm_all, spike_indices, dt, cell_info, save_path)
%   results = analyzeSpikeInitiation4Factors(..., 'param', value, ...)
%
% INPUTS:
%   Vm_all       - Membrane voltage trace (mV)
%   spike_indices - Spike initiation indices (elbow points)
%   dt           - Sampling interval (s)
%   cell_info    - Struct with fields: cell_name, freq_str, date_name, protocol_name
%   save_path    - Full path for saving figure (without extension)
%
% OPTIONAL PARAMETERS:
%   'PreSpikeWindowMs'    - Short window for Vm avg and dV/dt (default: 3)
%   'SpikeCountWindowMs'  - Window for recent spike counting (default: 50)
%   'MinSpikes'          - Minimum spikes required for analysis (default: 5)
%   'CreatePlot'         - Whether to create visualization (default: true)
%   'Verbose'            - Print debug information (default: true)
%
% OUTPUTS:
%   analysis_results - Struct containing:
%     .spike_initiation_voltages - Voltage at each elbow point (mV)
%     .avg_vm_before            - Average Vm before each spike (mV)
%     .dvdt_before              - Average dV/dt before each spike (mV/s)
%     .isi_durations            - Inter-spike intervals (ms)
%     .spike_counts_before      - Recent spike counts
%     .correlations             - Correlation coefficients
%     .statistics               - Summary statistics
%     .figure_path              - Path to saved figure
%     .success                  - Analysis success flag

%% Parse input arguments
p = inputParser;
addRequired(p, 'Vm_all', @(x) isnumeric(x) && isvector(x));
addRequired(p, 'spike_indices', @(x) isnumeric(x) && isvector(x));
addRequired(p, 'dt', @(x) isnumeric(x) && isscalar(x) && x > 0);
addRequired(p, 'cell_info', @isstruct);
addRequired(p, 'save_path', @ischar);

addParameter(p, 'PreSpikeWindowMs', 5, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'SpikeCountWindowMs', 50, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'MinSpikes', 5, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'CreatePlot', true, @islogical);
addParameter(p, 'Verbose', true, @islogical);

parse(p, Vm_all, spike_indices, dt, cell_info, save_path, varargin{:});

% Extract parameters
pre_spike_window_ms = p.Results.PreSpikeWindowMs;
spike_count_window_ms = p.Results.SpikeCountWindowMs;
min_spikes = p.Results.MinSpikes;
create_plot = p.Results.CreatePlot;
verbose = p.Results.Verbose;

% Initialize results structure
analysis_results = struct();
analysis_results.success = false;

%% Validate inputs
if length(spike_indices) < min_spikes
    if verbose
        fprintf('  Insufficient spikes for analysis (need ≥%d, have %d)\n', min_spikes, length(spike_indices));
    end
    analysis_results.error_message = sprintf('Insufficient spikes (%d < %d)', length(spike_indices), min_spikes);
    return;
end

% Ensure Vm_all is column vector
if isrow(Vm_all)
    Vm_all = Vm_all';
end

% Ensure spike_indices are sorted
spike_indices = sort(spike_indices);

%% Calculate analysis parameters
pre_spike_samples = round(pre_spike_window_ms / (dt * 1000));
count_window_samples = round(spike_count_window_ms / (dt * 1000));

if verbose
    fprintf('  Starting 4-factor spike initiation analysis\n');
    fprintf('    Pre-spike window: %.1f ms (%d samples)\n', pre_spike_window_ms, pre_spike_samples);
    fprintf('    Spike count window: %.1f ms (%d samples)\n', spike_count_window_ms, count_window_samples);
end

%% Extract 4 factors for each spike
spike_initiation_voltages = [];
avg_vm_before = [];
dvdt_before = [];
isi_durations = [];
spike_counts_before = [];

% Calculate factors for each spike (skip first spike for ISI)
for spike_idx = 2:length(spike_indices)
    current_spike_sample = spike_indices(spike_idx);
    
    % Skip if too close to beginning
    if current_spike_sample <= pre_spike_samples
        continue;
    end
    
    % 1. Spike initiation voltage (elbow voltage)
    spike_voltage = Vm_all(current_spike_sample);
    
    % 2. Average Vm before spike (short window)
    pre_window_start = current_spike_sample - pre_spike_samples;
    pre_window_end = current_spike_sample - 1;
    avg_vm = mean(Vm_all(pre_window_start:pre_window_end));
    
    % 3. dV/dt before spike (mean derivative in short window)
    vm_diff = diff(Vm_all(pre_window_start:current_spike_sample));
    avg_dvdt = mean(vm_diff) / dt; % Convert to mV/s
    
    % 4. ISI duration (time since last spike)
    prev_spike_sample = spike_indices(spike_idx - 1);
    isi_duration = (current_spike_sample - prev_spike_sample) * dt * 1000; % Convert to ms
    
    % 5. Spike count in preceding window
    count_window_start = max(1, current_spike_sample - count_window_samples);
    spikes_in_window = sum(spike_indices >= count_window_start & spike_indices < current_spike_sample);
    
    % Store factors
    spike_initiation_voltages(end+1) = spike_voltage;
    avg_vm_before(end+1) = avg_vm;
    dvdt_before(end+1) = avg_dvdt;
    isi_durations(end+1) = isi_duration;
    spike_counts_before(end+1) = spikes_in_window;
end

% Check if we have enough data points
if length(spike_initiation_voltages) < 3
    if verbose
        fprintf('  Insufficient valid spikes after filtering (need ≥3, have %d)\n', length(spike_initiation_voltages));
    end
    analysis_results.error_message = sprintf('Insufficient valid spikes after filtering (%d < 3)', length(spike_initiation_voltages));
    return;
end

if verbose
    fprintf('    Analyzed %d spikes for 4-factor analysis\n', length(spike_initiation_voltages));
    fprintf('    Ranges - Init voltage: %.1f to %.1f mV\n', min(spike_initiation_voltages), max(spike_initiation_voltages));
    fprintf('    Ranges - Pre-spike Vm: %.1f to %.1f mV\n', min(avg_vm_before), max(avg_vm_before));
end

%% Calculate correlations
r_vm = corr(avg_vm_before', spike_initiation_voltages');
r_dvdt = corr(dvdt_before', spike_initiation_voltages');
r_isi = corr(log(isi_durations'), spike_initiation_voltages'); % Log correlation for ISI
r_count = corr(spike_counts_before', spike_initiation_voltages');

if verbose
    fprintf('    Correlations - Vm: %.3f, dV/dt: %.3f, ISI: %.3f, Count: %.3f\n', r_vm, r_dvdt, r_isi, r_count);
end

%% Create visualization
figure_path = '';
if create_plot
    try
        figure_path = createVisualization(Vm_all, spike_indices, dt, cell_info, save_path, ...
            spike_initiation_voltages, avg_vm_before, dvdt_before, isi_durations, spike_counts_before, ...
            r_vm, r_dvdt, r_isi, r_count, pre_spike_window_ms, spike_count_window_ms);
        
        if verbose
            fprintf('    Figure saved: %s\n', figure_path);
        end
    catch ME
        if verbose
            fprintf('    Warning: Figure creation failed - %s\n', ME.message);
        end
        figure_path = 'FIGURE_CREATION_FAILED';
    end
end

%% Store results
analysis_results.spike_initiation_voltages = spike_initiation_voltages;
analysis_results.avg_vm_before = avg_vm_before;
analysis_results.dvdt_before = dvdt_before;
analysis_results.isi_durations = isi_durations;
analysis_results.spike_counts_before = spike_counts_before;

analysis_results.correlations = struct(...
    'vm_correlation', r_vm, ...
    'dvdt_correlation', r_dvdt, ...
    'isi_correlation', r_isi, ...
    'count_correlation', r_count, ...
    'strongest_correlation', max(abs([r_vm, r_dvdt, r_isi, r_count])));

analysis_results.statistics = struct(...
    'n_spikes_analyzed', length(spike_initiation_voltages), ...
    'initiation_voltage_mean', mean(spike_initiation_voltages), ...
    'initiation_voltage_std', std(spike_initiation_voltages), ...
    'initiation_voltage_range', [min(spike_initiation_voltages), max(spike_initiation_voltages)], ...
    'initiation_voltage_cv', std(spike_initiation_voltages)/abs(mean(spike_initiation_voltages)));

analysis_results.parameters = struct(...
    'pre_spike_window_ms', pre_spike_window_ms, ...
    'spike_count_window_ms', spike_count_window_ms, ...
    'min_spikes', min_spikes);

analysis_results.figure_path = figure_path;
analysis_results.success = true;

if verbose
    fprintf('  4-factor analysis complete: strongest correlation = %.3f\n', analysis_results.correlations.strongest_correlation);
end

end

%% SUBFUNCTIONS

function figure_path = createVisualization(Vm_all, spike_indices, dt, cell_info, save_path, ...
    spike_initiation_voltages, avg_vm_before, dvdt_before, isi_durations, spike_counts_before, ...
    r_vm, r_dvdt, r_isi, r_count, pre_spike_window_ms, spike_count_window_ms)

% Create figure
fig = figure('Position', [100, 100, 1600, 800]); % Good size for 2x4 layout

% Create time axis for traces
t_ms = (0:length(Vm_all)-1) * dt * 1000; % Convert to ms

% Define colors for different ranges
colors = lines(7);

% ===============================================================
%  2x4 SUBPLOT LAYOUT
% ===============================================================

% Find examples from different ranges
[low_vm_idx, high_vm_idx] = findExampleIndices(avg_vm_before);
[slow_dvdt_idx, fast_dvdt_idx] = findExampleIndices(dvdt_before);

% TOP ROW
% Subplot [1 2]: Example traces for Vm average ranges
subplot(2, 4, [1 2]);
plotExampleTraces(Vm_all, spike_indices, dt, t_ms, low_vm_idx, high_vm_idx, colors);
xlabel('Time relative to spike (ms)');
ylabel('Vm (mV)');
title('Example Traces: Different Pre-spike Vm');
legend('Low pre-Vm', 'High pre-Vm', 'Location', 'best');
grid on;

% Subplot 3: Average Vm before vs Spike initiation voltage
subplot(2, 4, 3);
scatter(avg_vm_before, spike_initiation_voltages, 50, colors(1,:), 'filled');
xlabel('Avg Vm before (mV)');
ylabel('Spike init voltage (mV)');
title(sprintf('Pre-spike Vm (%.0fms) vs Initiation', pre_spike_window_ms));
grid on;

text(0.05, 0.95, sprintf('r = %.3f', r_vm), 'Units', 'normalized', ...
    'VerticalAlignment', 'top', 'BackgroundColor', 'white', 'FontSize', 10);

% Subplot 4: ISI vs Spike initiation voltage
subplot(2, 4, 4);
scatter(isi_durations, spike_initiation_voltages, 50, colors(3,:), 'filled');
xlabel('ISI duration (ms)');
ylabel('Spike init voltage (mV)');
title('ISI vs Initiation');
grid on;
set(gca, 'XScale', 'log'); % Log scale for ISI
text(0.05, 0.95, sprintf('r = %.3f (log)', r_isi), 'Units', 'normalized', ...
    'VerticalAlignment', 'top', 'BackgroundColor', 'white', 'FontSize', 10);

% BOTTOM ROW
% Subplot [5 6]: Example traces for dV/dt ranges
subplot(2, 4, [5 6]);
plotExampleTraces(Vm_all, spike_indices, dt, t_ms, slow_dvdt_idx, fast_dvdt_idx, colors([3 4],:));
xlabel('Time relative to spike (ms)');
ylabel('Vm (mV)');
title('Example Traces: Different dV/dt');
legend('Slow approach', 'Fast approach', 'Location', 'best');
grid on;

% Subplot 7: dV/dt before vs Spike initiation voltage
subplot(2, 4, 7);
scatter(dvdt_before, spike_initiation_voltages, 50, colors(2,:), 'filled');
xlabel('dV/dt before (mV/s)');
ylabel('Spike init voltage (mV)');
title(sprintf('dV/dt (%.0fms) vs Initiation', pre_spike_window_ms));
grid on;

text(0.05, 0.95, sprintf('r = %.3f', r_dvdt), 'Units', 'normalized', ...
    'VerticalAlignment', 'top', 'BackgroundColor', 'white', 'FontSize', 10);

% Subplot 8: Spike count vs Spike initiation voltage
subplot(2, 4, 8);
scatter(spike_counts_before, spike_initiation_voltages, 50, colors(4,:), 'filled');
xlabel(sprintf('Spike count (%.0fms)', spike_count_window_ms));
ylabel('Spike init voltage (mV)');
title('Recent Activity vs Initiation');
grid on;
text(0.05, 0.95, sprintf('r = %.3f', r_count), 'Units', 'normalized', ...
    'VerticalAlignment', 'top', 'BackgroundColor', 'white', 'FontSize', 10);

% ===============================================================
%  SUMMARY STATISTICS (Add as text annotation on the figure)
% ===============================================================

% Create summary text as figure annotation instead of subplot
% Create summary text as figure annotation (positioned below the plots)
summary_text = {
    sprintf('ANALYSIS: %s - Freq %s Hz (%d spikes)', cell_info.cell_name, cell_info.freq_str, length(spike_initiation_voltages)),
    sprintf('CORRELATIONS: Vm=%.3f | dV/dt=%.3f | ISI=%.3f | Count=%.3f', r_vm, r_dvdt, r_isi, r_count),
    sprintf('INITIATION: %.1f±%.1f mV (range: %.1f-%.1f mV, CV=%.2f)', ...
        mean(spike_initiation_voltages), std(spike_initiation_voltages), ...
        min(spike_initiation_voltages), max(spike_initiation_voltages), ...
        std(spike_initiation_voltages)/abs(mean(spike_initiation_voltages))),
    sprintf('WINDOWS: %.0fms pre-spike, %.0fms counting', pre_spike_window_ms, spike_count_window_ms)
};

% Add main title to figure
sgtitle(sprintf('%s - Frequency %s Hz - Spike Initiation 4-Factor Analysis', ...
    cell_info.cell_name, cell_info.freq_str), 'FontSize', 14, 'FontWeight', 'bold');

% Add compact summary as text annotation at bottom
annotation('textbox', [0.1, 0.02, 0.8, 0.12], 'String', summary_text, ...
    'FontSize', 10, 'FontFamily', 'monospace', 'BackgroundColor', 'white', ...
    'EdgeColor', 'black', 'FitBoxToText', 'off', 'HorizontalAlignment', 'left');

% Save figure
figure_path = [save_path '.png'];
print(fig, figure_path, '-dpng', '-r300');

% Close figure to save memory
close(fig);

end

function [low_idx, high_idx] = findExampleIndices(data_array)
% Find examples from different ranges of data
data_range = max(data_array) - min(data_array);
low_threshold = min(data_array) + 0.33 * data_range;
high_threshold = min(data_array) + 0.67 * data_range;

low_idx = find(data_array <= low_threshold, 1, 'first');
high_idx = find(data_array >= high_threshold, 1, 'first');

% Fallback if ranges are too narrow
if isempty(low_idx)
    low_idx = 1;
end
if isempty(high_idx)
    high_idx = length(data_array);
end
end

function plotExampleTraces(Vm_all, spike_indices, dt, t_ms, idx1, idx2, colors)
% Plot example voltage traces around specified spike indices

trace_window_ms = 20; % ±20ms around spike
trace_window_samples = round(trace_window_ms / (dt * 1000));

hold on;

% Plot first example
if ~isempty(idx1) && idx1 <= length(spike_indices)-1
    spike_sample = spike_indices(idx1 + 1); % +1 because we skip first spike
    if spike_sample > trace_window_samples && spike_sample + trace_window_samples <= length(Vm_all)
        trace_start = spike_sample - trace_window_samples;
        trace_end = spike_sample + trace_window_samples;
        trace_t = t_ms(trace_start:trace_end) - t_ms(spike_sample);
        trace_vm = Vm_all(trace_start:trace_end);
        
        plot(trace_t, trace_vm, 'Color', colors(1,:), 'LineWidth', 2);
        plot(0, Vm_all(spike_sample), 'o', 'Color', colors(1,:), 'MarkerSize', 8, 'MarkerFaceColor', colors(1,:));
    end
end

% Plot second example
if ~isempty(idx2) && idx2 <= length(spike_indices)-1
    spike_sample = spike_indices(idx2 + 1); % +1 because we skip first spike
    if spike_sample > trace_window_samples && spike_sample + trace_window_samples <= length(Vm_all)
        trace_start = spike_sample - trace_window_samples;
        trace_end = spike_sample + trace_window_samples;
        trace_t = t_ms(trace_start:trace_end) - t_ms(spike_sample);
        trace_vm = Vm_all(trace_start:trace_end);
        
        plot(trace_t, trace_vm, 'Color', colors(2,:), 'LineWidth', 2);
        plot(0, Vm_all(spike_sample), 'o', 'Color', colors(2,:), 'MarkerSize', 8, 'MarkerFaceColor', colors(2,:));
    end
end
end