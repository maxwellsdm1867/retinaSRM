function fourFactorGuiSeparated(gui_data)
% FOURFACTORGUISEPARATED - Two-window interface for 4-factor analysis
%
% USAGE:
%   fourFactorGuiSeparated(gui_data)
%
% Creates separate windows for:
%   - Main plot window (6 subplots)
%   - Control panel window (dropdowns, sliders)

% Validate input
if ~isstruct(gui_data) || ~isfield(gui_data, 'organized_data')
    error('Invalid gui_data structure. Run GUI data extraction first.');
end

if isempty(gui_data.cell_names)
    error('No cell data found in gui_data. Check data extraction.');
end

% Initialize shared state
shared_state = struct();
shared_state.gui_data = gui_data;
shared_state.current_cell = gui_data.cell_names{1};
shared_state.current_frequency = '';
shared_state.current_data = [];
shared_state.vm_window = 3;      % Default pre-vm window (ms)
shared_state.dvdt_window = 3;    % Default dv/dt window (ms)
shared_state.count_window = 50;  % Default count window (ms)

% Get first cell's data
cell_field = matlab.lang.makeValidName(shared_state.current_cell);
if isfield(gui_data.organized_data, cell_field)
    frequencies = gui_data.organized_data.(cell_field).frequencies;
    if ~isempty(frequencies)
        shared_state.current_frequency = frequencies{1};
        shared_state.current_data = gui_data.organized_data.(cell_field).data(shared_state.current_frequency);
        
        % Set initial windows to auto-calculated if available
        if isfield(shared_state.current_data, 'auto_window_ms')
            shared_state.vm_window = shared_state.current_data.auto_window_ms;
            shared_state.dvdt_window = shared_state.current_data.auto_window_ms;
        end
    end
end

% Create main plot figure
plot_fig = figure('Name', 'Four Factor Analysis - Plots', ...
    'Position', [100 100 1200 800], ...
    'NumberTitle', 'off');

% Create control panel figure
control_fig = figure('Name', 'Four Factor Analysis - Controls', ...
    'Position', [1320 100 280 350], ...  % Smaller height without path box
    'NumberTitle', 'off');

% Store shared state and figure handles
shared_state.plot_fig = plot_fig;
shared_state.control_fig = control_fig;

% Create the layouts
shared_state.subplots = createPlotLayout(shared_state);
createControlLayout(shared_state);

% Initial update
updateAllPlots(shared_state);

fprintf('Four Factor GUI launched successfully (separated windows)\n');
fprintf('Current: %s - %s Hz\n', shared_state.current_cell, shared_state.current_frequency);

end

function subplots = createPlotLayout(shared_state)
% Create the 2x4 subplot layout in the plot figure

figure(shared_state.plot_fig);
clf;  % Clear the figure

% Create subplot handles - EXACT layout from working code
subplots = struct();

% TOP ROW
subplots.vm_traces = subplot(2, 4, [1 2]);     % VM traces (wide)
subplots.vm_scatter = subplot(2, 4, 3);       % Pre-VM scatter
subplots.isi_scatter = subplot(2, 4, 4);      % ISI scatter

% BOTTOM ROW  
subplots.dvdt_traces = subplot(2, 4, [5 6]);  % dV/dt traces (wide)
subplots.dvdt_scatter = subplot(2, 4, 7);     % dV/dt scatter
subplots.count_scatter = subplot(2, 4, 8);    % Count scatter

% Add initial titles
title(subplots.vm_traces, 'Example Traces: Different Pre-spike Vm');
title(subplots.vm_scatter, 'Pre-spike Vm vs Initiation');
title(subplots.isi_scatter, 'ISI vs Initiation');
title(subplots.dvdt_traces, 'Example Traces: Different dV/dt');
title(subplots.dvdt_scatter, 'dV/dt vs Initiation');
title(subplots.count_scatter, 'Recent Activity vs Initiation');

end

function createControlLayout(shared_state)
% Create the control panel in the control figure

figure(shared_state.control_fig);
clf;

% Create controls and store handles in figure UserData
controls = struct();

% Cell selection
uicontrol('Style', 'text', 'String', 'Cell:', ...
    'Position', [20 350 60 25], 'HorizontalAlignment', 'left');

controls.cell_popup = uicontrol('Style', 'popupmenu', ...
    'String', shared_state.gui_data.cell_names, ...
    'Value', 1, ...
    'Position', [20 320 150 25], ...
    'Callback', @(src, evt) cellChangedCallback(src.Value));

% Frequency selection
uicontrol('Style', 'text', 'String', 'Frequency (Hz):', ...
    'Position', [20 280 100 25], 'HorizontalAlignment', 'left');

controls.freq_popup = uicontrol('Style', 'popupmenu', ...
    'String', {''}, ...
    'Position', [20 250 150 25], ...
    'Callback', @(src, evt) frequencyChangedCallback(src.Value));

% VM Window slider and manual input
uicontrol('Style', 'text', 'String', 'Pre-VM Window (ms):', ...
    'Position', [20 180 150 20], 'HorizontalAlignment', 'left');

controls.vm_slider = uicontrol('Style', 'slider', ...
    'Min', 0.5, 'Max', 10, 'Value', shared_state.vm_window, ...
    'Position', [20 155 120 20], ...
    'Callback', @(src, evt) vmWindowChangedCallback(src.Value));

controls.vm_edit = uicontrol('Style', 'edit', ...
    'String', sprintf('%.1f', shared_state.vm_window), ...
    'Position', [145 155 35 20], ...
    'Callback', @(src, evt) vmEditCallback(str2double(src.String)));

controls.vm_label = uicontrol('Style', 'text', ...
    'String', sprintf('%.1f ms', shared_state.vm_window), ...
    'Position', [185 155 60 20], 'HorizontalAlignment', 'left');

% dV/dt Window slider and manual input
uicontrol('Style', 'text', 'String', 'dV/dt Window (ms):', ...
    'Position', [20 125 150 20], 'HorizontalAlignment', 'left');

controls.dvdt_slider = uicontrol('Style', 'slider', ...
    'Min', 0.5, 'Max', 10, 'Value', shared_state.dvdt_window, ...
    'Position', [20 100 120 20], ...
    'Callback', @(src, evt) dvdtWindowChangedCallback(src.Value));

controls.dvdt_edit = uicontrol('Style', 'edit', ...
    'String', sprintf('%.1f', shared_state.dvdt_window), ...
    'Position', [145 100 35 20], ...
    'Callback', @(src, evt) dvdtEditCallback(str2double(src.String)));

controls.dvdt_label = uicontrol('Style', 'text', ...
    'String', sprintf('%.1f ms', shared_state.dvdt_window), ...
    'Position', [185 100 60 20], 'HorizontalAlignment', 'left');

% Count Window slider and manual input
uicontrol('Style', 'text', 'String', 'Count Window (ms):', ...
    'Position', [20 70 150 20], 'HorizontalAlignment', 'left');

controls.count_slider = uicontrol('Style', 'slider', ...
    'Min', 10, 'Max', 200, 'Value', shared_state.count_window, ...
    'Position', [20 45 120 20], ...
    'Callback', @(src, evt) countWindowChangedCallback(src.Value));

controls.count_edit = uicontrol('Style', 'edit', ...
    'String', sprintf('%.0f', shared_state.count_window), ...
    'Position', [145 45 35 20], ...
    'Callback', @(src, evt) countEditCallback(str2double(src.String)));

controls.count_label = uicontrol('Style', 'text', ...
    'String', sprintf('%.0f ms', shared_state.count_window), ...
    'Position', [185 45 60 20], 'HorizontalAlignment', 'left');

% File path input



% Auto window info - just display the pre-calculated value
controls.auto_label = uicontrol('Style', 'text', ...
    'String', 'Auto Window: loading...', ...
    'Position', [20 25 250 15], 'HorizontalAlignment', 'left', ...
    'ForegroundColor', [0 0.6 0], 'FontSize', 8);

% Refresh Examples button
controls.refresh_btn = uicontrol('Style', 'pushbutton', ...
    'String', 'Refresh Example Traces', ...
    'Position', [20 5 120 15], ...
    'Callback', @(src, evt) refreshExamplesCallback());

% Save JPG button
controls.save_btn = uicontrol('Style', 'pushbutton', ...
    'String', 'Save as JPG', ...
    'Position', [150 5 100 15], ...
    'Callback', @(src, evt) saveJpgCallback());

% Store everything in the control figure's UserData
control_data = struct();
control_data.controls = controls;
control_data.shared_state = shared_state;
control_data.fixed_examples = []; % Store fixed examples for current cell/frequency
set(shared_state.control_fig, 'UserData', control_data);

% Initialize frequency dropdown
updateFrequencyDropdown();

end

function updateAllPlots(shared_state)
% Update all plots with current data and window settings

if isempty(shared_state.current_data)
    return;
end

current_data = shared_state.current_data;
vm_window = shared_state.vm_window;
dvdt_window = shared_state.dvdt_window;
count_window = shared_state.count_window;

% Extract basic data
Vm_all = current_data.Vm_all;
spike_indices = current_data.spike_indices;
dt = current_data.dt;

if length(spike_indices) < 2
    return; % Need at least 2 spikes for analysis
end

% Calculate factors with current window settings
[factors, examples] = calculateFactors(Vm_all, spike_indices, dt, vm_window, dvdt_window, count_window);

% Make sure we're plotting in the plot figure
figure(shared_state.plot_fig);

% Add supertitle with comprehensive cell information including metadata
cell_info = sprintf('Cell: %s | Frequency: %s Hz | Spikes: %d | Duration: %.1fs', ...
    shared_state.current_cell, shared_state.current_frequency, ...
    length(spike_indices), length(Vm_all) * dt);

% Try to extract additional metadata from current_data
if isfield(current_data, 'protocol') && isfield(current_data, 'date')
    % Keep original protocol name without changing special characters
    cell_info = sprintf('%s | Protocol: %s | Date: %s', cell_info, current_data.protocol, current_data.date);
end

sgtitle(cell_info, 'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'none');

% Update traces
updateTraces(shared_state, Vm_all, spike_indices, dt, examples, vm_window, dvdt_window);

% Update scatter plots
updateScatterPlots(shared_state, factors, examples);

% Note: Auto window info is updated in the callback functions where shared_state is accessible

end

function [factors, examples] = calculateFactors(Vm_all, spike_indices, dt, vm_window, dvdt_window, count_window, randomize_examples)
% Calculate all 4 factors with specified windows
% randomize_examples: optional flag to randomize example selection

if nargin < 7
    randomize_examples = false;
end

% Convert windows to samples
vm_samples = round(vm_window / (dt * 1000));
dvdt_samples = round(dvdt_window / (dt * 1000));
count_samples = round(count_window / (dt * 1000));

% Initialize arrays
initiation_voltages = [];
avg_vm_before = [];
dvdt_before = [];
isi_durations = [];
counts_before = [];

% Process each spike (skip first for ISI)
for i = 2:length(spike_indices)
    current_idx = spike_indices(i);
    
    % Skip if too close to beginning
    if current_idx <= max(vm_samples, dvdt_samples)
        continue;
    end
    
    % 1. Initiation voltage
    initiation_voltages(end+1) = Vm_all(current_idx);
    
    % 2. Average Vm before (vm_window)
    vm_start = current_idx - vm_samples;
    vm_end = current_idx - 1;
    avg_vm_before(end+1) = mean(Vm_all(vm_start:vm_end));
    
    % 3. dV/dt before (dvdt_window)
    dvdt_start = current_idx - dvdt_samples;
    dvdt_end = current_idx - 1;
    vm_diff = diff(Vm_all(dvdt_start:current_idx));
    dvdt_before(end+1) = mean(vm_diff) / dt;
    
    % 4. ISI duration
    prev_idx = spike_indices(i-1);
    isi_durations(end+1) = (current_idx - prev_idx) * dt * 1000; % ms
    
    % 5. Count in window
    count_start = max(1, current_idx - count_samples);
    counts_before(end+1) = sum(spike_indices >= count_start & spike_indices < current_idx);
end

% Check if we have enough data
if length(avg_vm_before) < 2
    % Return empty factors if insufficient data
    factors = struct();
    factors.initiation_voltages = [];
    factors.avg_vm_before = [];
    factors.dvdt_before = [];
    factors.isi_durations = [];
    factors.counts_before = [];
    
    examples = struct();
    examples.low_vm_idx = [];
    examples.high_vm_idx = [];
    examples.slow_dvdt_idx = [];
    examples.fast_dvdt_idx = [];
    return;
end

% Find examples for traces with better randomization
vm_range = max(avg_vm_before) - min(avg_vm_before);
dvdt_range = max(dvdt_before) - min(dvdt_before);

% Handle edge case where all values are the same
if vm_range == 0
    low_vm_threshold = min(avg_vm_before);
    high_vm_threshold = min(avg_vm_before);
else
    if randomize_examples
        % Use more aggressive randomization to ensure different examples each time
        rand_offset1 = 0.05 + 0.25 * rand();  % Random between 5-30%
        rand_offset2 = 0.70 + 0.25 * rand();  % Random between 70-95%
        low_vm_threshold = min(avg_vm_before) + rand_offset1 * vm_range;
        high_vm_threshold = min(avg_vm_before) + rand_offset2 * vm_range;
    else
        low_vm_threshold = min(avg_vm_before) + 0.33 * vm_range;
        high_vm_threshold = min(avg_vm_before) + 0.67 * vm_range;
    end
end

if dvdt_range == 0
    slow_dvdt_threshold = min(dvdt_before);
    fast_dvdt_threshold = min(dvdt_before);
else
    if randomize_examples
        % Use more aggressive randomization to ensure different examples each time
        rand_offset1 = 0.05 + 0.25 * rand();  % Random between 5-30%
        rand_offset2 = 0.70 + 0.25 * rand();  % Random between 70-95%
        slow_dvdt_threshold = min(dvdt_before) + rand_offset1 * dvdt_range;
        fast_dvdt_threshold = min(dvdt_before) + rand_offset2 * dvdt_range;
    else
        slow_dvdt_threshold = min(dvdt_before) + 0.33 * dvdt_range;
        fast_dvdt_threshold = min(dvdt_before) + 0.67 * dvdt_range;
    end
end

examples = struct();

% Find examples with multiple candidates to ensure variety
if randomize_examples
    % Find all candidates that meet criteria, then randomly select
    low_vm_candidates = find(avg_vm_before <= low_vm_threshold);
    high_vm_candidates = find(avg_vm_before >= high_vm_threshold);
    slow_dvdt_candidates = find(dvdt_before <= slow_dvdt_threshold);
    fast_dvdt_candidates = find(dvdt_before >= fast_dvdt_threshold);
    
    % Randomly select from candidates
    if ~isempty(low_vm_candidates)
        examples.low_vm_idx = low_vm_candidates(randi(length(low_vm_candidates)));
    else
        examples.low_vm_idx = [];
    end
    
    if ~isempty(high_vm_candidates)
        examples.high_vm_idx = high_vm_candidates(randi(length(high_vm_candidates)));
    else
        examples.high_vm_idx = [];
    end
    
    if ~isempty(slow_dvdt_candidates)
        examples.slow_dvdt_idx = slow_dvdt_candidates(randi(length(slow_dvdt_candidates)));
    else
        examples.slow_dvdt_idx = [];
    end
    
    if ~isempty(fast_dvdt_candidates)
        examples.fast_dvdt_idx = fast_dvdt_candidates(randi(length(fast_dvdt_candidates)));
    else
        examples.fast_dvdt_idx = [];
    end
else
    % Original method for non-randomized
    examples.low_vm_idx = find(avg_vm_before <= low_vm_threshold, 1, 'first');
    examples.high_vm_idx = find(avg_vm_before >= high_vm_threshold, 1, 'first');
    examples.slow_dvdt_idx = find(dvdt_before <= slow_dvdt_threshold, 1, 'first');
    examples.fast_dvdt_idx = find(dvdt_before >= fast_dvdt_threshold, 1, 'first');
end

% Store factors
factors = struct();
factors.initiation_voltages = initiation_voltages;
factors.avg_vm_before = avg_vm_before;
factors.dvdt_before = dvdt_before;
factors.isi_durations = isi_durations;
factors.counts_before = counts_before;

end

function updateTraces(shared_state, Vm_all, spike_indices, dt, examples, vm_window, dvdt_window)
% Update example trace plots - EXACT match to working code

t_ms = (0:length(Vm_all)-1) * dt * 1000;
% Changed trace window: -10ms to +3ms around spike initiation
pre_window_ms = 10;   % 10ms before spike
post_window_ms = 3;   % 3ms after spike
pre_window_samples = round(pre_window_ms / (dt * 1000));
post_window_samples = round(post_window_ms / (dt * 1000));
colors = lines(4);

% Calculate reasonable y-axis limits from the data
if length(spike_indices) > 10
    % Sample some spike traces to get realistic Vm range
    sample_indices = spike_indices(1:min(10, length(spike_indices)));
    sample_vms = [];
    for i = 1:length(sample_indices)
        idx = sample_indices(i);
        if idx > pre_window_samples && idx + post_window_samples <= length(Vm_all)
            trace_start = idx - pre_window_samples;
            trace_end = idx + post_window_samples;
            sample_vms = [sample_vms; Vm_all(trace_start:trace_end)];
        end
    end
    if ~isempty(sample_vms)
        y_min = min(sample_vms) - 5; % Add 5mV margin
        y_max = max(sample_vms) + 5;
        y_limits = [y_min y_max];
    else
        y_limits = [-80 -30]; % Default range
    end
else
    y_limits = [-80 -30]; % Default range
end

% TOP ROW - Subplot [1 2]: Example traces for Vm average ranges
subplot(shared_state.subplots.vm_traces);
cla;
hold on;

% Draw reference line for pre-spike Vm window
vm_window_start = -vm_window;
vm_window_end = 0;
% Draw semi-transparent rectangle for Vm window
patch([vm_window_start vm_window_end vm_window_end vm_window_start], ...
      [y_limits(1) y_limits(1) y_limits(2) y_limits(2)], ...
      [0.8 0.8 0.8], 'FaceAlpha', 0.3, 'EdgeColor', 'none');

% Plot traces using same logic as your working code
low_vm_idx = examples.low_vm_idx;
high_vm_idx = examples.high_vm_idx;

if ~isempty(low_vm_idx) && low_vm_idx <= length(spike_indices)-1
    idx = spike_indices(low_vm_idx + 1);
    if idx > pre_window_samples && idx + post_window_samples <= length(Vm_all)
        trace_start = idx - pre_window_samples;
        trace_end = idx + post_window_samples;
        trace_t = t_ms(trace_start:trace_end) - t_ms(idx);
        trace_vm = Vm_all(trace_start:trace_end);
        plot(trace_t, trace_vm, 'Color', colors(1,:), 'LineWidth', 2);
        % Mark the elbow point (spike initiation)
        plot(0, Vm_all(idx), 'o', 'Color', colors(1,:), 'MarkerSize', 8, 'MarkerFaceColor', colors(1,:), 'LineWidth', 2);
    end
end

if ~isempty(high_vm_idx) && high_vm_idx <= length(spike_indices)-1
    idx = spike_indices(high_vm_idx + 1);
    if idx > pre_window_samples && idx + post_window_samples <= length(Vm_all)
        trace_start = idx - pre_window_samples;
        trace_end = idx + post_window_samples;
        trace_t = t_ms(trace_start:trace_end) - t_ms(idx);
        trace_vm = Vm_all(trace_start:trace_end);
        plot(trace_t, trace_vm, 'Color', colors(2,:), 'LineWidth', 2);
        % Mark the elbow point (spike initiation)
        plot(0, Vm_all(idx), 'o', 'Color', colors(2,:), 'MarkerSize', 8, 'MarkerFaceColor', colors(2,:), 'LineWidth', 2);
    end
end

% Draw vertical line at spike initiation
plot([0 0], y_limits, 'k--', 'LineWidth', 1);

xlabel('Time relative to spike initiation (ms)');
ylabel('Vm (mV)');
title(sprintf('Example Traces: Different Pre-spike Vm (%.1fms window)', vm_window));

% Create legend entries only for traces that actually exist
legend_entries = {};
if ~isempty(low_vm_idx) && low_vm_idx <= length(spike_indices)-1
    legend_entries{end+1} = 'Pre-Vm window';
    legend_entries{end+1} = 'Low pre-Vm';
    if ~isempty(high_vm_idx) && high_vm_idx <= length(spike_indices)-1
        legend_entries{end+1} = 'High pre-Vm';
    end
else
    legend_entries{end+1} = 'Pre-Vm window';
    if ~isempty(high_vm_idx) && high_vm_idx <= length(spike_indices)-1
        legend_entries{end+1} = 'High pre-Vm';
    end
end

if ~isempty(legend_entries)
    legend(legend_entries, 'Location', 'best');
end
grid on;
xlim([-10 3]);  % Set fixed x-axis limits
ylim(y_limits); % Set fixed y-limits
hold off;

% BOTTOM ROW - Subplot [5 6]: Example traces for dV/dt ranges
subplot(shared_state.subplots.dvdt_traces);
cla;
hold on;

% Draw reference line for dV/dt window
dvdt_window_start = -dvdt_window;
dvdt_window_end = 0;
% Draw semi-transparent rectangle for dV/dt window
patch([dvdt_window_start dvdt_window_end dvdt_window_end dvdt_window_start], ...
      [y_limits(1) y_limits(1) y_limits(2) y_limits(2)], ...
      [0.9 0.9 0.6], 'FaceAlpha', 0.3, 'EdgeColor', 'none');

slow_dvdt_idx = examples.slow_dvdt_idx;
fast_dvdt_idx = examples.fast_dvdt_idx;

if ~isempty(slow_dvdt_idx) && slow_dvdt_idx <= length(spike_indices)-1
    idx = spike_indices(slow_dvdt_idx + 1);
    if idx > pre_window_samples && idx + post_window_samples <= length(Vm_all)
        trace_start = idx - pre_window_samples;
        trace_end = idx + post_window_samples;
        trace_t = t_ms(trace_start:trace_end) - t_ms(idx);
        trace_vm = Vm_all(trace_start:trace_end);
        plot(trace_t, trace_vm, 'Color', colors(3,:), 'LineWidth', 2);
        % Mark the elbow point (spike initiation)
        plot(0, Vm_all(idx), 'o', 'Color', colors(3,:), 'MarkerSize', 8, 'MarkerFaceColor', colors(3,:), 'LineWidth', 2);
    end
end

if ~isempty(fast_dvdt_idx) && fast_dvdt_idx <= length(spike_indices)-1
    idx = spike_indices(fast_dvdt_idx + 1);
    if idx > pre_window_samples && idx + post_window_samples <= length(Vm_all)
        trace_start = idx - pre_window_samples;
        trace_end = idx + post_window_samples;
        trace_t = t_ms(trace_start:trace_end) - t_ms(idx);
        trace_vm = Vm_all(trace_start:trace_end);
        plot(trace_t, trace_vm, 'Color', colors(4,:), 'LineWidth', 2);
        % Mark the elbow point (spike initiation)
        plot(0, Vm_all(idx), 'o', 'Color', colors(4,:), 'MarkerSize', 8, 'MarkerFaceColor', colors(4,:), 'LineWidth', 2);
    end
end

% Draw vertical line at spike initiation
plot([0 0], y_limits, 'k--', 'LineWidth', 1);

xlabel('Time relative to spike initiation (ms)');
ylabel('Vm (mV)');
title(sprintf('Example Traces: Different dV/dt (%.1fms window)', dvdt_window));

% Create legend entries only for traces that actually exist
legend_entries = {};
if ~isempty(slow_dvdt_idx) && slow_dvdt_idx <= length(spike_indices)-1
    legend_entries{end+1} = 'dV/dt window';
    legend_entries{end+1} = 'Slow approach';
    if ~isempty(fast_dvdt_idx) && fast_dvdt_idx <= length(spike_indices)-1
        legend_entries{end+1} = 'Fast approach';
    end
else
    legend_entries{end+1} = 'dV/dt window';
    if ~isempty(fast_dvdt_idx) && fast_dvdt_idx <= length(spike_indices)-1
        legend_entries{end+1} = 'Fast approach';
    end
end

if ~isempty(legend_entries)
    legend(legend_entries, 'Location', 'best');
end
grid on;
xlim([-10 3]);  % Set fixed x-axis limits
ylim(y_limits); % Set fixed y-limits
hold off;

end

function updateScatterPlots(shared_state, factors, examples)
% Update all scatter plots with correlations - EXACT match to working code
% Now includes markers for example traces

% Check if we have data
if isempty(factors.initiation_voltages)
    return;
end

colors = lines(4);
spike_initiation_voltages = factors.initiation_voltages;
avg_vm_before = factors.avg_vm_before;
dvdt_before = factors.dvdt_before;
isi_durations = factors.isi_durations;
spike_counts_before = factors.counts_before;

pre_spike_window_ms = shared_state.vm_window;
spike_count_window_ms = shared_state.count_window;

% Subplot 3: Average Vm before vs Spike initiation voltage
subplot(shared_state.subplots.vm_scatter);
cla;
scatter(avg_vm_before, spike_initiation_voltages, 50, colors(1,:), 'filled', 'MarkerFaceAlpha', 0.6);
hold on;

% Mark the example traces if they exist
if nargin > 2 && ~isempty(examples)
    if ~isempty(examples.low_vm_idx) && examples.low_vm_idx <= length(avg_vm_before)
        scatter(avg_vm_before(examples.low_vm_idx), spike_initiation_voltages(examples.low_vm_idx), ...
            120, colors(1,:), 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 3);
        text(avg_vm_before(examples.low_vm_idx), spike_initiation_voltages(examples.low_vm_idx), ...
            ' Low', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');
    end
    if ~isempty(examples.high_vm_idx) && examples.high_vm_idx <= length(avg_vm_before)
        scatter(avg_vm_before(examples.high_vm_idx), spike_initiation_voltages(examples.high_vm_idx), ...
            120, colors(2,:), 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 3);
        text(avg_vm_before(examples.high_vm_idx), spike_initiation_voltages(examples.high_vm_idx), ...
            ' High', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');
    end
end

xlabel('Avg Vm before (mV)');
ylabel('Spike init voltage (mV)');
title(sprintf('Pre-spike Vm (%.1fms) vs Initiation', pre_spike_window_ms));
grid on;
hold off;

if length(avg_vm_before) > 2
    r_vm = corr(avg_vm_before', spike_initiation_voltages');
    text(0.05, 0.95, sprintf('r = %.3f', r_vm), 'Units', 'normalized', ...
        'VerticalAlignment', 'top', 'BackgroundColor', 'white', 'FontSize', 10);
end

% Subplot 4: ISI vs Spike initiation voltage
subplot(shared_state.subplots.isi_scatter);
cla;
scatter(isi_durations, spike_initiation_voltages, 50, colors(3,:), 'filled', 'MarkerFaceAlpha', 0.6);
xlabel('ISI duration (ms)');
ylabel('Spike init voltage (mV)');
title('ISI vs Initiation');
grid on;
set(gca, 'XScale', 'log'); % Log scale for ISI

if length(isi_durations) > 2
    r_isi = corr(log(isi_durations'), spike_initiation_voltages');
    text(0.05, 0.95, sprintf('r = %.3f (log)', r_isi), 'Units', 'normalized', ...
        'VerticalAlignment', 'top', 'BackgroundColor', 'white', 'FontSize', 10);
end

% Subplot 7: dV/dt before vs Spike initiation voltage
subplot(shared_state.subplots.dvdt_scatter);
cla;
scatter(dvdt_before, spike_initiation_voltages, 50, colors(2,:), 'filled', 'MarkerFaceAlpha', 0.6);
hold on;

% Mark the example traces if they exist
if nargin > 2 && ~isempty(examples)
    if ~isempty(examples.slow_dvdt_idx) && examples.slow_dvdt_idx <= length(dvdt_before)
        scatter(dvdt_before(examples.slow_dvdt_idx), spike_initiation_voltages(examples.slow_dvdt_idx), ...
            120, colors(3,:), 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 3);
        text(dvdt_before(examples.slow_dvdt_idx), spike_initiation_voltages(examples.slow_dvdt_idx), ...
            ' Slow', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');
    end
    if ~isempty(examples.fast_dvdt_idx) && examples.fast_dvdt_idx <= length(dvdt_before)
        scatter(dvdt_before(examples.fast_dvdt_idx), spike_initiation_voltages(examples.fast_dvdt_idx), ...
            120, colors(4,:), 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 3);
        text(dvdt_before(examples.fast_dvdt_idx), spike_initiation_voltages(examples.fast_dvdt_idx), ...
            ' Fast', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');
    end
end

xlabel('dV/dt before (mV/s)');
ylabel('Spike init voltage (mV)');
title(sprintf('dV/dt (%.1fms) vs Initiation', pre_spike_window_ms));
grid on;
hold off;

if length(dvdt_before) > 2
    r_dvdt = corr(dvdt_before', spike_initiation_voltages');
    text(0.05, 0.95, sprintf('r = %.3f', r_dvdt), 'Units', 'normalized', ...
        'VerticalAlignment', 'top', 'BackgroundColor', 'white', 'FontSize', 10);
end

% Subplot 8: Spike count vs Spike initiation voltage
subplot(shared_state.subplots.count_scatter);
cla;
scatter(spike_counts_before, spike_initiation_voltages, 50, colors(4,:), 'filled', 'MarkerFaceAlpha', 0.6);
xlabel(sprintf('Spike count (%.0fms)', spike_count_window_ms));
ylabel('Spike init voltage (mV)');
title(sprintf('Recent Activity (%.1fms) vs Initiation', spike_count_window_ms));
grid on;

if length(spike_counts_before) > 2
    r_count = corr(spike_counts_before', spike_initiation_voltages');
    text(0.05, 0.95, sprintf('r = %.3f', r_count), 'Units', 'normalized', ...
        'VerticalAlignment', 'top', 'BackgroundColor', 'white', 'FontSize', 10);
end

end

%% CALLBACK FUNCTIONS USING USERDATA

function vmEditCallback(new_value)
% Handle manual VM window input
if isnan(new_value) || new_value < 0.5 || new_value > 10
    fprintf('Invalid VM window value. Must be between 0.5 and 10 ms\n');
    return;
end
vmWindowChangedCallback(new_value);
end

function dvdtEditCallback(new_value)
% Handle manual dV/dt window input
if isnan(new_value) || new_value < 0.5 || new_value > 10
    fprintf('Invalid dV/dt window value. Must be between 0.5 and 10 ms\n');
    return;
end
dvdtWindowChangedCallback(new_value);
end

function countEditCallback(new_value)
% Handle manual count window input
if isnan(new_value) || new_value < 10 || new_value > 200
    fprintf('Invalid count window value. Must be between 10 and 200 ms\n');
    return;
end
countWindowChangedCallback(new_value);
end

function vmWindowChangedCallback(new_value)
% Get data from control figure
control_figs = findall(0, 'Name', 'Four Factor Analysis - Controls');
if isempty(control_figs), return; end
control_data = get(control_figs(1), 'UserData');

% Update state and controls
control_data.shared_state.vm_window = new_value;
set(control_data.controls.vm_slider, 'Value', new_value);
set(control_data.controls.vm_edit, 'String', sprintf('%.1f', new_value));
set(control_data.controls.vm_label, 'String', sprintf('%.1f ms', new_value));

% Update UserData with new state
set(control_figs(1), 'UserData', control_data);

% Update plots using fixed examples
updateAllPlotsFixedExamples(control_data.shared_state, control_data.fixed_examples);
end

function dvdtWindowChangedCallback(new_value)
% Get data from control figure
control_figs = findall(0, 'Name', 'Four Factor Analysis - Controls');
if isempty(control_figs), return; end
control_data = get(control_figs(1), 'UserData');

% Update state and controls
control_data.shared_state.dvdt_window = new_value;
set(control_data.controls.dvdt_slider, 'Value', new_value);
set(control_data.controls.dvdt_edit, 'String', sprintf('%.1f', new_value));
set(control_data.controls.dvdt_label, 'String', sprintf('%.1f ms', new_value));

% Update UserData with new state
set(control_figs(1), 'UserData', control_data);

% Update plots using fixed examples
updateAllPlotsFixedExamples(control_data.shared_state, control_data.fixed_examples);
end

function countWindowChangedCallback(new_value)
% Get data from control figure
control_figs = findall(0, 'Name', 'Four Factor Analysis - Controls');
if isempty(control_figs), return; end
control_data = get(control_figs(1), 'UserData');

% Update state and controls
control_data.shared_state.count_window = new_value;
set(control_data.controls.count_slider, 'Value', new_value);
set(control_data.controls.count_edit, 'String', sprintf('%.0f', new_value));
set(control_data.controls.count_label, 'String', sprintf('%.0f ms', new_value));

% Update UserData with new state
set(control_figs(1), 'UserData', control_data);

% Update plots using fixed examples
updateAllPlotsFixedExamples(control_data.shared_state, control_data.fixed_examples);
end

function cellChangedCallback(new_index)
% Get data from control figure
control_figs = findall(0, 'Name', 'Four Factor Analysis - Controls');
if isempty(control_figs), return; end
control_data = get(control_figs(1), 'UserData');

% Update state
control_data.shared_state.current_cell = control_data.shared_state.gui_data.cell_names{new_index};

% Update UserData with new state
set(control_figs(1), 'UserData', control_data);

% Update frequency dropdown and plots
updateFrequencyDropdown();
fprintf('Switched to cell: %s\n', control_data.shared_state.current_cell);
end

function frequencyChangedCallback(new_index)
% Get data from control figure
control_figs = findall(0, 'Name', 'Four Factor Analysis - Controls');
if isempty(control_figs), return; end
control_data = get(control_figs(1), 'UserData');

% Update state
cell_field = matlab.lang.makeValidName(control_data.shared_state.current_cell);
frequencies = control_data.shared_state.gui_data.organized_data.(cell_field).frequencies;
control_data.shared_state.current_frequency = frequencies{new_index};
control_data.shared_state.current_data = control_data.shared_state.gui_data.organized_data.(cell_field).data(control_data.shared_state.current_frequency);

% Generate new fixed examples for the new frequency
if ~isempty(control_data.shared_state.current_data)
    current_data = control_data.shared_state.current_data;
    Vm_all = current_data.Vm_all;
    spike_indices = current_data.spike_indices;
    dt = current_data.dt;
    vm_window = control_data.shared_state.vm_window;
    dvdt_window = control_data.shared_state.dvdt_window;
    count_window = control_data.shared_state.count_window;
    
    % Calculate default examples for new frequency
    [~, new_examples] = calculateFactors(Vm_all, spike_indices, dt, vm_window, dvdt_window, count_window, false);
    control_data.fixed_examples = new_examples;
end

% Update UserData with new state
set(control_figs(1), 'UserData', control_data);

% Update auto window display
if isfield(control_data.shared_state.current_data, 'auto_window_ms')
    auto_window = control_data.shared_state.current_data.auto_window_ms;
    set(control_data.controls.auto_label, 'String', sprintf('Injected Current Autocorr: %.1f ms', auto_window));
else
    set(control_data.controls.auto_label, 'String', 'Injected Current Autocorr: not available');
end

% Update plots with fixed examples
updateAllPlotsFixedExamples(control_data.shared_state, control_data.fixed_examples);
fprintf('Switched to frequency: %s Hz\n', control_data.shared_state.current_frequency);
end

function saveJpgCallback()
% Save the current plot figure as JPG
plot_figs = findall(0, 'Name', 'Four Factor Analysis - Plots');
if isempty(plot_figs)
    fprintf('No plot figure found to save\n');
    return;
end

% Get current cell and frequency info for filename
control_figs = findall(0, 'Name', 'Four Factor Analysis - Controls');
if ~isempty(control_figs)
    control_data = get(control_figs(1), 'UserData');
    cell_name = control_data.shared_state.current_cell;
    frequency = control_data.shared_state.current_frequency;
    vm_window = control_data.shared_state.vm_window;
    dvdt_window = control_data.shared_state.dvdt_window;
    count_window = control_data.shared_state.count_window;
    
    % Use save path from gui_data
    if isfield(control_data.shared_state.gui_data, 'save_path')
        save_path = control_data.shared_state.gui_data.save_path;
    else
        save_path = pwd;  % Fallback to current directory
        fprintf('Warning: No save_path found in gui_data, using current directory\n');
    end
    
    if ~exist(save_path, 'dir')
        fprintf('Warning: Directory %s does not exist, using current directory\n', save_path);
        save_path = pwd;
    end
    
    % Create descriptive filename - replace backslashes with dashes
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    cell_name_safe = strrep(matlab.lang.makeValidName(cell_name), '\', '-');
    frequency_safe = strrep(frequency, '\', '-');
    filename = sprintf('FourFactor_%s_%sHz_VM%.1f_dVdt%.1f_Count%.0f_%s.jpg', ...
        cell_name_safe, frequency_safe, vm_window, dvdt_window, count_window, timestamp);
    
    % Full path
    full_path = fullfile(save_path, filename);
else
    % Fallback filename
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    filename = sprintf('FourFactorAnalysis_%s.jpg', timestamp);
    full_path = filename;
end

% Save the plot figure
figure(plot_figs(1));
print(plot_figs(1), full_path, '-djpeg', '-r300');  % High resolution JPG

fprintf('Saved plot as: %s\n', full_path);
end

function updateAllPlotsFixedExamples(shared_state, fixed_examples)
% Update plots using fixed examples (don't recalculate examples)

if isempty(shared_state.current_data)
    return;
end

current_data = shared_state.current_data;
vm_window = shared_state.vm_window;
dvdt_window = shared_state.dvdt_window;
count_window = shared_state.count_window;

% Extract basic data
Vm_all = current_data.Vm_all;
spike_indices = current_data.spike_indices;
dt = current_data.dt;

if length(spike_indices) < 2
    return; % Need at least 2 spikes for analysis
end

% Calculate factors with current window settings (no randomization)
[factors, ~] = calculateFactors(Vm_all, spike_indices, dt, vm_window, dvdt_window, count_window, false);

% Use fixed examples if available, otherwise calculate new ones
if isempty(fixed_examples)
    [~, examples] = calculateFactors(Vm_all, spike_indices, dt, vm_window, dvdt_window, count_window, false);
else
    examples = fixed_examples;
end

% Make sure we're plotting in the plot figure
figure(shared_state.plot_fig);

% Add supertitle with comprehensive cell information including metadata
cell_info = sprintf('Cell: %s | Frequency: %s Hz | Spikes: %d | Duration: %.1fs', ...
    shared_state.current_cell, shared_state.current_frequency, ...
    length(spike_indices), length(Vm_all) * dt);

% Try to extract additional metadata from current_data
if isfield(current_data, 'protocol') && isfield(current_data, 'date')
    cell_info = sprintf('%s | Protocol: %s | Date: %s', cell_info, current_data.protocol, current_data.date);
end

sgtitle(cell_info, 'FontSize', 14, 'FontWeight', 'bold');

% Update traces with fixed examples
updateTraces(shared_state, Vm_all, spike_indices, dt, examples, vm_window, dvdt_window);

% Update scatter plots
updateScatterPlots(shared_state, factors, examples);

end

function refreshExamplesCallback()
% Get data from control figure and refresh example traces
control_figs = findall(0, 'Name', 'Four Factor Analysis - Controls');
if isempty(control_figs)
    fprintf('Error: No control figure found\n');
    return; 
end
control_data = get(control_figs(1), 'UserData');

% Calculate new random examples and store them as fixed
if ~isempty(control_data.shared_state.current_data)
    current_data = control_data.shared_state.current_data;
    Vm_all = current_data.Vm_all;
    spike_indices = current_data.spike_indices;
    dt = current_data.dt;
    vm_window = control_data.shared_state.vm_window;
    dvdt_window = control_data.shared_state.dvdt_window;
    count_window = control_data.shared_state.count_window;
    
    fprintf('DEBUG: Generating new examples...\n');
    fprintf('  Data: %d spikes, VM window: %.1fms\n', length(spike_indices), vm_window);
    
    % Calculate new randomized examples
    [~, new_examples] = calculateFactors(Vm_all, spike_indices, dt, vm_window, dvdt_window, count_window, true);
    
    fprintf('  New examples: low_vm=%d, high_vm=%d, slow_dvdt=%d, fast_dvdt=%d\n', ...
        new_examples.low_vm_idx, new_examples.high_vm_idx, new_examples.slow_dvdt_idx, new_examples.fast_dvdt_idx);
    
    % Store as fixed examples
    control_data.fixed_examples = new_examples;
    set(control_figs(1), 'UserData', control_data);
    
    % Update plots with new fixed examples
    updateAllPlotsFixedExamples(control_data.shared_state, control_data.fixed_examples);
    fprintf('Refreshed example traces\n');
else
    fprintf('Error: No current data available\n');
end
end

function updateFrequencyDropdown()
% Get data from control figure
control_figs = findall(0, 'Name', 'Four Factor Analysis - Controls');
if isempty(control_figs), return; end
control_data = get(control_figs(1), 'UserData');

% Update frequency dropdown for current cell
cell_field = matlab.lang.makeValidName(control_data.shared_state.current_cell);
if isfield(control_data.shared_state.gui_data.organized_data, cell_field)
    frequencies = control_data.shared_state.gui_data.organized_data.(cell_field).frequencies;
    set(control_data.controls.freq_popup, 'String', frequencies);
    if ~isempty(frequencies)
        set(control_data.controls.freq_popup, 'Value', 1);
        frequencyChangedCallback(1);
    end
end
end