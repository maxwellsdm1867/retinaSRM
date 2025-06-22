function fourFactorGui(gui_data)
% FOURFACTORGUI - Interactive interface for 4-factor analysis
%
% USAGE:
%   fourFactorGui(gui_data)
%
% INPUT:
%   gui_data - Data structure from GUI extraction containing organized cell/frequency data

% Validate input
if ~isstruct(gui_data) || ~isfield(gui_data, 'organized_data')
    error('Invalid gui_data structure. Run GUI data extraction first.');
end

if isempty(gui_data.cell_names)
    error('No cell data found in gui_data. Check data extraction.');
end

% Create main figure
fig = uifigure('Name', 'Four Factor Analysis Interface', ...
    'Position', [100 100 1600 900], ...
    'Resize', 'on');

% Initialize GUI state
gui_state = struct();
gui_state.current_cell = gui_data.cell_names{1};
gui_state.current_frequency = '';
gui_state.current_data = [];
gui_state.vm_window = 3;      % Default pre-vm window (ms)
gui_state.dvdt_window = 3;    % Default dv/dt window (ms)
gui_state.count_window = 50;  % Default count window (ms)

% Get first cell's frequencies
cell_field = matlab.lang.makeValidName(gui_state.current_cell);
if isfield(gui_data.organized_data, cell_field)
    frequencies = gui_data.organized_data.(cell_field).frequencies;
    if ~isempty(frequencies)
        gui_state.current_frequency = frequencies{1};
        gui_state.current_data = gui_data.organized_data.(cell_field).data(gui_state.current_frequency);
        
        % Set initial windows to auto-calculated if available
        if isfield(gui_state.current_data, 'auto_window_ms')
            gui_state.vm_window = gui_state.current_data.auto_window_ms;
            gui_state.dvdt_window = gui_state.current_data.auto_window_ms;
        end
    end
end

% Store data in figure UserData for access by callbacks
fig.UserData = struct('gui_data', gui_data, 'gui_state', gui_state);
fprintf('DEBUG: UserData set, type: %s\n', class(fig.UserData));

% Create UI components in safe order
createAxes(fig);  % Create axes FIRST
fprintf('DEBUG: After createAxes, UserData type: %s\n', class(fig.UserData));

% Re-store UserData after createAxes in case it was cleared
fig.UserData.gui_data = gui_data;
fig.UserData.gui_state = gui_state;
fprintf('DEBUG: After re-storing, UserData type: %s\n', class(fig.UserData));

createControls(fig);  % Then controls (which may trigger updates)
fprintf('DEBUG: After createControls, UserData type: %s\n', class(fig.UserData));

updateDisplay(fig);
fprintf('DEBUG: After updateDisplay, UserData type: %s\n', class(fig.UserData));

fprintf('Four Factor GUI launched successfully\n');
fprintf('Current: %s - %s Hz\n', gui_state.current_cell, gui_state.current_frequency);

end

function createControls(fig)
% Create dropdown controls and sliders

% Verify UserData exists
if isempty(fig.UserData) || ~isstruct(fig.UserData)
    error('UserData not properly initialized');
end

% Top control panel
control_panel = uipanel(fig, 'Position', [0.02 0.9 0.96 0.08], ...
    'Title', 'Controls', 'FontWeight', 'bold');

% Cell dropdown
uilabel(control_panel, 'Position', [20 20 40 22], 'Text', 'Cell:');
cell_dropdown = uidropdown(control_panel, ...
    'Position', [70 20 100 22], ...
    'Items', fig.UserData.gui_data.cell_names, ...
    'Value', fig.UserData.gui_state.current_cell, ...
    'ValueChangedFcn', @(src, event) cellChanged(fig, src.Value));

% Frequency dropdown
uilabel(control_panel, 'Position', [190 20 70 22], 'Text', 'Frequency:');
freq_dropdown = uidropdown(control_panel, ...
    'Position', [270 20 80 22], ...
    'ValueChangedFcn', @(src, event) frequencyChanged(fig, src.Value));

% Auto window info
auto_label = uilabel(control_panel, 'Position', [370 20 200 22], ...
    'Text', 'Auto Window: calculating...', 'FontColor', [0 0.6 0]);

% Reset button
reset_btn = uibutton(control_panel, 'Text', 'Reset Windows', ...
    'Position', [590 20 100 22], ...
    'ButtonPushedFcn', @(src, event) resetWindows(fig));

% Store control handles
fig.UserData.controls = struct();
fig.UserData.controls.cell_dropdown = cell_dropdown;
fig.UserData.controls.freq_dropdown = freq_dropdown;
fig.UserData.controls.auto_label = auto_label;

% Update frequency dropdown for initial cell
updateFrequencyDropdown(fig);

end

function createAxes(fig)
% Create the 2x4 subplot layout matching analyzeSpikeInitiation4Factors

% Main plot area
plot_panel = uipanel(fig, 'Position', [0.02 0.02 0.96 0.86], ...
    'BorderType', 'none');

fprintf('DEBUG: Plot panel created at position [0.02 0.02 0.96 0.86]\n');

% Create 2x4 grid of axes using regular axes() instead of uiaxes()
axes_handles = struct();

% Calculate positions to match subplot(2,4,X) layout
subplot_width = 0.20;   % Width of each subplot
subplot_height = 0.35;  % Height of each subplot
left_margin = 0.08;     % Left margin
bottom_margin = 0.08;   % Bottom margin
h_spacing = 0.03;       % Horizontal spacing between plots
v_spacing = 0.08;       % Vertical spacing between rows

% TOP ROW (y = 0.55)
top_y = 0.55;

% Subplot [1 2]: Wide trace plot (spans 2 columns)
axes_handles.vm_traces = axes('Parent', plot_panel, 'Position', ...
    [left_margin, top_y, 2*subplot_width + h_spacing, subplot_height]);

% Subplot 3: Pre-Vm scatter
axes_handles.vm_scatter = axes('Parent', plot_panel, 'Position', ...
    [left_margin + 2*subplot_width + 2*h_spacing, top_y, subplot_width, subplot_height]);

% Subplot 4: ISI scatter  
axes_handles.isi_scatter = axes('Parent', plot_panel, 'Position', ...
    [left_margin + 3*subplot_width + 3*h_spacing, top_y, subplot_width, subplot_height]);

% BOTTOM ROW (y = 0.1)
bottom_y = 0.1;

% Subplot [5 6]: Wide dV/dt trace plot (spans 2 columns)
axes_handles.dvdt_traces = axes('Parent', plot_panel, 'Position', ...
    [left_margin, bottom_y, 2*subplot_width + h_spacing, subplot_height]);

% Subplot 7: dV/dt scatter
axes_handles.dvdt_scatter = axes('Parent', plot_panel, 'Position', ...
    [left_margin + 2*subplot_width + 2*h_spacing, bottom_y, subplot_width, subplot_height]);

% Subplot 8: Count scatter
axes_handles.count_scatter = axes('Parent', plot_panel, 'Position', ...
    [left_margin + 3*subplot_width + 3*h_spacing, bottom_y, subplot_width, subplot_height]);

fprintf('DEBUG: Created 6 axes\n');

% Add sliders with positions that let MATLAB handle the height automatically
% VM window slider (on top trace plot)
fprintf('DEBUG: Before vm_slider, UserData type: %s\n', class(fig.UserData));

% Create sliders without specifying height - let MATLAB use defaults
vm_slider = uislider(plot_panel);
vm_slider.Position = [left_margin + 0.02, top_y + subplot_height - 0.08, 0.4, 0.05];
vm_slider.Limits = [0.5 10];
vm_slider.Value = fig.UserData.gui_state.vm_window;

fprintf('DEBUG: After vm_slider, UserData type: %s\n', class(fig.UserData));

vm_label = uilabel(plot_panel, 'Position', [left_margin + 0.02, top_y + subplot_height - 0.03, 0.4, 0.03], ...
    'Text', sprintf('Pre-Vm Window: %.1f ms', fig.UserData.gui_state.vm_window), ...
    'BackgroundColor', 'white', 'HorizontalAlignment', 'center');

fprintf('DEBUG: After vm_label, UserData type: %s\n', class(fig.UserData));

% dV/dt window slider (on bottom trace plot) 
dvdt_slider = uislider(plot_panel);
dvdt_slider.Position = [left_margin + 0.02, bottom_y + subplot_height - 0.08, 0.4, 0.05];
dvdt_slider.Limits = [0.5 10];
dvdt_slider.Value = fig.UserData.gui_state.dvdt_window;

dvdt_label = uilabel(plot_panel, 'Position', [left_margin + 0.02, bottom_y + subplot_height - 0.03, 0.4, 0.03], ...
    'Text', sprintf('dV/dt Window: %.1f ms', fig.UserData.gui_state.dvdt_window), ...
    'BackgroundColor', 'white', 'HorizontalAlignment', 'center');

% Count window slider (on count scatter plot)
count_slider = uislider(plot_panel);
count_slider.Position = [left_margin + 3*subplot_width + 3*h_spacing + 0.01, bottom_y + 0.15, 0.15, 0.05];
count_slider.Limits = [10 200];
count_slider.Value = fig.UserData.gui_state.count_window;

count_label = uilabel(plot_panel, 'Position', [left_margin + 3*subplot_width + 3*h_spacing + 0.01, bottom_y + 0.20, 0.15, 0.03], ...
    'Text', sprintf('Count: %.0f ms', fig.UserData.gui_state.count_window), ...
    'BackgroundColor', 'white', 'HorizontalAlignment', 'center');

fprintf('DEBUG: Created sliders and labels\n');

% Store axes and slider handles
fprintf('DEBUG: About to store axes, UserData type: %s\n', class(fig.UserData));

if ~isstruct(fig.UserData)
    fprintf('ERROR: UserData is not a struct before storing axes!\n');
    fig.UserData = struct();
end

fig.UserData.axes = axes_handles;
fprintf('DEBUG: Stored axes, UserData type: %s\n', class(fig.UserData));

% Create sliders struct
fig.UserData.sliders = struct();
fig.UserData.sliders.vm_slider = vm_slider;
fig.UserData.sliders.vm_label = vm_label;
fig.UserData.sliders.dvdt_slider = dvdt_slider;
fig.UserData.sliders.dvdt_label = dvdt_label;
fig.UserData.sliders.count_slider = count_slider;
fig.UserData.sliders.count_label = count_label;

end

function updateDisplay(fig)
% Update all plots with current data and window settings

fprintf('DEBUG: updateDisplay called\n');

% Check UserData integrity first
if ~isstruct(fig.UserData)
    fprintf('ERROR: UserData corrupted in updateDisplay, type: %s\n', class(fig.UserData));
    return;
end

if ~isfield(fig.UserData, 'gui_state')
    fprintf('ERROR: UserData missing gui_state field\n');
    return;
end

if ~isfield(fig.UserData, 'axes')
    fprintf('ERROR: UserData missing axes field\n');
    return;
end

data = fig.UserData;
if isempty(data.gui_state.current_data)
    fprintf('DEBUG: No current_data, returning\n');
    return;
end

current_data = data.gui_state.current_data;
vm_window = data.gui_state.vm_window;
dvdt_window = data.gui_state.dvdt_window;
count_window = data.gui_state.count_window;

fprintf('DEBUG: Windows - VM: %.1f, dVdt: %.1f, Count: %.1f\n', vm_window, dvdt_window, count_window);

% Extract basic data
Vm_all = current_data.Vm_all;
spike_indices = current_data.spike_indices;
dt = current_data.dt;

fprintf('DEBUG: Data - Vm length: %d, spikes: %d, dt: %f\n', length(Vm_all), length(spike_indices), dt);

if length(spike_indices) < 2
    fprintf('DEBUG: Insufficient spikes (%d < 2), returning\n', length(spike_indices));
    return; % Need at least 2 spikes for analysis
end

% Calculate factors with current window settings
[factors, examples] = calculateFactors(Vm_all, spike_indices, dt, vm_window, dvdt_window, count_window);

fprintf('DEBUG: Calculated factors - initiation voltages: %d\n', length(factors.initiation_voltages));

% Update traces
updateTraces(fig, Vm_all, spike_indices, dt, examples, vm_window, dvdt_window);

% Update scatter plots
updateScatterPlots(fig, factors);

fprintf('DEBUG: Plots updated\n');

% Update auto window info - Check if controls exist
if isfield(data, 'controls') && isfield(data.controls, 'auto_label')
    if isfield(current_data, 'auto_window_ms')
        data.controls.auto_label.Text = sprintf('Auto Window: %.1f ms', current_data.auto_window_ms);
    else
        data.controls.auto_label.Text = 'Auto Window: not calculated';
    end
else
    fprintf('DEBUG: Controls not accessible for auto window update\n');
end

end

function [factors, examples] = calculateFactors(Vm_all, spike_indices, dt, vm_window, dvdt_window, count_window)
% Calculate all 4 factors with specified windows

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

% Find examples for traces
vm_range = max(avg_vm_before) - min(avg_vm_before);
dvdt_range = max(dvdt_before) - min(dvdt_before);

% Handle edge case where all values are the same
if vm_range == 0
    low_vm_threshold = min(avg_vm_before);
    high_vm_threshold = min(avg_vm_before);
else
    low_vm_threshold = min(avg_vm_before) + 0.33 * vm_range;
    high_vm_threshold = min(avg_vm_before) + 0.67 * vm_range;
end

if dvdt_range == 0
    slow_dvdt_threshold = min(dvdt_before);
    fast_dvdt_threshold = min(dvdt_before);
else
    slow_dvdt_threshold = min(dvdt_before) + 0.33 * dvdt_range;
    fast_dvdt_threshold = min(dvdt_before) + 0.67 * dvdt_range;
end

examples = struct();
examples.low_vm_idx = find(avg_vm_before <= low_vm_threshold, 1, 'first');
examples.high_vm_idx = find(avg_vm_before >= high_vm_threshold, 1, 'first');
examples.slow_dvdt_idx = find(dvdt_before <= slow_dvdt_threshold, 1, 'first');
examples.fast_dvdt_idx = find(dvdt_before >= fast_dvdt_threshold, 1, 'first');

% Store factors
factors = struct();
factors.initiation_voltages = initiation_voltages;
factors.avg_vm_before = avg_vm_before;
factors.dvdt_before = dvdt_before;
factors.isi_durations = isi_durations;
factors.counts_before = counts_before;

end

function updateTraces(fig, Vm_all, spike_indices, dt, examples, vm_window, dvdt_window)
% Update example trace plots

t_ms = (0:length(Vm_all)-1) * dt * 1000;
trace_window_ms = 20; % ±20ms around spike
trace_window_samples = round(trace_window_ms / (dt * 1000));

% Colors
colors = lines(4);

% VM traces
ax = fig.UserData.axes.vm_traces;
cla(ax);
hold(ax, 'on');

if ~isempty(examples.low_vm_idx) && examples.low_vm_idx <= length(spike_indices)-1
    idx = spike_indices(examples.low_vm_idx + 1);
    if idx > trace_window_samples && idx + trace_window_samples <= length(Vm_all)
        trace_start = idx - trace_window_samples;
        trace_end = idx + trace_window_samples;
        trace_t = t_ms(trace_start:trace_end) - t_ms(idx);
        trace_vm = Vm_all(trace_start:trace_end);
        plot(ax, trace_t, trace_vm, 'Color', colors(1,:), 'LineWidth', 2);
        plot(ax, 0, Vm_all(idx), 'o', 'Color', colors(1,:), 'MarkerSize', 8, 'MarkerFaceColor', colors(1,:));
    end
end

if ~isempty(examples.high_vm_idx) && examples.high_vm_idx <= length(spike_indices)-1
    idx = spike_indices(examples.high_vm_idx + 1);
    if idx > trace_window_samples && idx + trace_window_samples <= length(Vm_all)
        trace_start = idx - trace_window_samples;
        trace_end = idx + trace_window_samples;
        trace_t = t_ms(trace_start:trace_end) - t_ms(idx);
        trace_vm = Vm_all(trace_start:trace_end);
        plot(ax, trace_t, trace_vm, 'Color', colors(2,:), 'LineWidth', 2);
        plot(ax, 0, Vm_all(idx), 'o', 'Color', colors(2,:), 'MarkerSize', 8, 'MarkerFaceColor', colors(2,:));
    end
end

xlabel(ax, 'Time relative to spike initiation (ms)');
ylabel(ax, 'Vm (mV)');
title(ax, sprintf('Example Traces: Different Pre-Vm (%.1fms)', vm_window));
legend(ax, {'Low pre-Vm', 'High pre-Vm'}, 'Location', 'best');
grid(ax, 'on');
hold(ax, 'off');

% dV/dt traces
ax = fig.UserData.axes.dvdt_traces;
cla(ax);
hold(ax, 'on');

if ~isempty(examples.slow_dvdt_idx) && examples.slow_dvdt_idx <= length(spike_indices)-1
    idx = spike_indices(examples.slow_dvdt_idx + 1);
    if idx > trace_window_samples && idx + trace_window_samples <= length(Vm_all)
        trace_start = idx - trace_window_samples;
        trace_end = idx + trace_window_samples;
        trace_t = t_ms(trace_start:trace_end) - t_ms(idx);
        trace_vm = Vm_all(trace_start:trace_end);
        plot(ax, trace_t, trace_vm, 'Color', colors(3,:), 'LineWidth', 2);
        plot(ax, 0, Vm_all(idx), 'o', 'Color', colors(3,:), 'MarkerSize', 8, 'MarkerFaceColor', colors(3,:));
    end
end

if ~isempty(examples.fast_dvdt_idx) && examples.fast_dvdt_idx <= length(spike_indices)-1
    idx = spike_indices(examples.fast_dvdt_idx + 1);
    if idx > trace_window_samples && idx + trace_window_samples <= length(Vm_all)
        trace_start = idx - trace_window_samples;
        trace_end = idx + trace_window_samples;
        trace_t = t_ms(trace_start:trace_end) - t_ms(idx);
        trace_vm = Vm_all(trace_start:trace_end);
        plot(ax, trace_t, trace_vm, 'Color', colors(4,:), 'LineWidth', 2);
        plot(ax, 0, Vm_all(idx), 'o', 'Color', colors(4,:), 'MarkerSize', 8, 'MarkerFaceColor', colors(4,:));
    end
end

xlabel(ax, 'Time relative to spike initiation (ms)');
ylabel(ax, 'Vm (mV)');
title(ax, sprintf('Example Traces: Different dV/dt (%.1fms)', dvdt_window));
legend(ax, {'Slow approach', 'Fast approach'}, 'Location', 'best');
grid(ax, 'on');
hold(ax, 'off');

end

function updateScatterPlots(fig, factors)
% Update all scatter plots with correlations

fprintf('DEBUG: updateScatterPlots called with %d data points\n', length(factors.initiation_voltages));

% Check if we have data
if isempty(factors.initiation_voltages)
    % Clear all plots if no data
    axes_list = {'vm_scatter', 'dvdt_scatter', 'isi_scatter', 'count_scatter'};
    for i = 1:length(axes_list)
        ax = fig.UserData.axes.(axes_list{i});
        cla(ax);
        text(ax, 0.5, 0.5, 'Insufficient data', 'Units', 'normalized', ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
    end
    return;
end

colors = lines(4);

% Pre-Vm scatter
ax = fig.UserData.axes.vm_scatter;
fprintf('DEBUG: Plotting vm_scatter\n');
cla(ax);
if length(factors.avg_vm_before) > 0
    scatter(ax, factors.avg_vm_before, factors.initiation_voltages, 50, colors(1,:), 'filled');
    xlabel(ax, 'Avg Vm before (mV)');
    ylabel(ax, 'Initiation voltage (mV)');
    title(ax, 'Pre-Vm vs Initiation');
    grid(ax, 'on');
    
    % Force axes to be visible
    ax.Visible = 'on';
    
    if length(factors.avg_vm_before) > 2
        r_vm = corr(factors.avg_vm_before', factors.initiation_voltages');
        text(ax, 0.05, 0.95, sprintf('r = %.3f', r_vm), 'Units', 'normalized', ...
            'VerticalAlignment', 'top', 'BackgroundColor', 'white', 'FontSize', 10);
    end
    fprintf('DEBUG: vm_scatter plot completed\n');
end

% dV/dt scatter
ax = fig.UserData.axes.dvdt_scatter;
fprintf('DEBUG: Plotting dvdt_scatter\n');
cla(ax);
if length(factors.dvdt_before) > 0
    scatter(ax, factors.dvdt_before, factors.initiation_voltages, 50, colors(2,:), 'filled');
    xlabel(ax, 'dV/dt before (mV/s)');
    ylabel(ax, 'Initiation voltage (mV)');
    title(ax, 'dV/dt vs Initiation');
    grid(ax, 'on');
    
    % Force axes to be visible
    ax.Visible = 'on';
    
    if length(factors.dvdt_before) > 2
        r_dvdt = corr(factors.dvdt_before', factors.initiation_voltages');
        text(ax, 0.05, 0.95, sprintf('r = %.3f', r_dvdt), 'Units', 'normalized', ...
            'VerticalAlignment', 'top', 'BackgroundColor', 'white', 'FontSize', 10);
    end
    fprintf('DEBUG: dvdt_scatter plot completed\n');
end

% ISI scatter  
ax = fig.UserData.axes.isi_scatter;
fprintf('DEBUG: Plotting isi_scatter\n');
cla(ax);
if length(factors.isi_durations) > 0
    scatter(ax, factors.isi_durations, factors.initiation_voltages, 50, colors(3,:), 'filled');
    xlabel(ax, 'ISI duration (ms)');
    ylabel(ax, 'Initiation voltage (mV)');
    title(ax, 'ISI vs Initiation');
    set(ax, 'XScale', 'log');
    grid(ax, 'on');
    
    % Force axes to be visible
    ax.Visible = 'on';
    
    if length(factors.isi_durations) > 2
        r_isi = corr(log(factors.isi_durations'), factors.initiation_voltages');
        text(ax, 0.05, 0.95, sprintf('r = %.3f (log)', r_isi), 'Units', 'normalized', ...
            'VerticalAlignment', 'top', 'BackgroundColor', 'white', 'FontSize', 10);
    end
    fprintf('DEBUG: isi_scatter plot completed\n');
end

% Count scatter
ax = fig.UserData.axes.count_scatter;
fprintf('DEBUG: Plotting count_scatter\n');
cla(ax);
if length(factors.counts_before) > 0
    scatter(ax, factors.counts_before, factors.initiation_voltages, 50, colors(4,:), 'filled');
    xlabel(ax, sprintf('Count (%.0fms)', fig.UserData.gui_state.count_window));
    ylabel(ax, 'Initiation voltage (mV)');
    title(ax, 'Recent Activity vs Initiation');
    grid(ax, 'on');
    
    % Force axes to be visible
    ax.Visible = 'on';
    
    if length(factors.counts_before) > 2
        r_count = corr(factors.counts_before', factors.initiation_voltages');
        text(ax, 0.05, 0.95, sprintf('r = %.3f', r_count), 'Units', 'normalized', ...
            'VerticalAlignment', 'top', 'BackgroundColor', 'white', 'FontSize', 10);
    end
    fprintf('DEBUG: count_scatter plot completed\n');
end

fprintf('DEBUG: All scatter plots completed\n');

end

%% CALLBACK FUNCTIONS

function cellChanged(fig, new_cell)
% Handle cell dropdown change
fig.UserData.gui_state.current_cell = new_cell;
updateFrequencyDropdown(fig);
updateDisplay(fig);
fprintf('Switched to cell: %s\n', new_cell);
end

function frequencyChanged(fig, new_frequency)
% Handle frequency dropdown change  
fig.UserData.gui_state.current_frequency = new_frequency;
cell_field = matlab.lang.makeValidName(fig.UserData.gui_state.current_cell);
fig.UserData.gui_state.current_data = fig.UserData.gui_data.organized_data.(cell_field).data(new_frequency);

% Reset windows to auto-calculated for new frequency
if isfield(fig.UserData.gui_state.current_data, 'auto_window_ms')
    auto_window = fig.UserData.gui_state.current_data.auto_window_ms;
    fig.UserData.gui_state.vm_window = auto_window;
    fig.UserData.gui_state.dvdt_window = auto_window;
    
    % Update sliders
    fig.UserData.sliders.vm_slider.Value = auto_window;
    fig.UserData.sliders.dvdt_slider.Value = auto_window;
    fig.UserData.sliders.vm_label.Text = sprintf('Pre-Vm Window: %.1f ms', auto_window);
    fig.UserData.sliders.dvdt_label.Text = sprintf('dV/dt Window: %.1f ms', auto_window);
end

updateDisplay(fig);
fprintf('Switched to frequency: %s Hz\n', new_frequency);
end

function updateFrequencyDropdown(fig)
% Update frequency dropdown for current cell
cell_field = matlab.lang.makeValidName(fig.UserData.gui_state.current_cell);
if isfield(fig.UserData.gui_data.organized_data, cell_field)
    frequencies = fig.UserData.gui_data.organized_data.(cell_field).frequencies;
    fig.UserData.controls.freq_dropdown.Items = frequencies;
    if ~isempty(frequencies)
        fig.UserData.controls.freq_dropdown.Value = frequencies{1};
        frequencyChanged(fig, frequencies{1});
    end
end
end

function vmWindowChanged(fig, new_value)
% Handle VM window slider change
try
    if ~isstruct(fig.UserData)
        fprintf('ERROR: UserData corrupted in vmWindowChanged, type: %s\n', class(fig.UserData));
        return;
    end
    fig.UserData.gui_state.vm_window = new_value;
    fig.UserData.sliders.vm_label.Text = sprintf('Pre-Vm Window: %.1f ms', new_value);
    updateDisplay(fig);
catch ME
    fprintf('Error in vmWindowChanged: %s\n', ME.message);
end
end

function dvdtWindowChanged(fig, new_value)
% Handle dV/dt window slider change
try
    if ~isstruct(fig.UserData)
        fprintf('ERROR: UserData corrupted in dvdtWindowChanged, type: %s\n', class(fig.UserData));
        return;
    end
    fig.UserData.gui_state.dvdt_window = new_value;
    fig.UserData.sliders.dvdt_label.Text = sprintf('dV/dt Window: %.1f ms', new_value);
    updateDisplay(fig);
catch ME
    fprintf('Error in dvdtWindowChanged: %s\n', ME.message);
end
end

function countWindowChanged(fig, new_value)
% Handle count window slider change
try
    if ~isstruct(fig.UserData)
        fprintf('ERROR: UserData corrupted in countWindowChanged, type: %s\n', class(fig.UserData));
        return;
    end
    fig.UserData.gui_state.count_window = new_value;
    fig.UserData.sliders.count_label.Text = sprintf('Count: %.0f ms', new_value);
    updateDisplay(fig);
catch ME
    fprintf('Error in countWindowChanged: %s\n', ME.message);
end
end

function resetWindows(fig)
% Reset all windows to auto-calculated values
if isfield(fig.UserData.gui_state.current_data, 'auto_window_ms')
    auto_window = fig.UserData.gui_state.current_data.auto_window_ms;
    
    % Update state
    fig.UserData.gui_state.vm_window = auto_window;
    fig.UserData.gui_state.dvdt_window = auto_window;
    fig.UserData.gui_state.count_window = 50; % Reset to default
    
    % Update sliders
    fig.UserData.sliders.vm_slider.Value = auto_window;
    fig.UserData.sliders.dvdt_slider.Value = auto_window;
    fig.UserData.sliders.count_slider.Value = 50;
    
    % Update labels
    fig.UserData.sliders.vm_label.Text = sprintf('Pre-Vm Window: %.1f ms', auto_window);
    fig.UserData.sliders.dvdt_label.Text = sprintf('dV/dt Window: %.1f ms', auto_window);
    fig.UserData.sliders.count_label.Text = 'Count: 50 ms';
    
    updateDisplay(fig);
    fprintf('Reset windows to auto-calculated values\n');
else
    fprintf('No auto-calculated window available\n');
end
end