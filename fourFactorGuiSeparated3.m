function fourFactorGuiSeparated3(gui_data)
% FOURFACTORGUISEPARATED3 - Two-window interface for 4-factor analysis with spike filtering
%
% USAGE:
%   fourFactorGuiSeparated3(gui_data)
%
% INPUTS:
%   gui_data - Structure containing organized cell data, created by running the
%              "GUI DATA EXTRACTION" section in curreinjt_ai_population.m
%
% Creates separate windows for:
%   - Main plot window (6 subplots)
%   - Control panel window (hierarchical dropdowns, sliders)
%
% FEATURES:
%   - Hierarchical cell selection: Cell Type -> Cell -> Frequency
%   - Spike mode dropdown: Clean Spikes / All Spikes / Residual
%   - Exclusion window control for defining "clean" spikes
%   - Manual/Auto apply mode for parameter changes
%   - Axis locking, plot export, debug visualization
%   - Linked figure closing (both windows close together)
%   - Keyboard shortcuts for efficient workflow:
%     * W/S: Cycle through spike modes (forward/backward)
%     * C: Refresh analysis with current parameters
%     * Q: Toggle manual apply mode on/off
%     * E: Apply pending changes (in manual mode)
%     * R: Toggle axis lock on/off
%
% PREREQUISITES:
%   Before running this GUI, you must:
%   1. Run curreinjt_ai_population.m through the main analysis loop
%   2. Execute the "GUI DATA EXTRACTION" section at the end
%   3. Pass the resulting gui_data variable to this function

% Input validation with helpful error messages
if nargin < 1
    error(['No input provided. Usage: fourFactorGuiSeparated3(gui_data)\n' ...
           'First run the GUI DATA EXTRACTION section in curreinjt_ai_population.m, ' ...
           'then call this function with the gui_data variable.']);
end

if ~exist('gui_data', 'var')
    error(['GUI data not provided. Please run the GUI data extraction section ' ...
           'from curreinjt_ai_population.m first, then call this function with ' ...
           'the gui_data variable: fourFactorGuiSeparated3(gui_data)']);
end

if ~isstruct(gui_data) || ~isfield(gui_data, 'organized_data')
    error(['Invalid gui_data structure. The gui_data must be a struct with ' ...
           'an ''organized_data'' field. Please run the GUI DATA EXTRACTION ' ...
           'section in curreinjt_ai_population.m first.']);
end

if ~isfield(gui_data, 'cell_names') || isempty(gui_data.cell_names)
    error(['No cell data found in gui_data. Check that the GUI data extraction ' ...
           'section in curreinjt_ai_population.m successfully processed your data ' ...
           'and found cells with sufficient spike data (>= 5 spikes).']);
end

% Initialize shared state
shared_state = struct();
shared_state.gui_data = gui_data;

% Organize cells by type (extract from cell names)
shared_state.cell_organization = organizeCellsByType(gui_data);
cell_types = fieldnames(shared_state.cell_organization);
if ~isempty(cell_types)
    shared_state.current_cell_type = cell_types{1};
    type_cells = shared_state.cell_organization.(cell_types{1});
    if ~isempty(type_cells)
        shared_state.current_cell = type_cells{1};
    else
        shared_state.current_cell = gui_data.cell_names{1}; % Fallback
    end
else
    shared_state.current_cell_type = '';
    shared_state.current_cell = gui_data.cell_names{1}; % Fallback
end

shared_state.current_frequency = '';
shared_state.current_data = [];
shared_state.vm_window = 3;         % Default pre-vm window (ms)
shared_state.dvdt_window = 3;       % Default dv/dt window (ms)
shared_state.count_window = 50;     % Default count window (ms)
shared_state.spike_mode = 'All Spikes';     % NEW: Default spike mode
shared_state.exclusion_window = 10; % NEW: Default exclusion window (ms)

% Initialize control mode states
shared_state.manual_apply_mode = false;  % Default to auto mode
shared_state.has_pending_changes = false;  % Track if there are pending changes
shared_state.axis_locked = false;  % Track axis lock state
shared_state.locked_axis_limits = struct();  % Store locked axis limits

% Create debounce timer for smooth slider interactions
shared_state.update_timer = timer('ExecutionMode', 'singleShot', ...
                                  'TimerFcn', @(~,~) debouncedUpdateCallback(), ...
                                  'StartDelay', 0.25); % 250ms delay

% Check if spike visualization is requested
if isfield(gui_data, 'cleanSpikeVis') && gui_data.cleanSpikeVis
    shared_state.show_spike_vis = true;
else
    shared_state.show_spike_vis = false;
end

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
plot_fig = figure('Name', 'Four Factor Analysis - Plots (v3)', ...
    'Position', [100 100 1600 800], ...  % Wider for 2x6 layout
    'NumberTitle', 'off', ...
    'KeyPressFcn', @(src,evt) handleKeyPress(shared_state, evt), ...
    'CloseRequestFcn', @(src,evt) cleanupAndClose(shared_state, src));

% Create control panel figure
control_fig = figure('Name', 'Four Factor Analysis - Controls (v3)', ...
    'Position', [1320 100 320 850], ...  % Increased height to 850 to fit export section
    'NumberTitle', 'off', ...
    'KeyPressFcn', @(src,evt) handleKeyPress(shared_state, evt), ...
    'CloseRequestFcn', @(src,evt) cleanupAndClose(shared_state, src));

% Store shared state and figure handles
shared_state.plot_fig = plot_fig;
shared_state.control_fig = control_fig;

% Create the layouts
shared_state.subplots = createPlotLayout(shared_state);
createControlLayout(shared_state);

% Create spike filtering visualization if requested
if shared_state.show_spike_vis
    shared_state.vis_fig = createSpikeVisualization(shared_state);
end

% Create extreme groups visualization if requested
if isfield(shared_state.gui_data, 'plot_extreme_groups') && shared_state.gui_data.plot_extreme_groups
    shared_state.extreme_fig = createExtremeGroupsVisualization(shared_state);
end

% Initial update
updateAllPlots(shared_state);

fprintf('Four Factor GUI v3 launched successfully (separated windows)\n');
fprintf('Current: %s - %s Hz - Mode: %s\n', shared_state.current_cell, shared_state.current_frequency, shared_state.spike_mode);
fprintf('Keyboard shortcuts: W/S=spike mode, C=refresh, Q=manual mode, E=apply, R=axis lock (auto-refocus)\n');

% Keyboard shortcuts handler
    function handleKeyPress(~, event)
        % Handle keyboard shortcuts
        % w/s: Cycle through spike modes
        % c: Refresh everything according to current parameters
        % q: Toggle manual mode on/off
        % e: Apply changes in manual mode
        % r: Lock/unlock axis
        
        if isempty(event.Key)
            return;
        end
        
        % Get current shared state from control figure
        control_figs = findall(0, 'Name', 'Four Factor Analysis - Controls (v3)');
        if isempty(control_figs), return; end
        
        switch lower(event.Key)
            case 'w'
                % Cycle forward through spike modes
                cycleSpikeMode([], 1);
                
            case 's'
                % Cycle backward through spike modes
                cycleSpikeMode([], -1);
                
            case 'c'
                % Refresh everything according to current parameters
                refreshAnalysis([]);
                
            case 'q'
                % Toggle manual mode on/off
                toggleManualMode([]);
                
            case 'e'
                % Apply changes in manual mode
                applyPendingChanges([]);
                
            case 'r'
                % Lock/unlock axis
                toggleAxisLock([]);
                
            otherwise
                % Do nothing for other keys
        end
        
        % FOCUS MANAGEMENT: Keep focus on control panel for consecutive key presses
        % This prevents focus from returning to command line after keyboard shortcuts
        if ~isempty(control_figs)
            figure(control_figs(1));  % Bring control panel to front and focus
        end
    end

end

function subplots = createPlotLayout(shared_state)
% Create the 2x6 subplot layout in the plot figure

figure(shared_state.plot_fig);
clf;  % Clear the figure

% Create subplot handles - 2x6 layout as described
subplots = struct();

% TOP ROW
subplots.vm_random = subplot(2, 6, [1 2]);        % Columns 1-2: VM random traces
subplots.vm_traces = subplot(2, 6, [3 4]);        % Columns 3-4: VM example traces
subplots.vm_scatter = subplot(2, 6, 5);           % Column 5: Pre-VM scatter
subplots.isi_scatter = subplot(2, 6, 6);          % Column 6: ISI scatter

% BOTTOM ROW  
subplots.dvdt_random = subplot(2, 6, [7 8]);      % Columns 1-2: dV/dt random traces
subplots.dvdt_traces = subplot(2, 6, [9 10]);     % Columns 3-4: dV/dt example traces
subplots.dvdt_scatter = subplot(2, 6, 11);        % Column 5: dV/dt scatter
subplots.count_scatter = subplot(2, 6, 12);       % Column 6: Count scatter

% Add initial titles
title(subplots.vm_random, '20 Random VM Traces + Average');
title(subplots.vm_traces, 'VM Examples: Low vs High Pre-spike');
title(subplots.vm_scatter, 'Pre-spike Vm vs Initiation');
title(subplots.isi_scatter, 'ISI vs Initiation');
title(subplots.dvdt_random, '20 Random dV/dt Traces + Average');
title(subplots.dvdt_traces, 'dV/dt Examples: Slow vs Fast');
title(subplots.dvdt_scatter, 'dV/dt vs Initiation');
title(subplots.count_scatter, 'Recent Activity vs Initiation');

end

function createControlLayout(shared_state)
% Create the control panel in the control figure

figure(shared_state.control_fig);
clf;

% Create controls and store handles in figure UserData
controls = struct();

% Start from top with better spacing
y_pos = 820;  % Start higher due to taller window (850 height)

%% KEYBOARD SHORTCUTS SECTION
uicontrol('Style', 'text', 'String', '=== KEYBOARD SHORTCUTS ===', ...
    'Position', [20 y_pos 280 20], 'HorizontalAlignment', 'center', ...
    'FontWeight', 'bold', 'BackgroundColor', [0.8 0.9 1]);
y_pos = y_pos - 25;

% Keyboard shortcuts info
uicontrol('Style', 'text', 'String', 'W/S: Cycle spike modes  |  C: Refresh  |  Q: Manual mode', ...
    'Position', [20 y_pos 280 15], 'HorizontalAlignment', 'center', ...
    'FontSize', 8, 'ForegroundColor', [0 0 0.8]);
y_pos = y_pos - 15;
uicontrol('Style', 'text', 'String', 'E: Apply changes  |  R: Lock axis (persists!)  |  (Auto-refocus enabled)', ...
    'Position', [20 y_pos 280 15], 'HorizontalAlignment', 'center', ...
    'FontSize', 8, 'ForegroundColor', [0 0 0.8]);
y_pos = y_pos - 35;

%% DATA SELECTION SECTION
uicontrol('Style', 'text', 'String', '=== DATA SELECTION ===', ...
    'Position', [20 y_pos 280 20], 'HorizontalAlignment', 'center', ...
    'FontWeight', 'bold', 'BackgroundColor', [0.9 0.9 0.9]);
y_pos = y_pos - 30;

% Cell Type selection
uicontrol('Style', 'text', 'String', 'Cell Type:', ...
    'Position', [20 y_pos 70 20], 'HorizontalAlignment', 'left');
controls.celltype_popup = uicontrol('Style', 'popupmenu', ...
    'String', {''}, ...
    'Value', 1, ...
    'Position', [90 y_pos 190 25], ...
    'Callback', @(src, evt) cellTypeChangedCallback(src.Value));
y_pos = y_pos - 35;

% Cell selection (now depends on cell type)
uicontrol('Style', 'text', 'String', 'Cell:', ...
    'Position', [20 y_pos 60 20], 'HorizontalAlignment', 'left');
controls.cell_popup = uicontrol('Style', 'popupmenu', ...
    'String', {''}, ...
    'Value', 1, ...
    'Position', [80 y_pos 200 25], ...
    'Callback', @(src, evt) cellChangedCallback(src.Value));
y_pos = y_pos - 35;

% Frequency selection
uicontrol('Style', 'text', 'String', 'Frequency (Hz):', ...
    'Position', [20 y_pos 100 20], 'HorizontalAlignment', 'left');
controls.freq_popup = uicontrol('Style', 'popupmenu', ...
    'String', {''}, ...
    'Position', [120 y_pos 160 25], ...
    'Callback', @(src, evt) frequencyChangedCallback(src.Value));
y_pos = y_pos - 45;

%% SPIKE FILTERING SECTION
uicontrol('Style', 'text', 'String', '=== SPIKE FILTERING ===', ...
    'Position', [20 y_pos 280 20], 'HorizontalAlignment', 'center', ...
    'FontWeight', 'bold', 'BackgroundColor', [0.9 0.9 0.9]);
y_pos = y_pos - 30;

% Spike Mode selection
uicontrol('Style', 'text', 'String', 'Spike Mode:', ...
    'Position', [20 y_pos 100 20], 'HorizontalAlignment', 'left');
controls.spike_mode_popup = uicontrol('Style', 'popupmenu', ...
    'String', {'All Spikes', 'Clean Spikes', 'Residual'}, ...
    'Value', 1, ...
    'Position', [120 y_pos 160 25], ...
    'Callback', @(src, evt) spikeModeChangedCallback(src.Value));
% Add keyboard shortcut label for spike mode
uicontrol('Style', 'text', 'String', '[W/S]', ...
    'Position', [285 y_pos 25 20], 'HorizontalAlignment', 'center', ...
    'FontSize', 8, 'ForegroundColor', [0.5 0.5 0.5]);
y_pos = y_pos - 35;

% Exclusion Window slider and manual input
uicontrol('Style', 'text', 'String', 'Exclusion Window (ms):', ...
    'Position', [20 y_pos 150 20], 'HorizontalAlignment', 'left');
y_pos = y_pos - 25;

controls.exclusion_slider = uicontrol('Style', 'slider', ...
    'Min', 0, 'Max', 1150, 'Value', shared_state.exclusion_window, ...
    'Position', [20 y_pos 180 20], ...
    'SliderStep', [1/1150, 1/1150], ...  % Both small and large step = 1ms
    'Callback', @(src, evt) exclusionWindowChangedCallback(round(src.Value)));

controls.exclusion_edit = uicontrol('Style', 'edit', ...
    'String', sprintf('%.0f', shared_state.exclusion_window), ...
    'Position', [210 y_pos 40 20], ...
    'Callback', @(src, evt) exclusionEditCallback(str2double(src.String)));

controls.exclusion_label = uicontrol('Style', 'text', ...
    'String', sprintf('%.0f ms', shared_state.exclusion_window), ...
    'Position', [255 y_pos 60 20], 'HorizontalAlignment', 'left');
y_pos = y_pos - 45;

%% ANALYSIS WINDOWS SECTION  
uicontrol('Style', 'text', 'String', '=== ANALYSIS WINDOWS ===', ...
    'Position', [20 y_pos 280 20], 'HorizontalAlignment', 'center', ...
    'FontWeight', 'bold', 'BackgroundColor', [0.9 0.9 0.9]);
y_pos = y_pos - 30;

% VM Window slider and manual input
uicontrol('Style', 'text', 'String', 'Pre-VM Window (ms):', ...
    'Position', [20 y_pos 150 20], 'HorizontalAlignment', 'left');
y_pos = y_pos - 25;

controls.vm_slider = uicontrol('Style', 'slider', ...
    'Min', 0.5, 'Max', 10, 'Value', shared_state.vm_window, ...
    'Position', [20 y_pos 180 20], ...
    'Callback', @(src, evt) vmWindowChangedCallback(src.Value));

controls.vm_edit = uicontrol('Style', 'edit', ...
    'String', sprintf('%.1f', shared_state.vm_window), ...
    'Position', [210 y_pos 40 20], ...
    'Callback', @(src, evt) vmEditCallback(str2double(src.String)));

controls.vm_label = uicontrol('Style', 'text', ...
    'String', sprintf('%.1f ms', shared_state.vm_window), ...
    'Position', [255 y_pos 60 20], 'HorizontalAlignment', 'left');
y_pos = y_pos - 35;

% dV/dt Window slider and manual input
uicontrol('Style', 'text', 'String', 'dV/dt Window (ms):', ...
    'Position', [20 y_pos 150 20], 'HorizontalAlignment', 'left');
y_pos = y_pos - 25;

controls.dvdt_slider = uicontrol('Style', 'slider', ...
    'Min', 0.5, 'Max', 10, 'Value', shared_state.dvdt_window, ...
    'Position', [20 y_pos 180 20], ...
    'Callback', @(src, evt) dvdtWindowChangedCallback(src.Value));

controls.dvdt_edit = uicontrol('Style', 'edit', ...
    'String', sprintf('%.1f', shared_state.dvdt_window), ...
    'Position', [210 y_pos 40 20], ...
    'Callback', @(src, evt) dvdtEditCallback(str2double(src.String)));

controls.dvdt_label = uicontrol('Style', 'text', ...
    'String', sprintf('%.1f ms', shared_state.dvdt_window), ...
    'Position', [255 y_pos 60 20], 'HorizontalAlignment', 'left');
y_pos = y_pos - 35;

% Count Window slider and manual input
uicontrol('Style', 'text', 'String', 'Count Window (ms):', ...
    'Position', [20 y_pos 150 20], 'HorizontalAlignment', 'left');
y_pos = y_pos - 25;

controls.count_slider = uicontrol('Style', 'slider', ...
    'Min', 10, 'Max', 200, 'Value', shared_state.count_window, ...
    'Position', [20 y_pos 180 20], ...
    'Callback', @(src, evt) countWindowChangedCallback(src.Value));

controls.count_edit = uicontrol('Style', 'edit', ...
    'String', sprintf('%.0f', shared_state.count_window), ...
    'Position', [210 y_pos 40 20], ...
    'Callback', @(src, evt) countEditCallback(str2double(src.String)));

controls.count_label = uicontrol('Style', 'text', ...
    'String', sprintf('%.0f ms', shared_state.count_window), ...
    'Position', [255 y_pos 60 20], 'HorizontalAlignment', 'left');
y_pos = y_pos - 35;

% Auto window info
controls.auto_label = uicontrol('Style', 'text', ...
    'String', 'Auto Window: loading...', ...
    'Position', [20 y_pos 280 15], 'HorizontalAlignment', 'left', ...
    'ForegroundColor', [0 0.6 0], 'FontSize', 8);
y_pos = y_pos - 35;

%% ACTION BUTTONS SECTION
uicontrol('Style', 'text', 'String', '=== ACTIONS ===', ...
    'Position', [20 y_pos 280 20], 'HorizontalAlignment', 'center', ...
    'FontWeight', 'bold', 'BackgroundColor', [0.9 0.9 0.9]);
y_pos = y_pos - 30;

% Button row 1: Refresh and Save
controls.refresh_btn = uicontrol('Style', 'pushbutton', ...
    'String', 'Refresh Example Traces', ...
    'Position', [20 y_pos 130 25], ...
    'Callback', @(src, evt) refreshExamplesCallback());
% Add keyboard shortcut label for refresh
uicontrol('Style', 'text', 'String', '[C]', ...
    'Position', [150 y_pos+2 15 20], 'HorizontalAlignment', 'center', ...
    'FontSize', 8, 'ForegroundColor', [0.5 0.5 0.5]);

controls.save_btn = uicontrol('Style', 'pushbutton', ...
    'String', 'Save as JPG', ...
    'Position', [160 y_pos 120 25], ...
    'Callback', @(src, evt) saveJpgCallback());
y_pos = y_pos - 35;

% Debug button (only show if debug visualization is enabled)
if shared_state.show_spike_vis
    controls.refresh_debug_btn = uicontrol('Style', 'pushbutton', ...
        'String', 'Refresh Debug View', ...
        'Position', [20 y_pos 260 25], ...
        'Callback', @(src, evt) refreshDebugCallback());
    y_pos = y_pos - 35;
end

%% CONTROL MODE SECTION
uicontrol('Style', 'text', 'String', '=== CONTROL MODE ===', ...
    'Position', [20 y_pos 280 20], 'HorizontalAlignment', 'center', ...
    'FontWeight', 'bold', 'BackgroundColor', [0.9 0.9 0.9]);
y_pos = y_pos - 30;

% Manual Apply Mode checkbox
controls.manual_mode_checkbox = uicontrol('Style', 'checkbox', ...
    'String', 'Manual Apply Mode', ...
    'Position', [20 y_pos 180 20], ...
    'Value', 0, ...  % Default to auto mode (debounced)
    'Callback', @(src, evt) manualModeToggleCallback(src.Value));
% Add keyboard shortcut label for manual mode
uicontrol('Style', 'text', 'String', '[Q]', ...
    'Position', [200 y_pos 20 20], 'HorizontalAlignment', 'center', ...
    'FontSize', 8, 'ForegroundColor', [0.5 0.5 0.5]);
y_pos = y_pos - 30;

% Button row 2: Apply Changes and Axis Lock
controls.apply_btn = uicontrol('Style', 'pushbutton', ...
    'String', 'Apply Changes', ...
    'Position', [20 y_pos 120 25], ...
    'Enable', 'off', ...  % Initially disabled
    'BackgroundColor', [0.9 0.9 0.9], ...  % Gray when disabled
    'Callback', @(src, evt) applyChangesCallback());
% Add keyboard shortcut label for apply
uicontrol('Style', 'text', 'String', '[E]', ...
    'Position', [140 y_pos+2 15 20], 'HorizontalAlignment', 'center', ...
    'FontSize', 8, 'ForegroundColor', [0.5 0.5 0.5]);

controls.axis_lock_btn = uicontrol('Style', 'togglebutton', ...
    'String', 'Axis Lock: OFF', ...
    'Position', [150 y_pos 130 25], ...
    'Value', 0, ...  % Initially unlocked
    'BackgroundColor', [0.9 0.9 0.9], ...  % Light gray when off (matches new scheme)
    'Callback', @(src, evt) axisLockToggleCallback(src.Value));
% Add keyboard shortcut label for axis lock
uicontrol('Style', 'text', 'String', '[R]', ...
    'Position', [285 y_pos+2 15 20], 'HorizontalAlignment', 'center', ...
    'FontSize', 8, 'ForegroundColor', [0.5 0.5 0.5]);
y_pos = y_pos - 35;

%% DATA EXPORT SECTION
uicontrol('Style', 'text', 'String', '=== DATA EXPORT ===', ...
    'Position', [20 y_pos 280 20], 'HorizontalAlignment', 'center', ...
    'FontWeight', 'bold', 'BackgroundColor', [0.9 0.9 0.9]);
y_pos = y_pos - 30;

% Export button
controls.export_btn = uicontrol('Style', 'pushbutton', ...
    'String', 'Export Analysis & Plots', ...
    'Position', [20 y_pos 180 25], ...
    'BackgroundColor', [0.9 1 0.9], ...  % Light green
    'Callback', @(src, evt) exportAnalysisCallback());

% Store everything in the control figure's UserData
control_data = struct();
control_data.controls = controls;
control_data.shared_state = shared_state;
control_data.fixed_examples = []; % Store fixed examples for current cell/frequency
set(shared_state.control_fig, 'UserData', control_data);

% Initialize hierarchical dropdowns
initializeCellTypeDropdown();
updateCellDropdown();
updateFrequencyDropdown();

end

function cell_organization = organizeCellsByType(gui_data)
% Organize cells by actual cell type from protocol field
% Uses the protocol field which contains cell types like "RGC\OFF parasol"

cell_organization = struct();

fprintf('DEBUG: Starting cell organization. Total cells in gui_data.cell_names: %d\n', length(gui_data.cell_names));
fprintf('DEBUG: Available fields in gui_data.organized_data: %s\n', strjoin(fieldnames(gui_data.organized_data), ', '));

for i = 1:length(gui_data.cell_names)
    cell_name = gui_data.cell_names{i};
    cell_field = matlab.lang.makeValidName(cell_name);
    
    fprintf('DEBUG: Processing cell %d: "%s" -> field "%s"\n', i, cell_name, cell_field);
    
    if isfield(gui_data.organized_data, cell_field)
        % Get the first data entry to extract cell type
        frequencies = gui_data.organized_data.(cell_field).frequencies;
        fprintf('DEBUG: Cell %s has %d frequencies: %s\n', cell_name, length(frequencies), strjoin(frequencies, ', '));
        
        if ~isempty(frequencies)
            first_freq = frequencies{1};
            first_data = gui_data.organized_data.(cell_field).data(first_freq);
            
            % Extract cell type from protocol field
            if isstruct(first_data) && isfield(first_data, 'protocol') && ~isempty(first_data.protocol)
                cell_type = first_data.protocol;
                
                % Clean up cell type name for struct field name
                cell_type_field = matlab.lang.makeValidName(strrep(cell_type, '\', '_'));
                
                % Add to organization
                if ~isfield(cell_organization, cell_type_field)
                    cell_organization.(cell_type_field) = {};
                    % Store the display name for the cell type
                    cell_organization.([cell_type_field '_display']) = cell_type;
                end
                cell_organization.(cell_type_field){end+1} = cell_name;
                
                fprintf('Organized cell %s under type: %s\n', cell_name, cell_type);
            else
                % Fallback: if no protocol field, put in "Unknown" category
                if ~isfield(cell_organization, 'Unknown')
                    cell_organization.Unknown = {};
                    cell_organization.Unknown_display = 'Unknown';
                end
                cell_organization.Unknown{end+1} = cell_name;
                fprintf('Cell %s has no protocol info, placed in Unknown\n', cell_name);
            end
        else
            fprintf('WARNING: Cell %s has no frequency data, skipping\n', cell_name);
        end
    else
        fprintf('ERROR: Cell field %s not found in organized_data (original name: %s)\n', cell_field, cell_name);
        fprintf('DEBUG: This could be due to name validation issues in data extraction\n');
    end
end

% If no organization was possible, create a single "All" category
if isempty(fieldnames(cell_organization))
    cell_organization.All = gui_data.cell_names;
    cell_organization.All_display = 'All Cells';
    fprintf('No protocol data found, created single "All" category\n');
end

fprintf('Cell organization complete. Found cell types:\n');
cell_type_fields = fieldnames(cell_organization);
for i = 1:length(cell_type_fields)
    field_name = cell_type_fields{i};
    if ~endsWith(field_name, '_display') && isfield(cell_organization, [field_name '_display'])
        display_name = cell_organization.([field_name '_display']);
        cell_count = length(cell_organization.(field_name));
        fprintf('  %s: %d cells\n', display_name, cell_count);
    end
end

end

function display_name = createDisplayName(original_name, parts)
% Create a display name from cell name parts
% Simplified to handle pre-formatted date-cell combinations (e.g., "06-May-2025_Cell1")

display_name = original_name; % Default fallback

if length(parts) >= 2
    date_part = '';
    cell_part = '';
    
    for i = 1:length(parts)
        part = parts{i};
        
        % Check if this looks like a pre-formatted date (contains dashes)
        if isempty(date_part) && contains(part, '-')
            % If it contains dashes, assume it's already formatted nicely
            date_part = part;
        end
        
        % Look for cell identifiers
        if isempty(cell_part)
            % Pattern 1: Contains "cell" (case insensitive)
            if contains(lower(part), 'cell')
                cell_part = part;
            % Pattern 2: Starts with 'C' followed by digits
            elseif length(part) >= 2 && part(1) == 'C' && all(isstrprop(part(2:end), 'digit'))
                cell_part = part;
            % Pattern 3: Just digits (potential cell number)
            elseif all(isstrprop(part, 'digit')) && length(part) <= 3 && str2double(part) > 0
                cell_part = ['Cell' part];
            end
        end
    end
    
    % Create display name with both date and cell when available
    if ~isempty(date_part) && ~isempty(cell_part)
        display_name = sprintf('%s - %s', date_part, cell_part);
    elseif ~isempty(date_part)
        % If we have date but no clear cell identifier, try to extract from original name
        if contains(lower(original_name), 'cell')
            display_name = sprintf('%s - %s', date_part, original_name);
        else
            display_name = date_part;
        end
    elseif ~isempty(cell_part)
        display_name = cell_part;
    else
        % Last resort: try to make the original name more readable
        display_name = strrep(strrep(original_name, '_', ' '), '-', ' ');
    end
end

end

function filtered_indices = filterSpikes(spike_indices, mode, exclusion_window_ms, dt)
% Filter spikes based on the selected mode
% 
% INPUTS:
%   spike_indices: original spike indices
%   mode: 'All Spikes', 'Clean Spikes', or 'Residual'
%   exclusion_window_ms: window size in ms for exclusion
%   dt: sampling interval in seconds
%
% OUTPUTS:
%   filtered_indices: filtered spike indices

if strcmp(mode, 'All Spikes')
    filtered_indices = spike_indices;
    return;
end

% Convert exclusion window to samples
exclusion_samples = round(exclusion_window_ms / (dt * 1000));

% Find clean spikes (no preceding spike within exclusion window)
clean_mask = true(size(spike_indices));

for i = 1:length(spike_indices)
    current_idx = spike_indices(i);
    
    % Check if any previous spike falls within exclusion window
    for j = 1:i-1
        prev_idx = spike_indices(j);
        if (current_idx - prev_idx) <= exclusion_samples
            clean_mask(i) = false;
            break;
        end
    end
end

if strcmp(mode, 'Clean Spikes')
    filtered_indices = spike_indices(clean_mask);
else % 'Residual'
    filtered_indices = spike_indices(~clean_mask);
end

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
spike_mode = shared_state.spike_mode;
exclusion_window = shared_state.exclusion_window;

% Extract basic data
Vm_all = current_data.Vm_all;
spike_indices = current_data.spike_indices;
dt = current_data.dt;

if length(spike_indices) < 2
    return; % Need at least 2 spikes for analysis
end

% NEW: Filter spikes based on mode
filtered_spike_indices = filterSpikes(spike_indices, spike_mode, exclusion_window, dt);

if length(filtered_spike_indices) < 2
    fprintf('Warning: Only %d spikes after filtering with mode "%s"\n', length(filtered_spike_indices), spike_mode);
    if isempty(filtered_spike_indices)
        return; % Can't analyze with no spikes
    end
end

% Calculate factors with filtered spikes
[factors, examples] = calculateFactors(Vm_all, filtered_spike_indices, dt, vm_window, dvdt_window, count_window);

% Make sure we're plotting in the plot figure
figure(shared_state.plot_fig);

% Add supertitle with comprehensive cell information including spike filtering info
cell_info = sprintf('Cell: %s | Frequency: %s Hz | Mode: %s | Spikes: %d/%d (Total: %d) | Duration: %.1fs', ...
    shared_state.current_cell, shared_state.current_frequency, spike_mode, ...
    length(filtered_spike_indices), length(spike_indices), length(spike_indices), length(Vm_all));

% Add exclusion window info for clean/residual modes
if ~strcmp(spike_mode, 'All Spikes')
    cell_info = sprintf('%s | Exclusion: %.0fms', cell_info, exclusion_window);
end

% Try to extract additional metadata - show cell type and date
cell_type_display = 'Unknown';
if ~isempty(shared_state.current_cell_type) && isfield(shared_state.cell_organization, [shared_state.current_cell_type '_display'])
    cell_type_display = shared_state.cell_organization.([shared_state.current_cell_type '_display']);
elseif isfield(current_data, 'protocol') && ~isempty(current_data.protocol)
    cell_type_display = current_data.protocol;
end

if isfield(current_data, 'date')
    cell_info = sprintf('%s | Cell Type: %s | Date: %s', cell_info, cell_type_display, current_data.date);
else
    cell_info = sprintf('%s | Cell Type: %s', cell_info, cell_type_display);
end

sgtitle(cell_info, 'FontSize', 14, 'FontWeight', 'bold');

% Update traces with filtered spikes
updateTraces(shared_state, Vm_all, filtered_spike_indices, dt, examples, vm_window, dvdt_window);

% Update scatter plots
updateScatterPlots(shared_state, factors, examples);

% Update spike filtering visualization if enabled (function not yet implemented)
% if shared_state.show_spike_vis && isfield(shared_state, 'vis_fig') && ishandle(shared_state.vis_fig)
%     updateSpikeVisualization(shared_state, shared_state.vis_fig);
% end

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
% Update all trace plots - both random traces and example traces

% Changed trace window: -10ms to +3ms around spike initiation
pre_window_ms = 10;   % 10ms before spike
post_window_ms = 3;   % 3ms after spike
pre_window_samples = round(pre_window_ms / (dt * 1000));
post_window_samples = round(post_window_ms / (dt * 1000));
t_ms = (0:length(Vm_all)-1) * dt * 1000;
colors = lines(4);

% Calculate reasonable y-axis limits from the data
y_limits = calculateYLimits(Vm_all, spike_indices, pre_window_samples, post_window_samples);

% Update VM random traces (columns 1-2)
updateRandomTraces(shared_state, Vm_all, spike_indices, dt, vm_window, y_limits, 'vm');

% Update VM example traces (columns 3-4) 
updateExampleTraces(shared_state, Vm_all, spike_indices, dt, examples, vm_window, y_limits, 'vm');

% Update dV/dt random traces (columns 1-2)
updateRandomTraces(shared_state, Vm_all, spike_indices, dt, dvdt_window, y_limits, 'dvdt');

% Update dV/dt example traces (columns 3-4)
updateExampleTraces(shared_state, Vm_all, spike_indices, dt, examples, dvdt_window, y_limits, 'dvdt');

end

function updateRandomTraces(shared_state, Vm_all, spike_indices, dt, window, y_limits, trace_type)
% Update random trace plots with 20 traces + average + shaded window

pre_window_ms = 10;   
post_window_ms = 3;   
pre_window_samples = round(pre_window_ms / (dt * 1000));
post_window_samples = round(post_window_ms / (dt * 1000));
t_ms = (0:length(Vm_all)-1) * dt * 1000;

% Select subplot
if strcmp(trace_type, 'vm')
    subplot(shared_state.subplots.vm_random);
    window_start = -window;
    window_end = 0;
    title_str = sprintf('20 Random VM Traces + Average (%.1fms window)', window);
    shaded_color = [0.8 0.8 0.8];
    ylabel_str = 'Vm (mV)';
else
    subplot(shared_state.subplots.dvdt_random);
    window_start = -window;
    window_end = 0;
    title_str = sprintf('20 Random dV/dt Traces + Average (%.1fms window)', window);
    shaded_color = [0.9 0.9 0.6];
    ylabel_str = 'dV/dt (mV/s)';
end

cla;
hold on;

% Select 20 random spikes for display
valid_indices = [];
for i = 1:length(spike_indices)
    idx = spike_indices(i);
    if idx > pre_window_samples && idx + post_window_samples <= length(Vm_all)
        valid_indices(end+1) = i;
    end
end

if length(valid_indices) >= 20
    selected_indices = valid_indices(randperm(length(valid_indices), 20));
else
    selected_indices = valid_indices;
end

% Compute traces and collect for average
all_traces = [];
trace_times = [];
dvdt_ylimits = [inf -inf];  % Track dV/dt limits
vm_ylimits = [inf -inf];    % Track VM limits

for i = 1:length(selected_indices)
    spike_idx = spike_indices(selected_indices(i));
    trace_start = spike_idx - pre_window_samples;
    trace_end = spike_idx + post_window_samples;
    trace_t = t_ms(trace_start:trace_end) - t_ms(spike_idx);
    
    if strcmp(trace_type, 'vm')
        trace_data = Vm_all(trace_start:trace_end);
        % Track VM limits
        vm_ylimits(1) = min(vm_ylimits(1), min(trace_data));
        vm_ylimits(2) = max(vm_ylimits(2), max(trace_data));
    else
        % For dV/dt traces, compute derivative properly
        vm_trace = Vm_all(trace_start:trace_end);
        % Compute dV/dt: central difference for interior points
        dvdt_trace = zeros(size(vm_trace));
        dvdt_trace(1) = (vm_trace(2) - vm_trace(1)) / dt;  % Forward diff at start
        dvdt_trace(end) = (vm_trace(end) - vm_trace(end-1)) / dt;  % Backward diff at end
        for j = 2:length(vm_trace)-1
            dvdt_trace(j) = (vm_trace(j+1) - vm_trace(j-1)) / (2 * dt);  % Central diff
        end
        trace_data = dvdt_trace / 1000;  % Convert to V/s, then *1000 for mV/s gives mV/ms
        
        % Track dV/dt limits
        dvdt_ylimits(1) = min(dvdt_ylimits(1), min(trace_data));
        dvdt_ylimits(2) = max(dvdt_ylimits(2), max(trace_data));
    end
    
    % Plot individual trace in light gray
    plot(trace_t, trace_data, 'Color', [0.7 0.7 0.7], 'LineWidth', 0.5);
    
    % Store for average
    if isempty(all_traces)
        all_traces = trace_data;
        trace_times = trace_t;
    else
        all_traces = [all_traces, trace_data];
    end
end

% Set appropriate y-limits
if strcmp(trace_type, 'vm')
    % For VM, use the provided y_limits
    current_ylimits = y_limits;
else
    % For dV/dt, use computed limits with some margin
    if dvdt_ylimits(1) < dvdt_ylimits(2)
        margin = (dvdt_ylimits(2) - dvdt_ylimits(1)) * 0.1;
        current_ylimits = [dvdt_ylimits(1) - margin, dvdt_ylimits(2) + margin];
    else
        current_ylimits = [-50 50];  % Default dV/dt range
    end
end

% Draw analysis window
patch([window_start window_end window_end window_start], ...
      [current_ylimits(1) current_ylimits(1) current_ylimits(2) current_ylimits(2)], ...
      shaded_color, 'FaceAlpha', 0.3, 'EdgeColor', 'none');

% Plot average trace in black
if ~isempty(all_traces)
    avg_trace = mean(all_traces, 2);
    plot(trace_times, avg_trace, 'k-', 'LineWidth', 3);
    
    % Mark spike initiation point
    spike_time_idx = find(trace_times >= -0.1 & trace_times <= 0.1, 1);  % Find t=0
    if ~isempty(spike_time_idx)
        plot(0, avg_trace(spike_time_idx), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
    end
end

% Draw vertical line at spike initiation
plot([0 0], current_ylimits, 'k--', 'LineWidth', 1);

xlabel('Time relative to spike initiation (ms)');
ylabel(ylabel_str);
title(title_str);
xlim([-10 3]);
ylim(current_ylimits);
grid on;
hold off;

end

function updateExampleTraces(shared_state, Vm_all, spike_indices, dt, examples, window, y_limits, trace_type)
% Update example trace plots (existing functionality)

pre_window_ms = 10;   
post_window_ms = 3;   
pre_window_samples = round(pre_window_ms / (dt * 1000));
post_window_samples = round(post_window_ms / (dt * 1000));
t_ms = (0:length(Vm_all)-1) * dt * 1000;
colors = lines(4);

% Select subplot and get examples
if strcmp(trace_type, 'vm')
    subplot(shared_state.subplots.vm_traces);
    low_idx = examples.low_vm_idx;
    high_idx = examples.high_vm_idx;
    window_start = -window;
    window_end = 0;
    title_str = sprintf('VM Examples: Low vs High Pre-spike (%.1fms window)', window);
    shaded_color = [0.8 0.8 0.8];
    legend_low = 'Low pre-Vm';
    legend_high = 'High pre-Vm';
else
    subplot(shared_state.subplots.dvdt_traces);
    low_idx = examples.slow_dvdt_idx;
    high_idx = examples.fast_dvdt_idx;
    window_start = -window;
    window_end = 0;
    title_str = sprintf('dV/dt Examples: Slow vs Fast (%.1fms window)', window);
    shaded_color = [0.9 0.9 0.6];
    legend_low = 'Slow approach';
    legend_high = 'Fast approach';
end

cla;
hold on;

% Draw analysis window
patch([window_start window_end window_end window_start], ...
      [y_limits(1) y_limits(1) y_limits(2) y_limits(2)], ...
      shaded_color, 'FaceAlpha', 0.3, 'EdgeColor', 'none');

% Plot example traces
legend_entries = {sprintf('%.1fms window', window)};

if ~isempty(low_idx) && low_idx <= length(spike_indices)
    idx = spike_indices(low_idx);
    if idx > pre_window_samples && idx + post_window_samples <= length(Vm_all)
        trace_start = idx - pre_window_samples;
        trace_end = idx + post_window_samples;
        trace_t = t_ms(trace_start:trace_end) - t_ms(idx);
        trace_vm = Vm_all(trace_start:trace_end);
        plot(trace_t, trace_vm, 'Color', colors(1,:), 'LineWidth', 2);
        plot(0, Vm_all(idx), 'o', 'Color', colors(1,:), 'MarkerSize', 8, 'MarkerFaceColor', colors(1,:), 'LineWidth', 2);
        legend_entries{end+1} = legend_low;
    end
end

if ~isempty(high_idx) && high_idx <= length(spike_indices)
    idx = spike_indices(high_idx);
    if idx > pre_window_samples && idx + post_window_samples <= length(Vm_all)
        trace_start = idx - pre_window_samples;
        trace_end = idx + post_window_samples;
        trace_t = t_ms(trace_start:trace_end) - t_ms(idx);
        trace_vm = Vm_all(trace_start:trace_end);
        plot(trace_t, trace_vm, 'Color', colors(2,:), 'LineWidth', 2);
        plot(0, Vm_all(idx), 'o', 'Color', colors(2,:), 'MarkerSize', 8, 'MarkerFaceColor', colors(2,:), 'LineWidth', 2);
        legend_entries{end+1} = legend_high;
    end
end

% Draw vertical line at spike initiation
plot([0 0], y_limits, 'k--', 'LineWidth', 1);

xlabel('Time relative to spike initiation (ms)');
ylabel('Vm (mV)');
title(title_str);
if length(legend_entries) > 1
    legend(legend_entries, 'Location', 'best');
end
xlim([-10 3]);
ylim(y_limits);
grid on;
hold off;

end

function updateScatterPlots(shared_state, factors, examples)
% Update all scatter plots with correlations - EXACT match to working code
% Now includes markers for example traces and color coding for clean/residual spikes
% Added axis locking to maintain consistent scales when switching spike modes

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

% Initialize axis limits storage if not present
if ~isfield(shared_state, 'scatter_axis_limits')
    shared_state.scatter_axis_limits = struct();
end

% NEW: Determine clean/residual status for color coding
% Handle color coding for different spike modes
clean_colors = [];
if ~isempty(shared_state.current_data)
    current_data = shared_state.current_data;
    all_spike_indices = current_data.spike_indices;
    dt = current_data.dt;
    exclusion_window = shared_state.exclusion_window;
    
    if strcmp(shared_state.spike_mode, 'All Spikes')
        % All Spikes mode: Color code each spike as clean (green) or residual (blue)
        clean_spike_indices = filterSpikes(all_spike_indices, 'Clean Spikes', exclusion_window, dt);
        
        % Create color array: green for clean spikes, blue for residual spikes
        clean_colors = zeros(length(spike_initiation_voltages), 3);
        
        % The factors correspond to spikes starting from index 2 (since ISI calculation skips first spike)
        % Map each spike in factors to clean (green) or residual (blue)
        for i = 1:length(spike_initiation_voltages)
            current_spike_idx = all_spike_indices(i + 1); % +1 because factors skip first spike
            
            % Check if this spike is in the clean list
            if any(clean_spike_indices == current_spike_idx)
                clean_colors(i, :) = [0, 0.8, 0]; % Green for clean spikes
            else
                clean_colors(i, :) = [0, 0.4, 0.8]; % Blue for residual spikes
            end
        end
        
    elseif strcmp(shared_state.spike_mode, 'Clean Spikes')
        % Clean Spikes mode: All points should be green (since they're all clean)
        clean_colors = repmat([0, 0.8, 0], length(spike_initiation_voltages), 1);
        
    elseif strcmp(shared_state.spike_mode, 'Residual')
        % Residual mode: All points should be blue (since they're all residual)
        clean_colors = repmat([0, 0.4, 0.8], length(spike_initiation_voltages), 1);
    end
end

% Subplot 3: Average Vm before vs Spike initiation voltage
subplot(shared_state.subplots.vm_scatter);
cla;

% Use color coding if available, otherwise use default colors
if ~isempty(clean_colors)
    scatter(avg_vm_before, spike_initiation_voltages, 50, clean_colors, 'filled', 'MarkerFaceAlpha', 0.8);
else
    scatter(avg_vm_before, spike_initiation_voltages, 50, colors(1,:), 'filled', 'MarkerFaceAlpha', 0.6);
end
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

% Add legend for color coding if applicable
if ~isempty(clean_colors)
    if strcmp(shared_state.spike_mode, 'All Spikes')
        % Show both clean and residual in legend for All Spikes mode
        h1 = scatter(NaN, NaN, 50, [0, 0.8, 0], 'filled', 'MarkerFaceAlpha', 0.8);
        h2 = scatter(NaN, NaN, 50, [0, 0.4, 0.8], 'filled', 'MarkerFaceAlpha', 0.8);
        legend([h1, h2], {'Clean Spikes', 'Residual Spikes'}, 'Location', 'best', 'FontSize', 8);
    elseif strcmp(shared_state.spike_mode, 'Clean Spikes')
        % Show only clean spikes in legend
        h1 = scatter(NaN, NaN, 50, [0, 0.8, 0], 'filled', 'MarkerFaceAlpha', 0.8);
        legend(h1, {'Clean Spikes'}, 'Location', 'best', 'FontSize', 8);
    elseif strcmp(shared_state.spike_mode, 'Residual')
        % Show only residual spikes in legend
        h2 = scatter(NaN, NaN, 50, [0, 0.4, 0.8], 'filled', 'MarkerFaceAlpha', 0.8);
        legend(h2, {'Residual Spikes'}, 'Location', 'best', 'FontSize', 8);
    end
end

hold off;

if length(avg_vm_before) > 2
    r_vm = corr(avg_vm_before', spike_initiation_voltages');
    text(0.05, 0.95, sprintf('r = %.3f', r_vm), 'Units', 'normalized', ...
        'VerticalAlignment', 'top', 'BackgroundColor', 'white', 'FontSize', 10);
end

% Handle axis locking for VM scatter plot
if shared_state.axis_locked && isfield(shared_state, 'locked_axis_limits') && ...
   isfield(shared_state.locked_axis_limits, 'vm_scatter_xlim') && isfield(shared_state.locked_axis_limits, 'vm_scatter_ylim')
    % Apply locked axis limits
    xlim(shared_state.locked_axis_limits.vm_scatter_xlim);
    ylim(shared_state.locked_axis_limits.vm_scatter_ylim);
end

% Subplot 4: ISI vs Spike initiation voltage
subplot(shared_state.subplots.isi_scatter);
cla;

% Use color coding if available, otherwise use default colors
if ~isempty(clean_colors)
    scatter(isi_durations, spike_initiation_voltages, 50, clean_colors, 'filled', 'MarkerFaceAlpha', 0.8);
else
    scatter(isi_durations, spike_initiation_voltages, 50, colors(3,:), 'filled', 'MarkerFaceAlpha', 0.6);
end
xlabel('ISI duration (ms)');
ylabel('Spike init voltage (mV)');
title('ISI vs Initiation');
grid on;
set(gca, 'XScale', 'log'); % Log scale for ISI

% Add legend for color coding if applicable
if ~isempty(clean_colors)
    hold on;
    if strcmp(shared_state.spike_mode, 'All Spikes')
        % Show both clean and residual in legend for All Spikes mode
        h1 = scatter(NaN, NaN, 50, [0, 0.8, 0], 'filled', 'MarkerFaceAlpha', 0.8);
        h2 = scatter(NaN, NaN, 50, [0, 0.4, 0.8], 'filled', 'MarkerFaceAlpha', 0.8);
        legend([h1, h2], {'Clean Spikes', 'Residual Spikes'}, 'Location', 'best', 'FontSize', 8);
    elseif strcmp(shared_state.spike_mode, 'Clean Spikes')
        % Show only clean spikes in legend
        h1 = scatter(NaN, NaN, 50, [0, 0.8, 0], 'filled', 'MarkerFaceAlpha', 0.8);
        legend(h1, {'Clean Spikes'}, 'Location', 'best', 'FontSize', 8);
    elseif strcmp(shared_state.spike_mode, 'Residual')
        % Show only residual spikes in legend
        h2 = scatter(NaN, NaN, 50, [0, 0.4, 0.8], 'filled', 'MarkerFaceAlpha', 0.8);
        legend(h2, {'Residual Spikes'}, 'Location', 'best', 'FontSize', 8);
    end
    hold off;
end

if length(isi_durations) > 2
    r_isi = corr(log(isi_durations'), spike_initiation_voltages');
    text(0.05, 0.95, sprintf('r = %.3f (log)', r_isi), 'Units', 'normalized', ...
        'VerticalAlignment', 'top', 'BackgroundColor', 'white', 'FontSize', 10);
end

% Handle axis locking for ISI scatter plot
if shared_state.axis_locked && isfield(shared_state, 'locked_axis_limits') && ...
   isfield(shared_state.locked_axis_limits, 'isi_scatter_xlim') && isfield(shared_state.locked_axis_limits, 'isi_scatter_ylim')
    % Apply locked axis limits
    xlim(shared_state.locked_axis_limits.isi_scatter_xlim);
    ylim(shared_state.locked_axis_limits.isi_scatter_ylim);
end

% Subplot 7: dV/dt before vs Spike initiation voltage
subplot(shared_state.subplots.dvdt_scatter);
cla;

% Use color coding if available, otherwise use default colors
if ~isempty(clean_colors)
    scatter(dvdt_before, spike_initiation_voltages, 50, clean_colors, 'filled', 'MarkerFaceAlpha', 0.8);
else
    scatter(dvdt_before, spike_initiation_voltages, 50, colors(2,:), 'filled', 'MarkerFaceAlpha', 0.6);
end
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

% Add legend for color coding if applicable
if ~isempty(clean_colors)
    if strcmp(shared_state.spike_mode, 'All Spikes')
        % Show both clean and residual in legend for All Spikes mode
        h1 = scatter(NaN, NaN, 50, [0, 0.8, 0], 'filled', 'MarkerFaceAlpha', 0.8);
        h2 = scatter(NaN, NaN, 50, [0, 0.4, 0.8], 'filled', 'MarkerFaceAlpha', 0.8);
        legend([h1, h2], {'Clean Spikes', 'Residual Spikes'}, 'Location', 'best', 'FontSize', 8);
    elseif strcmp(shared_state.spike_mode, 'Clean Spikes')
        % Show only clean spikes in legend
        h1 = scatter(NaN, NaN, 50, [0, 0.8, 0], 'filled', 'MarkerFaceAlpha', 0.8);
        legend(h1, {'Clean Spikes'}, 'Location', 'best', 'FontSize', 8);
    elseif strcmp(shared_state.spike_mode, 'Residual')
        % Show only residual spikes in legend
        h2 = scatter(NaN, NaN, 50, [0, 0.4, 0.8], 'filled', 'MarkerFaceAlpha', 0.8);
        legend(h2, {'Residual Spikes'}, 'Location', 'best', 'FontSize', 8);
    end
end

hold off;

if length(dvdt_before) > 2
    r_dvdt = corr(dvdt_before', spike_initiation_voltages');
    text(0.05, 0.95, sprintf('r = %.3f', r_dvdt), 'Units', 'normalized', ...
        'VerticalAlignment', 'top', 'BackgroundColor', 'white', 'FontSize', 10);
end

% Handle axis locking for dV/dt scatter plot
if shared_state.axis_locked && isfield(shared_state, 'locked_axis_limits') && ...
   isfield(shared_state.locked_axis_limits, 'dvdt_scatter_xlim') && isfield(shared_state.locked_axis_limits, 'dvdt_scatter_ylim')
    % Apply locked axis limits
    xlim(shared_state.locked_axis_limits.dvdt_scatter_xlim);
    ylim(shared_state.locked_axis_limits.dvdt_scatter_ylim);
end

% Subplot 8: Spike count vs Spike initiation voltage
subplot(shared_state.subplots.count_scatter);
cla;

% Use color coding if available, otherwise use default colors
if ~isempty(clean_colors)
    scatter(spike_counts_before, spike_initiation_voltages, 50, clean_colors, 'filled', 'MarkerFaceAlpha', 0.8);
else
    scatter(spike_counts_before, spike_initiation_voltages, 50, colors(4,:), 'filled', 'MarkerFaceAlpha', 0.6);
end
xlabel(sprintf('Spike count (%.0fms)', spike_count_window_ms));
ylabel('Spike init voltage (mV)');
title(sprintf('Recent Activity (%.1fms) vs Initiation', spike_count_window_ms));
grid on;

% Add legend for color coding if applicable
if ~isempty(clean_colors)
    hold on;
    if strcmp(shared_state.spike_mode, 'All Spikes')
        % Show both clean and residual in legend for All Spikes mode
        h1 = scatter(NaN, NaN, 50, [0, 0.8, 0], 'filled', 'MarkerFaceAlpha', 0.8);
        h2 = scatter(NaN, NaN, 50, [0, 0.4, 0.8], 'filled', 'MarkerFaceAlpha', 0.8);
        legend([h1, h2], {'Clean Spikes', 'Residual Spikes'}, 'Location', 'best', 'FontSize', 8);
    elseif strcmp(shared_state.spike_mode, 'Clean Spikes')
        % Show only clean spikes in legend
        h1 = scatter(NaN, NaN, 50, [0, 0.8, 0], 'filled', 'MarkerFaceAlpha', 0.8);
        legend(h1, {'Clean Spikes'}, 'Location', 'best', 'FontSize', 8);
    elseif strcmp(shared_state.spike_mode, 'Residual')
        % Show only residual spikes in legend
        h2 = scatter(NaN, NaN, 50, [0, 0.4, 0.8], 'filled', 'MarkerFaceAlpha', 0.8);
        legend(h2, {'Residual Spikes'}, 'Location', 'best', 'FontSize', 8);
    end
    hold off;
end

if length(spike_counts_before) > 2
    r_count = corr(spike_counts_before', spike_initiation_voltages');
    text(0.05, 0.95, sprintf('r = %.3f', r_count), 'Units', 'normalized', ...
        'VerticalAlignment', 'top', 'BackgroundColor', 'white', 'FontSize', 10);
end

% Handle axis locking for spike count scatter plot
if shared_state.axis_locked && isfield(shared_state, 'locked_axis_limits') && ...
   isfield(shared_state.locked_axis_limits, 'count_scatter_xlim') && isfield(shared_state.locked_axis_limits, 'count_scatter_ylim')
    % Apply locked axis limits
    xlim(shared_state.locked_axis_limits.count_scatter_xlim);
    ylim(shared_state.locked_axis_limits.count_scatter_ylim);
end

end

%% NEW CALLBACK FUNCTIONS FOR SPIKE FILTERING

function spikeModeChangedCallback(new_index)
% Handle spike mode dropdown change
control_figs = findall(0, 'Name', 'Four Factor Analysis - Controls (v3)');
if isempty(control_figs), return; end
control_data = get(control_figs(1), 'UserData');

% Update state
modes = {'All Spikes', 'Clean Spikes', 'Residual'};
control_data.shared_state.spike_mode = modes{new_index};

% Generate new fixed examples for the new mode
if ~isempty(control_data.shared_state.current_data)
    current_data = control_data.shared_state.current_data;
    Vm_all = current_data.Vm_all;
    spike_indices = current_data.spike_indices;
    dt = current_data.dt;
    
    % Filter spikes based on new mode
    filtered_spike_indices = filterSpikes(spike_indices, control_data.shared_state.spike_mode, ...
        control_data.shared_state.exclusion_window, dt);
    
    if length(filtered_spike_indices) >= 2
        vm_window = control_data.shared_state.vm_window;
        dvdt_window = control_data.shared_state.dvdt_window;
        count_window = control_data.shared_state.count_window;
        
        % Calculate new examples for filtered spikes
        [~, new_examples] = calculateFactors(Vm_all, filtered_spike_indices, dt, vm_window, dvdt_window, count_window, false);
        control_data.fixed_examples = new_examples;
    else
        control_data.fixed_examples = [];
    end
end

% Update UserData with new state
set(control_figs(1), 'UserData', control_data);

% Check if we're in manual apply mode
if control_data.shared_state.manual_apply_mode
    % Manual mode: Just mark as having pending changes and enable Apply button
    control_data.shared_state.has_pending_changes = true;
    set(control_data.controls.apply_btn, 'Enable', 'on');
    set(control_data.controls.apply_btn, 'BackgroundColor', [0.2 0.8 0.2]);  % Green when enabled
    set(control_data.controls.apply_btn, 'String', 'Apply Changes *');  % Asterisk indicates pending
    set(control_figs(1), 'UserData', control_data);
    fprintf('Switched to spike mode: %s (pending - click Apply)\n', control_data.shared_state.spike_mode);
else
    % Auto mode: Apply changes immediately
    actualApplyChanges(control_data);
    set(control_figs(1), 'UserData', control_data);
    fprintf('Switched to spike mode: %s\n', control_data.shared_state.spike_mode);
end
end

function exclusionWindowChangedCallback(new_value)
% Get data from control figure
control_figs = findall(0, 'Name', 'Four Factor Analysis - Controls (v3)');
if isempty(control_figs), return; end
control_data = get(control_figs(1), 'UserData');

% Update state and UI controls immediately (for responsive feel)
control_data.shared_state.exclusion_window = new_value;
set(control_data.controls.exclusion_slider, 'Value', new_value);
set(control_data.controls.exclusion_edit, 'String', sprintf('%.0f', new_value));
set(control_data.controls.exclusion_label, 'String', sprintf('%.0f ms', new_value));

% Update UserData immediately
set(control_figs(1), 'UserData', control_data);

% Check if we're in manual apply mode
if control_data.shared_state.manual_apply_mode
    % Manual mode: Just mark as having pending changes and enable Apply button
    control_data.shared_state.has_pending_changes = true;
    set(control_data.controls.apply_btn, 'Enable', 'on');
    set(control_data.controls.apply_btn, 'BackgroundColor', [0.2 0.8 0.2]);  % Green when enabled
    set(control_data.controls.apply_btn, 'String', 'Apply Changes *');  % Asterisk indicates pending
    set(control_figs(1), 'UserData', control_data);
    fprintf('Exclusion window changed to: %.0fms (pending - click Apply)\n', new_value);
else
    % Auto mode: Use debounced update as before
    % Show visual feedback that update is pending
    plot_figs = findall(0, 'Name', 'Four Factor Analysis - Plots (v3)');
    if ~isempty(plot_figs)
        figure(plot_figs(1));
        sgtitle('Four Factor Analysis - Updating...', 'FontSize', 16, 'Color', 'r');
    end
    
    % DEBOUNCED UPDATE: Cancel any pending update and start a new delayed update
    try
        stop(control_data.shared_state.update_timer);
    catch
        % Timer might not be running, ignore error
    end
    start(control_data.shared_state.update_timer);
    
    fprintf('Exclusion window changed to: %.0fms (auto-updating...)\n', new_value);
end
end

function exclusionEditCallback(new_value)
% Handle manual exclusion window input
if isnan(new_value) || new_value < 0 || new_value > 1150
    fprintf('Invalid exclusion window value. Must be between 0 and 1150 ms\n');
    return;
end
exclusionWindowChangedCallback(new_value);
end

function applyChangesCallback()
% Apply pending changes when Apply button is clicked
control_figs = findall(0, 'Name', 'Four Factor Analysis - Controls (v3)');
if isempty(control_figs), return; end
control_data = get(control_figs(1), 'UserData');

if control_data.shared_state.manual_apply_mode && control_data.shared_state.has_pending_changes
    % Apply the changes
    actualApplyChanges(control_data);
    
    % Clear pending changes
    control_data.shared_state.has_pending_changes = false;
    
    % Update Apply button state
    set(control_data.controls.apply_btn, 'Enable', 'off');
    set(control_data.controls.apply_btn, 'BackgroundColor', [0.9 0.9 0.9]);
    set(control_data.controls.apply_btn, 'String', 'Apply Changes');
    
    set(control_figs(1), 'UserData', control_data);
    fprintf('Applied pending changes\n');
else
    fprintf('No pending changes to apply\n');
end
end

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
control_figs = findall(0, 'Name', 'Four Factor Analysis - Controls (v3)');
if isempty(control_figs), return; end
control_data = get(control_figs(1), 'UserData');

% Update state and controls
control_data.shared_state.vm_window = new_value;
set(control_data.controls.vm_slider, 'Value', new_value);
set(control_data.controls.vm_edit, 'String', sprintf('%.1f', new_value));
set(control_data.controls.vm_label, 'String', sprintf('%.1f ms', new_value));

% Update UserData with new state
set(control_figs(1), 'UserData', control_data);

% Handle manual/auto mode
if control_data.shared_state.manual_apply_mode
    % Manual mode: Just mark as having pending changes and enable Apply button
    control_data.shared_state.has_pending_changes = true;
    set(control_data.controls.apply_btn, 'Enable', 'on');
    set(control_data.controls.apply_btn, 'BackgroundColor', [0.2 0.8 0.2]);  % Green when enabled
    set(control_data.controls.apply_btn, 'String', 'Apply Changes *');  % Asterisk indicates pending
    set(control_figs(1), 'UserData', control_data);
    fprintf('VM window changed to: %.1fms (pending - click Apply)\n', new_value);
else
    % In auto mode, use debounced update
    if isvalid(control_data.shared_state.update_timer)
        stop(control_data.shared_state.update_timer);
        start(control_data.shared_state.update_timer);
    end
    fprintf('VM window changed to: %.1fms (auto-updating...)\n', new_value);
end
end

function dvdtWindowChangedCallback(new_value)
% Get data from control figure
control_figs = findall(0, 'Name', 'Four Factor Analysis - Controls (v3)');
if isempty(control_figs), return; end
control_data = get(control_figs(1), 'UserData');

% Update state and controls
control_data.shared_state.dvdt_window = new_value;
set(control_data.controls.dvdt_slider, 'Value', new_value);
set(control_data.controls.dvdt_edit, 'String', sprintf('%.1f', new_value));
set(control_data.controls.dvdt_label, 'String', sprintf('%.1f ms', new_value));

% Update UserData with new state
set(control_figs(1), 'UserData', control_data);

% Handle manual/auto mode
if control_data.shared_state.manual_apply_mode
    % Manual mode: Just mark as having pending changes and enable Apply button
    control_data.shared_state.has_pending_changes = true;
    set(control_data.controls.apply_btn, 'Enable', 'on');
    set(control_data.controls.apply_btn, 'BackgroundColor', [0.2 0.8 0.2]);  % Green when enabled
    set(control_data.controls.apply_btn, 'String', 'Apply Changes *');  % Asterisk indicates pending
    set(control_figs(1), 'UserData', control_data);
    fprintf('dV/dt window changed to: %.1fms (pending - click Apply)\n', new_value);
else
    % In auto mode, use debounced update
    if isvalid(control_data.shared_state.update_timer)
        stop(control_data.shared_state.update_timer);
        start(control_data.shared_state.update_timer);
    end
    fprintf('dV/dt window changed to: %.1fms (auto-updating...)\n', new_value);
end
end

function countWindowChangedCallback(new_value)
% Get data from control figure
control_figs = findall(0, 'Name', 'Four Factor Analysis - Controls (v3)');
if isempty(control_figs), return; end
control_data = get(control_figs(1), 'UserData');

% Update state and controls
control_data.shared_state.count_window = new_value;
set(control_data.controls.count_slider, 'Value', new_value);
set(control_data.controls.count_edit, 'String', sprintf('%.0f', new_value));
set(control_data.controls.count_label, 'String', sprintf('%.0f ms', new_value));

% Update UserData with new state
set(control_figs(1), 'UserData', control_data);

% Handle manual/auto mode
if control_data.shared_state.manual_apply_mode
    % Manual mode: Just mark as having pending changes and enable Apply button
    control_data.shared_state.has_pending_changes = true;
    set(control_data.controls.apply_btn, 'Enable', 'on');
    set(control_data.controls.apply_btn, 'BackgroundColor', [0.2 0.8 0.2]);  % Green when enabled
    set(control_data.controls.apply_btn, 'String', 'Apply Changes *');  % Asterisk indicates pending
    set(control_figs(1), 'UserData', control_data);
    fprintf('Count window changed to: %.0fms (pending - click Apply)\n', new_value);
else
    % In auto mode, use debounced update
    if isvalid(control_data.shared_state.update_timer)
        stop(control_data.shared_state.update_timer);
        start(control_data.shared_state.update_timer);
    end
    fprintf('Count window changed to: %.0fms (auto-updating...)\n', new_value);
end
end

function initializeCellTypeDropdown()
% Initialize the cell type dropdown with available cell types
control_figs = findall(0, 'Name', 'Four Factor Analysis - Controls (v3)');
if isempty(control_figs), return; end
control_data = get(control_figs(1), 'UserData');

cell_types = fieldnames(control_data.shared_state.cell_organization);
% Filter out display name fields and create display names
display_names = {};
field_names = {};
for i = 1:length(cell_types)
    field_name = cell_types{i};
    if ~endsWith(field_name, '_display') && isfield(control_data.shared_state.cell_organization, [field_name '_display'])
        display_name = control_data.shared_state.cell_organization.([field_name '_display']);
        display_names{end+1} = display_name;
        field_names{end+1} = field_name;
        fprintf('Cell type dropdown: %s -> %s\n', field_name, display_name);
    end
end

% Set dropdown to display names
set(control_data.controls.celltype_popup, 'String', display_names);
% Store field names for reference
control_data.cell_type_fields = field_names;
set(control_figs(1), 'UserData', control_data);

if ~isempty(display_names)
    set(control_data.controls.celltype_popup, 'Value', 1);
end
end

function updateCellDropdown()
% Update the cell dropdown based on selected cell type
control_figs = findall(0, 'Name', 'Four Factor Analysis - Controls (v3)');
if isempty(control_figs), return; end
control_data = get(control_figs(1), 'UserData');

current_cell_type = control_data.shared_state.current_cell_type;
if isfield(control_data.shared_state.cell_organization, current_cell_type)
    cells_of_type = control_data.shared_state.cell_organization.(current_cell_type);
    
    % Try to preserve current cell selection
    current_cell = control_data.shared_state.current_cell;
    cell_index = 1; % Default to first cell
    
    % Find the index of the current cell if it exists in the new list
    if ~isempty(current_cell)
        cell_match = find(strcmp(cells_of_type, current_cell));
        if ~isempty(cell_match)
            cell_index = cell_match(1); % Use first match if multiple
        end
    end
    
    % Create display names for cells
    display_names = cell(size(cells_of_type));
    for i = 1:length(cells_of_type)
        parts = split(cells_of_type{i}, '_');
        display_names{i} = createDisplayName(cells_of_type{i}, parts);
    end
    
    set(control_data.controls.cell_popup, 'String', display_names);
    if ~isempty(cells_of_type)
        set(control_data.controls.cell_popup, 'Value', cell_index);
        % Update the current cell name to match the selection
        control_data.shared_state.current_cell = cells_of_type{cell_index};
        set(control_figs(1), 'UserData', control_data);
    end
else
    set(control_data.controls.cell_popup, 'String', {''});
    set(control_data.controls.cell_popup, 'Value', 1);
end
end

function cellTypeChangedCallback(new_index)
% Handle cell type dropdown change
control_figs = findall(0, 'Name', 'Four Factor Analysis - Controls (v3)');
if isempty(control_figs), return; end
control_data = get(control_figs(1), 'UserData');

% Get the field name from the stored mapping
if isfield(control_data, 'cell_type_fields') && new_index <= length(control_data.cell_type_fields)
    field_name = control_data.cell_type_fields{new_index};
    control_data.shared_state.current_cell_type = field_name;
    
    % Update cell dropdown
    cells_of_type = control_data.shared_state.cell_organization.(field_name);
    
    % Create display names for cells
    display_names = cell(size(cells_of_type));
    for i = 1:length(cells_of_type)
        parts = split(cells_of_type{i}, '_');
        display_names{i} = createDisplayName(cells_of_type{i}, parts);
    end
    
    set(control_data.controls.cell_popup, 'String', display_names);
    if ~isempty(cells_of_type)
        set(control_data.controls.cell_popup, 'Value', 1);
        
        % Update current cell name
        control_data.shared_state.current_cell = cells_of_type{1};
    end
    
    % Save updated data
    set(control_figs(1), 'UserData', control_data);
    
    % Update frequency dropdown and refresh plots
    updateFrequencyDropdown();
    updateAllPlotsFixedExamples(control_data.shared_state, control_data.fixed_examples);
    
    % Update extreme groups visualization if enabled
    if isfield(control_data.shared_state.gui_data, 'plot_extreme_groups') && control_data.shared_state.gui_data.plot_extreme_groups && ...
       isfield(control_data.shared_state, 'extreme_fig') && ishandle(control_data.shared_state.extreme_fig)
        updateExtremeGroupsVisualization(control_data.shared_state, control_data.shared_state.extreme_fig);
    end
    
    % Get display name for logging
    display_name = 'Unknown';
    if isfield(control_data.shared_state.cell_organization, [field_name '_display'])
        display_name = control_data.shared_state.cell_organization.([field_name '_display']);
    end
    
    fprintf('Switched to cell type: %s (%s)\n', display_name, field_name);
else
    fprintf('Error: Invalid cell type index %d\n', new_index);
end
end

function cellChangedCallback(new_index)
% Get data from control figure
control_figs = findall(0, 'Name', 'Four Factor Analysis - Controls (v3)');
if isempty(control_figs), return; end
control_data = get(control_figs(1), 'UserData');

% Get the actual cell name from the selected index within the current cell type
current_cell_type = control_data.shared_state.current_cell_type;
if isfield(control_data.shared_state.cell_organization, current_cell_type)
    cells_of_type = control_data.shared_state.cell_organization.(current_cell_type);
    if new_index <= length(cells_of_type)
        control_data.shared_state.current_cell = cells_of_type{new_index};
    end
else
    % Fallback to old method
    control_data.shared_state.current_cell = control_data.shared_state.gui_data.cell_names{new_index};
end

% Update UserData with new state
set(control_figs(1), 'UserData', control_data);

% AXIS LOCK PRESERVED: Keep axis lock state when changing cells
% This allows for consistent comparisons across cells

% Update frequency dropdown and plots
updateFrequencyDropdown();
fprintf('Switched to cell: %s\n', control_data.shared_state.current_cell);
end

function frequencyChangedCallback(new_index)
% Get data from control figure
control_figs = findall(0, 'Name', 'Four Factor Analysis - Controls (v3)');
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
    
    % Filter spikes based on current mode
    filtered_spike_indices = filterSpikes(spike_indices, control_data.shared_state.spike_mode, ...
        control_data.shared_state.exclusion_window, dt);
    
    if length(filtered_spike_indices) >= 2
        vm_window = control_data.shared_state.vm_window;
        dvdt_window = control_data.shared_state.dvdt_window;
        count_window = control_data.shared_state.count_window;
        
        % Calculate default examples for new frequency
        [~, new_examples] = calculateFactors(Vm_all, filtered_spike_indices, dt, vm_window, dvdt_window, count_window, false);
        control_data.fixed_examples = new_examples;
    else
        control_data.fixed_examples = [];
    end
end

% Update UserData with new state
set(control_figs(1), 'UserData', control_data);

% AXIS LOCK PRESERVED: Keep axis lock state when changing frequency
% This allows for consistent comparisons across frequencies

% Update auto window display
if isfield(control_data.shared_state.current_data, 'auto_window_ms')
    auto_window = control_data.shared_state.current_data.auto_window_ms;
    set(control_data.controls.auto_label, 'String', sprintf('Injected Current Autocorr: %.1f ms', auto_window));
else
    set(control_data.controls.auto_label, 'String', 'Injected Current Autocorr: not available');
end

% Update plots with fixed examples
updateAllPlotsFixedExamples(control_data.shared_state, control_data.fixed_examples);

% Update extreme groups visualization if enabled
if isfield(control_data.shared_state.gui_data, 'plot_extreme_groups') && control_data.shared_state.gui_data.plot_extreme_groups && ...
   isfield(control_data.shared_state, 'extreme_fig') && ishandle(control_data.shared_state.extreme_fig)
    updateExtremeGroupsVisualization(control_data.shared_state, control_data.shared_state.extreme_fig);
end

% Update spike visualization if enabled (function not yet implemented)
% if control_data.shared_state.show_spike_vis && isfield(control_data.shared_state, 'vis_fig') && ishandle(control_data.shared_state.vis_fig)
%     updateSpikeVisualization(control_data.shared_state, control_data.shared_state.vis_fig);
% end

fprintf('Switched to frequency: %s Hz\n', control_data.shared_state.current_frequency);
end

function saveJpgCallback()
% Save the current plot figure as JPG
plot_figs = findall(0, 'Name', 'Four Factor Analysis - Plots (v3)');
if isempty(plot_figs)
    fprintf('No plot figure found to save\n');
    return;
end

% Get current cell and frequency info for filename
control_figs = findall(0, 'Name', 'Four Factor Analysis - Controls (v3)');
if ~isempty(control_figs)
    control_data = get(control_figs(1), 'UserData');
    cell_name = control_data.shared_state.current_cell;
    frequency = control_data.shared_state.current_frequency;
    spike_mode = control_data.shared_state.spike_mode;
    exclusion_window = control_data.shared_state.exclusion_window;
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
    spike_mode_safe = strrep(spike_mode, ' ', '');
    
    filename = sprintf('FourFactor_%s_%sHz_%s_Ex%.0f_VM%.1f_dVdt%.1f_Count%.0f_%s.jpg', ...
        cell_name_safe, frequency_safe, spike_mode_safe, exclusion_window, ...
        vm_window, dvdt_window, count_window, timestamp);
    
    % Full path
    full_path = fullfile(save_path, filename);
else
    % Fallback filename
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    filename = sprintf('FourFactorAnalysis_v3_%s.jpg', timestamp);
    full_path = filename;
end

% Save the plot figure
figure(plot_figs(1));

% Try to save with error handling
try
    % Use a simpler approach to avoid path issues
    [filepath, ~, ~] = fileparts(full_path);
    if isempty(filepath)
        filepath = pwd;
    end
    
    % Ensure directory exists
    if ~exist(filepath, 'dir')
        mkdir(filepath);
    end
    
    % Save using exportgraphics for better compatibility (if available)
    if exist('exportgraphics', 'file')
        exportgraphics(plot_figs(1), full_path, 'Resolution', 300);
    else
        % Fallback to print with simpler options
        print(plot_figs(1), '-djpeg', '-r300', full_path);
    end
    
    fprintf('Saved plot as: %s\n', full_path);
    
catch ME
    % If that fails, try saving to current directory with a simple name
    fprintf('Error saving to %s: %s\n', full_path, ME.message);
    
    % Fallback: save to current directory with timestamp
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    fallback_name = sprintf('FourFactorPlot_%s.jpg', timestamp);
    
    try
        if exist('exportgraphics', 'file')
            exportgraphics(plot_figs(1), fallback_name, 'Resolution', 300);
        else
            print(plot_figs(1), '-djpeg', '-r300', fallback_name);
        end
        fprintf('Saved plot as fallback: %s\n', fallback_name);
    catch ME2
        fprintf('Failed to save plot: %s\n', ME2.message);
    end
end
end

function extreme_fig = createExtremeGroupsVisualization(shared_state)
% Create a figure to visualize extreme groups (highest/lowest VM, fastest/slowest dV/dt)
% Top row: VM extremes (low, high, scatter with marked points)
% Bottom row: dV/dt extremes (slow, fast, scatter with marked points)

extreme_fig = figure('Name', 'Extreme Groups Analysis', ...
    'Position', [50 50 1400 900], ...
    'NumberTitle', 'off');

% Add title
sgtitle('Extreme Groups Analysis - Top/Bottom 20 Examples', 'FontSize', 16, 'FontWeight', 'bold');

% Initialize with current data if available
if ~isempty(shared_state.current_data)
    updateExtremeGroupsVisualization(shared_state, extreme_fig);
end

end

function updateExtremeGroupsVisualization(shared_state, extreme_fig)
% Update the extreme groups visualization with current data and settings

if isempty(shared_state.current_data)
    return;
end

current_data = shared_state.current_data;
Vm_all = current_data.Vm_all;
spike_indices = current_data.spike_indices;
dt = current_data.dt;
vm_window = shared_state.vm_window;
dvdt_window = shared_state.dvdt_window;
spike_mode = shared_state.spike_mode;
exclusion_window = shared_state.exclusion_window;

if length(spike_indices) < 2
    return;
end

% Filter spikes based on current mode
filtered_spike_indices = filterSpikes(spike_indices, spike_mode, exclusion_window, dt);

if length(filtered_spike_indices) < 20
    fprintf('Warning: Need at least 20 spikes for extreme groups analysis (have %d)\n', length(filtered_spike_indices));
    return;
end

% Calculate factors for all spikes
[factors, ~] = calculateFactors(Vm_all, filtered_spike_indices, dt, vm_window, dvdt_window, shared_state.count_window, false);

% Check if we have sufficient data
if isempty(factors.avg_vm_before) || length(factors.avg_vm_before) < 4
    fprintf('Warning: Insufficient data for extreme groups visualization\n');
    return;
end

% Extract VM and dV/dt values from the struct
vm_values = factors.avg_vm_before;  % Pre-spike VM
dvdt_values = factors.dvdt_before;  % dV/dt

% Find extreme indices (top/bottom 20)
n_extremes = min(20, floor(length(filtered_spike_indices) / 4)); % Use 20 or 1/4 of available spikes
[~, vm_low_idx] = sort(vm_values, 'ascend');
[~, vm_high_idx] = sort(vm_values, 'descend');
[~, dvdt_slow_idx] = sort(dvdt_values, 'ascend');
[~, dvdt_fast_idx] = sort(dvdt_values, 'descend');

vm_low_extremes = vm_low_idx(1:n_extremes);
vm_high_extremes = vm_high_idx(1:n_extremes);
dvdt_slow_extremes = dvdt_slow_idx(1:n_extremes);
dvdt_fast_extremes = dvdt_fast_idx(1:n_extremes);

% Convert to actual spike indices
vm_low_spikes = filtered_spike_indices(vm_low_extremes);
vm_high_spikes = filtered_spike_indices(vm_high_extremes);
dvdt_slow_spikes = filtered_spike_indices(dvdt_slow_extremes);
dvdt_fast_spikes = filtered_spike_indices(dvdt_fast_extremes);

% Extract traces for extremes
window_samples = round(vm_window / (dt * 1000));
time_ms = (-window_samples:window_samples-1) * dt * 1000;

% Define colors
clean_color = [0.2 0.8 0.2];  % Green for clean spikes
residual_color = [0.8 0.2 0.2];  % Red for residual spikes
low_marker_color = [0.6 0.2 0.8];  % Purple for low extremes
high_marker_color = [1.0 0.8 0.0];  % Yellow for high extremes

figure(extreme_fig);
clf;

% TOP ROW - VM EXTREMES
% Top Left: Lowest VM traces
subplot(2,3,1);
vm_low_traces = [];
for i = 1:length(vm_low_spikes)
    spike_idx = vm_low_spikes(i);
    start_idx = spike_idx - window_samples;
    end_idx = spike_idx + window_samples - 1;
    
    if start_idx > 0 && end_idx <= length(Vm_all)
        trace = Vm_all(start_idx:end_idx);
        vm_low_traces = [vm_low_traces; trace'];
        plot(time_ms, trace, 'Color', [clean_color, 0.3], 'LineWidth', 0.5);
        hold on;
    end
end
if ~isempty(vm_low_traces)
    avg_trace = mean(vm_low_traces, 1);
    plot(time_ms, avg_trace, 'Color', clean_color, 'LineWidth', 3, 'DisplayName', 'Average');
end
title(sprintf('Lowest Pre-spike VM (N=%d)', size(vm_low_traces, 1)), 'FontSize', 12);
xlabel('Time from spike (ms)');
ylabel('Vm (mV)');
grid on;
hold off;

% Top Middle: Highest VM traces  
subplot(2,3,2);
vm_high_traces = [];
for i = 1:length(vm_high_spikes)
    spike_idx = vm_high_spikes(i);
    start_idx = spike_idx - window_samples;
    end_idx = spike_idx + window_samples - 1;
    
    if start_idx > 0 && end_idx <= length(Vm_all)
        trace = Vm_all(start_idx:end_idx);
        vm_high_traces = [vm_high_traces; trace'];
        plot(time_ms, trace, 'Color', [residual_color, 0.3], 'LineWidth', 0.5);
        hold on;
    end
end
if ~isempty(vm_high_traces)
    avg_trace = mean(vm_high_traces, 1);
    plot(time_ms, avg_trace, 'Color', residual_color, 'LineWidth', 3, 'DisplayName', 'Average');
end
title(sprintf('Highest Pre-spike VM (N=%d)', size(vm_high_traces, 1)), 'FontSize', 12);
xlabel('Time from spike (ms)');
ylabel('Vm (mV)');
grid on;
hold off;

% Top Right: VM Scatter with marked extremes
subplot(2,3,3);
scatter(vm_values, factors.initiation_voltages, 20, 'k', 'filled'); % All points in gray
hold on;
% Mark extreme points
scatter(vm_values(vm_low_extremes), factors.initiation_voltages(vm_low_extremes), 60, low_marker_color, 'filled', 'MarkerEdgeColor', 'k');
scatter(vm_values(vm_high_extremes), factors.initiation_voltages(vm_high_extremes), 60, high_marker_color, 'filled', 'MarkerEdgeColor', 'k');
xlabel('Pre-spike Vm (mV)');
ylabel('Spike Initiation (mV)');
title('VM Scatter - Marked Extremes');
grid on;
legend({'All spikes', 'Lowest VM (purple)', 'Highest VM (yellow)'}, 'Location', 'best');
hold off;

% BOTTOM ROW - dV/dt EXTREMES
% Bottom Left: Slowest dV/dt traces
subplot(2,3,4);
dvdt_slow_traces = [];
for i = 1:length(dvdt_slow_spikes)
    spike_idx = dvdt_slow_spikes(i);
    start_idx = spike_idx - window_samples;
    end_idx = spike_idx + window_samples - 1;
    
    if start_idx > 0 && end_idx <= length(Vm_all)
        trace = Vm_all(start_idx:end_idx);
        dvdt_slow_traces = [dvdt_slow_traces; trace'];
        plot(time_ms, trace, 'Color', [clean_color, 0.3], 'LineWidth', 0.5);
        hold on;
    end
end
if ~isempty(dvdt_slow_traces)
    avg_trace = mean(dvdt_slow_traces, 1);
    plot(time_ms, avg_trace, 'Color', clean_color, 'LineWidth', 3, 'DisplayName', 'Average');
end
title(sprintf('Slowest dV/dt (N=%d)', size(dvdt_slow_traces, 1)), 'FontSize', 12);
xlabel('Time from spike (ms)');
ylabel('Vm (mV)');
grid on;
hold off;

% Bottom Middle: Fastest dV/dt traces  
subplot(2,3,5);
dvdt_fast_traces = [];
for i = 1:length(dvdt_fast_spikes)
    spike_idx = dvdt_fast_spikes(i);
    start_idx = spike_idx - window_samples;
    end_idx = spike_idx + window_samples - 1;
    
    if start_idx > 0 && end_idx <= length(Vm_all)
        trace = Vm_all(start_idx:end_idx);
        dvdt_fast_traces = [dvdt_fast_traces; trace'];
        plot(time_ms, trace, 'Color', [residual_color, 0.3], 'LineWidth', 0.5);
        hold on;
    end
end
if ~isempty(dvdt_fast_traces)
    avg_trace = mean(dvdt_fast_traces, 1);
    plot(time_ms, avg_trace, 'Color', residual_color, 'LineWidth', 3, 'DisplayName', 'Average');
end
title(sprintf('Fastest dV/dt (N=%d)', size(dvdt_fast_traces, 1)), 'FontSize', 12);
xlabel('Time from spike (ms)');
ylabel('Vm (mV)');
grid on;
hold off;

% Bottom Right: dV/dt Scatter with marked extremes
subplot(2,3,6);
scatter(dvdt_values, factors.initiation_voltages, 20, 'k', 'filled'); % All points in gray
hold on;
% Mark extreme points
scatter(dvdt_values(dvdt_slow_extremes), factors.initiation_voltages(dvdt_slow_extremes), 60, low_marker_color, 'filled', 'MarkerEdgeColor', 'k');
scatter(dvdt_values(dvdt_fast_extremes), factors.initiation_voltages(dvdt_fast_extremes), 60, high_marker_color, 'filled', 'MarkerEdgeColor', 'k');
xlabel('dV/dt (mV/ms)');
ylabel('Spike Initiation (mV)');
title('dV/dt Scatter - Marked Extremes');
grid on;
legend({'All spikes', 'Slowest dV/dt (purple)', 'Fastest dV/dt (yellow)'}, 'Location', 'best');
hold off;

% Add comprehensive title with current settings
cell_info = sprintf('Cell: %s | Frequency: %s Hz | Mode: %s | Extremes: %d each | Total Spikes: %d', ...
    shared_state.current_cell, shared_state.current_frequency, spike_mode, ...
    n_extremes, length(filtered_spike_indices));
sgtitle({cell_info, 'Extreme Groups Analysis - Top/Bottom Examples'}, 'FontSize', 12, 'FontWeight', 'bold');

end

function y_limits = calculateYLimits(Vm_all, spike_indices, pre_window_samples, post_window_samples)
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
end

function actualApplyChanges(control_data)
% Apply all current settings and update plots
try
    % Update plots with current settings
    updateAllPlotsFixedExamples(control_data.shared_state, control_data.fixed_examples);
    
    % Update extreme groups visualization if enabled
    if isfield(control_data.shared_state.gui_data, 'plot_extreme_groups') && control_data.shared_state.gui_data.plot_extreme_groups && ...
       isfield(control_data.shared_state, 'extreme_fig') && ishandle(control_data.shared_state.extreme_fig)
        updateExtremeGroupsVisualization(control_data.shared_state, control_data.shared_state.extreme_fig);
    end
    
    fprintf('Applied changes successfully\n');
catch ME
    fprintf('Error applying changes: %s\n', ME.message);
    rethrow(ME);
end
end

function refreshDebugCallback()
% Refresh the debug visualization with a new random spike-rich segment
control_figs = findall(0, 'Name', 'Four Factor Analysis - Controls (v3)');
if isempty(control_figs)
    fprintf('Error: No control figure found\n');
    return; 
end
control_data = get(control_figs(1), 'UserData');

% Check if debug visualization is enabled and exists
if ~control_data.shared_state.show_spike_vis || ~isfield(control_data.shared_state, 'vis_fig') || isempty(control_data.shared_state.vis_fig)
    fprintf('Debug visualization is not enabled or figure not found\n');
    return;
end

% Update the debug visualization (this will automatically select a new random segment)
if ~isempty(control_data.shared_state.current_data)
    updateSpikeVisualization(control_data.shared_state, control_data.shared_state.vis_fig);
    fprintf('Refreshed debug view with new spike-rich segment\n');
else
    fprintf('Error: No current data available for debug refresh\n');
end
end

function debouncedUpdateCallback()
% Debounced callback for updating plots - triggered after a delay when sliding stops

control_figs = findall(0, 'Name', 'Four Factor Analysis - Controls (v3)');
if isempty(control_figs), return; end
control_data = get(control_figs(1), 'UserData');

% Update state from controls
control_data.shared_state.vm_window = get(control_data.controls.vm_slider, 'Value');
control_data.shared_state.dvdt_window = get(control_data.controls.dvdt_slider, 'Value');
control_data.shared_state.count_window = get(control_data.controls.count_slider, 'Value');
control_data.shared_state.exclusion_window = get(control_data.controls.exclusion_slider, 'Value');

% Update UserData immediately
set(control_figs(1), 'UserData', control_data);

% Check if we're in manual apply mode
if control_data.shared_state.manual_apply_mode
    mode_str = 'MANUAL - changes pending';
    % Don't update plots, just mark as having pending changes
    control_data.shared_state.has_pending_changes = true;
    set(control_data.controls.apply_btn, 'Enable', 'on');
    set(control_data.controls.apply_btn, 'BackgroundColor', [0.2 0.8 0.2]);
    set(control_data.controls.apply_btn, 'String', 'Apply Changes *');
    set(control_figs(1), 'UserData', control_data);
else
    mode_str = 'AUTO - updating plots';
    % Auto mode: update plots immediately
    updateAllPlotsFixedExamples(control_data.shared_state, control_data.fixed_examples);
    
    % Update extreme groups visualization if enabled
    if isfield(control_data.shared_state.gui_data, 'plot_extreme_groups') && control_data.shared_state.gui_data.plot_extreme_groups && ...
       isfield(control_data.shared_state, 'extreme_fig') && ishandle(control_data.shared_state.extreme_fig)
        updateExtremeGroupsVisualization(control_data.shared_state, control_data.shared_state.extreme_fig);
    end
end

fprintf('Debounced update completed - Mode: %s\n', mode_str);
end

function applyPendingChanges(~)
% Apply pending changes in manual mode
control_figs = findall(0, 'Name', 'Four Factor Analysis - Controls (v3)');
if isempty(control_figs), return; end
control_data = get(control_figs(1), 'UserData');

if control_data.shared_state.manual_apply_mode && control_data.shared_state.has_pending_changes
    % Apply the changes
    actualApplyChanges(control_data);
    control_data.shared_state.has_pending_changes = false;
    
    % Update Apply button state
    set(control_data.controls.apply_btn, 'Enable', 'off');
    set(control_data.controls.apply_btn, 'BackgroundColor', [0.9 0.9 0.9]);
    set(control_data.controls.apply_btn, 'String', 'Apply Changes');
    
    set(control_figs(1), 'UserData', control_data);
    fprintf('Applied pending changes\n');
else
    fprintf('No pending changes to apply\n');
end
end

function toggleAxisLock(~)
% Toggle axis lock on/off
axisLockToggleCallback([]);
end

function updateFrequencyDropdown()
% Get data from control figure
control_figs = findall(0, 'Name', 'Four Factor Analysis - Controls (v3)');
if isempty(control_figs), return; end
control_data = get(control_figs(1), 'UserData');

% Update frequency dropdown for current cell
cell_field = matlab.lang.makeValidName(control_data.shared_state.current_cell);
if isfield(control_data.shared_state.gui_data.organized_data, cell_field)
    frequencies = control_data.shared_state.gui_data.organized_data.(cell_field).frequencies;
    
    % Try to preserve current frequency selection
    current_freq = control_data.shared_state.current_frequency;
    freq_index = 1; % Default to first frequency
    
    % Find the index of the current frequency if it exists in the new list
    if ~isempty(current_freq)
        freq_match = find(strcmp(frequencies, current_freq));
        if ~isempty(freq_match)
            freq_index = freq_match(1); % Use first match if multiple
        end
    end
    
    set(control_data.controls.freq_popup, 'String', frequencies);
    if ~isempty(frequencies)
        set(control_data.controls.freq_popup, 'Value', freq_index);
        frequencyChangedCallback(freq_index);
    end
end
end

function cleanupAndClose(shared_state, fig_handle)
% Cleanup function called when GUI figures are closed
% Closes both control panel and plot figures together and cleans up resources

fprintf('Closing Four Factor Analysis GUI...\n');

try
    % Stop and clean up the debounce timer
    if isfield(shared_state, 'update_timer') && isvalid(shared_state.update_timer)
        stop(shared_state.update_timer);
        delete(shared_state.update_timer);
        fprintf('Debounce timer cleaned up\n');
    end
catch ME
    fprintf('Warning: Error cleaning up timer: %s\n', ME.message);
end

try
    % Find both GUI figures
    plot_figs = findall(0, 'Name', 'Four Factor Analysis - Plots (v3)');
    control_figs = findall(0, 'Name', 'Four Factor Analysis - Controls (v3)');
    extreme_figs = findall(0, 'Name', 'Extreme Groups Analysis');
    
    % Close both figures (avoid closing the same figure twice)
    for i = 1:length(plot_figs)
        if ishandle(plot_figs(i)) && plot_figs(i) ~= fig_handle
            delete(plot_figs(i));
        end
    end
    
    for i = 1:length(control_figs)
        if ishandle(control_figs(i)) && control_figs(i) ~= fig_handle
            delete(control_figs(i));
        end
    end
    
    % Close extreme groups figure if it exists
    for i = 1:length(extreme_figs)
        if ishandle(extreme_figs(i)) && extreme_figs(i) ~= fig_handle
            delete(extreme_figs(i));
        end
    end
    
    % Finally close the triggering figure
    if ishandle(fig_handle)
        delete(fig_handle);
    end
    
    fprintf('All GUI figures closed\n');
catch ME
    fprintf('Warning: Error during figure cleanup: %s\n', ME.message);
    % Fallback: force close the triggering figure
    if ishandle(fig_handle)
        delete(fig_handle);
    end
end

% Clear any variables from base workspace if needed
try
    % Check if there are any GUI-related variables to clean up
    fprintf('GUI cleanup completed\n');
catch ME
    fprintf('Warning: Error during workspace cleanup: %s\n', ME.message);
end

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
spike_mode = shared_state.spike_mode;
exclusion_window = shared_state.exclusion_window;

% Extract basic data
Vm_all = current_data.Vm_all;
spike_indices = current_data.spike_indices;
dt = current_data.dt;

if length(spike_indices) < 2
    return; % Need at least 2 spikes for analysis
end

% Filter spikes based on mode
filtered_spike_indices = filterSpikes(spike_indices, spike_mode, exclusion_window, dt);

if length(filtered_spike_indices) < 2
    fprintf('Warning: Only %d spikes after filtering with mode "%s"\n', length(filtered_spike_indices), spike_mode);
    if isempty(filtered_spike_indices)
        return; % Can't analyze with no spikes
    end
end

% Calculate factors with current window settings (no randomization)
[factors, ~] = calculateFactors(Vm_all, filtered_spike_indices, dt, vm_window, dvdt_window, count_window, false);

% Use fixed examples if available, otherwise calculate new ones
if isempty(fixed_examples)
    [~, examples] = calculateFactors(Vm_all, filtered_spike_indices, dt, vm_window, dvdt_window, count_window, false);
else
    examples = fixed_examples;
end

% Make sure we're plotting in the plot figure
if isfield(shared_state, 'plot_fig') && ~isempty(shared_state.plot_fig) && isvalid(shared_state.plot_fig)
    figure(shared_state.plot_fig);
    
    % Add supertitle with comprehensive cell information including spike filtering info
    cell_info = sprintf('Cell: %s | Frequency: %s Hz | Mode: %s | Spikes: %d/%d (Total: %d) | Duration: %.1fs', ...
        shared_state.current_cell, shared_state.current_frequency, spike_mode, ...
        length(filtered_spike_indices), length(spike_indices), length(spike_indices), length(Vm_all) * dt);

    % Add exclusion window info for clean/residual modes
    if ~strcmp(spike_mode, 'All Spikes')
        cell_info = sprintf('%s | Exclusion: %.0fms', cell_info, exclusion_window);
    end

    % Try to extract additional metadata from current_data
    if isfield(current_data, 'protocol') && isfield(current_data, 'date')
        cell_info = sprintf('%s | Protocol: %s | Date: %s', cell_info, current_data.protocol, current_data.date);
    end

    sgtitle(cell_info, 'FontSize', 14, 'FontWeight', 'bold');

    % Update traces with fixed examples
    updateTraces(shared_state, Vm_all, filtered_spike_indices, dt, examples, vm_window, dvdt_window);

    % Update scatter plots
    updateScatterPlots(shared_state, factors, examples);
end

end

function axisLockToggleCallback(is_locked)
% Toggle axis lock on/off
control_figs = findall(0, 'Name', 'Four Factor Analysis - Controls (v3)');
if isempty(control_figs), return; end
control_data = get(control_figs(1), 'UserData');

control_data.shared_state.axis_locked = logical(is_locked);

if control_data.shared_state.axis_locked
    % Update button appearance for locked state
    set(control_data.controls.axis_lock_btn, 'String', 'Axis Lock: ON');
    set(control_data.controls.axis_lock_btn, 'BackgroundColor', [0.2 0.8 0.2]); % Green when locked
    
    % Lock the current axis limits from ALL scatter plots
    plot_figs = findall(0, 'Name', 'Four Factor Analysis - Plots (v3)');
    if ~isempty(plot_figs)
        figure(plot_figs(1));
        % Store current axis limits for all scatter plot subplots
        subplots = control_data.shared_state.subplots;
        control_data.shared_state.locked_axis_limits = struct();
        
        % Store VM scatter plot limits
        if isfield(subplots, 'vm_scatter') && ishandle(subplots.vm_scatter)
            axes(subplots.vm_scatter);
            control_data.shared_state.locked_axis_limits.vm_scatter_xlim = xlim();
            control_data.shared_state.locked_axis_limits.vm_scatter_ylim = ylim();
        end
        
        % Store ISI scatter plot limits
        if isfield(subplots, 'isi_scatter') && ishandle(subplots.isi_scatter)
            axes(subplots.isi_scatter);
            control_data.shared_state.locked_axis_limits.isi_scatter_xlim = xlim();
            control_data.shared_state.locked_axis_limits.isi_scatter_ylim = ylim();
        end
        
        % Store dV/dt scatter plot limits  
        if isfield(subplots, 'dvdt_scatter') && ishandle(subplots.dvdt_scatter)
            axes(subplots.dvdt_scatter);
            control_data.shared_state.locked_axis_limits.dvdt_scatter_xlim = xlim();
            control_data.shared_state.locked_axis_limits.dvdt_scatter_ylim = ylim();
        end
        
        % Store spike count scatter plot limits
        if isfield(subplots, 'count_scatter') && ishandle(subplots.count_scatter)
            axes(subplots.count_scatter);
            control_data.shared_state.locked_axis_limits.count_scatter_xlim = xlim();
            control_data.shared_state.locked_axis_limits.count_scatter_ylim = ylim();
        end
    end
    fprintf('Axis limits locked for all scatter plots\n');
else
    % Update button appearance for unlocked state
    set(control_data.controls.axis_lock_btn, 'String', 'Axis Lock: OFF');
    set(control_data.controls.axis_lock_btn, 'BackgroundColor', [0.9 0.9 0.9]); % Gray when unlocked
    
    % Clear locked limits
    control_data.shared_state.locked_axis_limits = struct();
    fprintf('Axis limits unlocked\n');
end

set(control_figs(1), 'UserData', control_data);
end

function exportAnalysisCallback()
% Export current analysis data and plots
control_figs = findall(0, 'Name', 'Four Factor Analysis - Controls (v3)');
if isempty(control_figs), return; end
control_data = get(control_figs(1), 'UserData');

if isempty(control_data.shared_state.current_data)
    fprintf('No data to export\n');
    return;
end

% Get current analysis parameters
current_data = control_data.shared_state.current_data;
cell_name = control_data.shared_state.current_cell;
frequency = control_data.shared_state.current_frequency;
spike_mode = control_data.shared_state.spike_mode;
vm_window = control_data.shared_state.vm_window;
dvdt_window = control_data.shared_state.dvdt_window;
count_window = control_data.shared_state.count_window;
exclusion_window = control_data.shared_state.exclusion_window;

% Create export data structure
export_data = struct();
export_data.cell_name = cell_name;
export_data.frequency = frequency;
export_data.spike_mode = spike_mode;
export_data.parameters = struct();
export_data.parameters.vm_window = vm_window;
export_data.parameters.dvdt_window = dvdt_window;
export_data.parameters.count_window = count_window;
export_data.parameters.exclusion_window = exclusion_window;
export_data.timestamp = datestr(now, 'yyyymmdd_HHMMSS');

% Calculate factors for export
Vm_all = current_data.Vm_all;
spike_indices = current_data.spike_indices;
dt = current_data.dt;
filtered_spike_indices = filterSpikes(spike_indices, spike_mode, exclusion_window, dt);

if length(filtered_spike_indices) >= 2
    [factors, examples] = calculateFactors(Vm_all, filtered_spike_indices, dt, vm_window, dvdt_window, count_window);
    export_data.factors = factors;
    export_data.examples = examples;
    export_data.spike_count = struct();
    export_data.spike_count.total = length(spike_indices);
    export_data.spike_count.filtered = length(filtered_spike_indices);
else
    fprintf('Warning: Not enough filtered spikes for export\n');
    export_data.factors = struct();
    export_data.examples = struct();
end

% Create filename
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
cell_name_safe = strrep(matlab.lang.makeValidName(cell_name), '\', '-');
frequency_safe = strrep(frequency, '\', '-');
spike_mode_safe = strrep(spike_mode, ' ', '');

% Use save path from gui_data
if isfield(control_data.shared_state.gui_data, 'save_path') && exist(control_data.shared_state.gui_data.save_path, 'dir')
    save_path = control_data.shared_state.gui_data.save_path;
else
    save_path = pwd;
end

% Save data file
data_filename = sprintf('%s_analysis_%sHz_%s_%s_data.mat', ...
    cell_name_safe, frequency_safe, spike_mode_safe, timestamp);
data_full_path = fullfile(save_path, data_filename);

try
    save(data_full_path, 'export_data');
    fprintf('Exported analysis data to: %s\n', data_full_path);
catch ME
    fprintf('Error saving data: %s\n', ME.message);
end

% Also save plots
saveJpgCallback();
end

function manualModeToggleCallback(new_value)
% Toggle manual apply mode on/off
control_figs = findall(0, 'Name', 'Four Factor Analysis - Controls (v3)');
if isempty(control_figs), return; end
control_data = get(control_figs(1), 'UserData');

control_data.shared_state.manual_apply_mode = logical(new_value);

% Update Apply button state
if control_data.shared_state.manual_apply_mode
    set(control_data.controls.apply_btn, 'Enable', 'on');
    set(control_data.controls.apply_btn, 'BackgroundColor', [0.2 0.8 0.2]);  % Green when enabled
    set(control_data.controls.apply_btn, 'String', 'Apply Changes *');  % Asterisk indicates pending
    fprintf('Manual apply mode enabled\n');
else
    set(control_data.controls.apply_btn, 'Enable', 'off');
    set(control_data.controls.apply_btn, 'BackgroundColor', [0.9 0.9 0.9]);
    set(control_data.controls.apply_btn, 'String', 'Apply Changes');
    fprintf('Manual apply mode disabled\n');
    
    % Auto-apply: immediately apply any pending changes
    if control_data.shared_state.has_pending_changes
        actualApplyChanges(control_data);
        control_data.shared_state.has_pending_changes = false;
    end
end

% Update UserData with new state
set(control_figs(1), 'UserData', control_data);
end
