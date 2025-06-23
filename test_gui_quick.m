% Quick test to check if the GUI launches without errors
fprintf('=== QUICK GUI TEST ===\n');

% Create minimal test data structure 
gui_data = struct();
gui_data.cell_names = {'Cell1-2024-01-01', 'Cell2-2024-01-02'};
gui_data.organized_data = struct();
gui_data.plot_extreme_groups = true;
gui_data.cleanSpikeVis = false;
gui_data.save_path = pwd;

% Create some test data for each cell
for i = 1:length(gui_data.cell_names)
    cell_name = gui_data.cell_names{i};
    cell_field = matlab.lang.makeValidName(cell_name);
    
    % Create the data structure
    gui_data.organized_data.(cell_field) = struct();
    gui_data.organized_data.(cell_field).frequencies = {'10', '50', '100', '200'};
    gui_data.organized_data.(cell_field).data = containers.Map();
    
    % Add test data for each frequency
    for j = 1:length(gui_data.organized_data.(cell_field).frequencies)
        freq_str = gui_data.organized_data.(cell_field).frequencies{j};
        
        % Create dummy data entry
        entry = struct();
        entry.cell_id = cell_name;
        entry.frequency = freq_str;
        entry.Vm_all = randn(10000, 1) * 5 - 60; % Random voltage trace
        entry.elbow_indices = sort(randperm(10000, 50)); % Random spike indices
        entry.dt = 0.0001; % 0.1 ms sampling
        entry.duration_s = 1.0;
        entry.n_spikes = 50;
        entry.firing_rate_Hz = 50;
        
        gui_data.organized_data.(cell_field).data(freq_str) = entry;
    end
end

fprintf('Test data created with %d cells\n', length(gui_data.cell_names));
fprintf('Attempting to launch GUI...\n');

% Try to launch the GUI
try
    fourFactorGuiSeparated3(gui_data);
    fprintf('✓ GUI launched successfully!\n');
catch ME
    fprintf('✗ GUI launch failed: %s\n', ME.message);
    fprintf('Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('  File: %s, Line: %d, Function: %s\n', ...
            ME.stack(i).file, ME.stack(i).line, ME.stack(i).name);
    end
end
