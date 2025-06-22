%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SODIUM CHANNEL DYNAMICS - DATA EXTRACTION AND ANALYSIS (NO PLOTTING)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%*************************************************************************
% Initialization
%*************************************************************************
clear; clc;

% jauimodel setup
loader = edu.washington.rieke.Analysis.getEntityLoader();
treeFactory = edu.washington.rieke.Analysis.getEpochTreeFactory();

% Data paths (adjust as needed)
dataFolder = '/Users/maxwellsdm/Library/CloudStorage/GoogleDrive-maxwellsdm1867@gmail.com/.shortcut-targets-by-id/1CbZnfugdi-p4jM3fe7t-NyGk5v2ZcPdk/ParasolCenterSurround/Data/';
exportFolder = dataFolder;

import auimodel.*
import vuidocument.*

% Analysis parameters
params = struct();
params.Amp = 'Amp1';
params.SamplingInterval = 0.0001;  % 10 kHz sampling
params.Verbose = 1;

% Log storage configuration
log_config = struct();
log_config.save_path = '/Users/maxwellsdm/Library/CloudStorage/GoogleDrive-maxwellsdm1867@gmail.com/.shortcut-targets-by-id/1CbZnfugdi-p4jM3fe7t-NyGk5v2ZcPdk/ParasolCenterSurround/Analysis/SodiumDynamics/';
log_config.create_plots = false;   % Disable plotting
log_config.verbose_logging = true; % Detailed logging

% Create log directory if it doesn't exist
if ~exist(log_config.save_path, 'dir')
    mkdir(log_config.save_path);
    fprintf('Created log directory: %s\n', log_config.save_path);
end

% Spike detection parameters
spike_params = struct();
spike_params.vm_thresh = -20;       % Voltage threshold (mV)
spike_params.d2v_thresh = 50;       % Second derivative threshold
spike_params.search_back_ms = 2;    % Search back window (ms)
spike_params.plot_flag = false;     % Suppress plots during batch processing

% Voltage preprocessing parameters
filter_params = struct();
filter_params.cutoff_freq_Hz = 90;  % Low-pass cutoff for cleaned voltage

fprintf('=== SODIUM CHANNEL DYNAMICS ANALYSIS INITIALIZED ===\n');

% Load epoch list (modify filename as needed)
list = loader.loadEpochList([exportFolder 'currinjt250503.mat'], dataFolder);

% Build tree structure
dateSplit = @(list)splitOnExperimentDate(list);
dateSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(list, dateSplit);

tree = riekesuite.analysis.buildTree(list, {
    'protocolSettings(source:type)',        % Level 1: Cell Type ← This is what we need!
    dateSplit_java,                         % Level 2: Date
    'cell.label',                          % Level 3: Cell (individual cell)
    'protocolSettings(epochGroup:label)',   % Level 4: Group
    'protocolSettings(frequencyCutoff)',    % Level 5: Frequency
    'protocolSettings(currentSD)'           % Level 6: Current SD
    });

% Launch GUI for selection
gui = epochTreeGUI(tree);

fprintf('Please select the top-level node in the GUI, then run the next section\n');
fprintf('GUI is now open for node selection...\n');
