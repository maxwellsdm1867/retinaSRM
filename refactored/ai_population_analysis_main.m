%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SODIUM CHANNEL DYNAMICS - MAIN ANALYSIS ORCHESTRATOR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script coordinates the analysis of sodium channel dynamics across
% multiple cells, frequencies, and current injection levels. It serves as
% the main entry point for the analysis pipeline.
%
% The script follows these main steps:
% 1. Initialize environment and load configurations
% 2. Set up data structures and logging
% 3. Load and process epoch data
% 4. Navigate through data hierarchy
% 5. Perform analysis at each level
% 6. Generate reports and visualizations
%
% Dependencies:
% - config/AnalysisConfig.m
% - config/PathConfig.m
% - config/SpikeConfig.m
% - core/TreeNavigator.m
% - core/DataExtractor.m
% - core/AnalysisExecutor.m
% - analysis/FrequencyAnalyzer.m
% - reporting/AnalysisReporter.m
% - reporting/DebugReporter.m
% - reporting/StatisticsReporter.m
%
% Author: Maxwell
% Date: 2024
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialize Environment
close all; clear; clc;

% Add current directory and subdirectories to MATLAB path
current_dir = fileparts(mfilename('fullpath'));
addpath(genpath(current_dir));

% Explicitly add core directory to ensure it's in the path
core_dir = fullfile(current_dir, 'core');
addpath(core_dir);

% Import required packages for riekesuite functionality
import auimodel.*
import vuidocument.*

fprintf('=== SODIUM CHANNEL DYNAMICS ANALYSIS INITIALIZED ===\n');
fprintf('Current directory: %s\n', current_dir);
fprintf('Core directory: %s\n', core_dir);

%% Load Configurations
try
    % Load analysis configuration
    analysisConfig = AnalysisConfig();
    analysisConfig.validate();
    analysisConfig.display();
    
    % Load path configuration
    pathConfig = PathConfig();
    pathConfig.validate();
    pathConfig.display();
    
    % Load spike detection configuration
    spikeConfig = SpikeConfig();
    spikeConfig.validate();
    spikeConfig.display();
    
catch ME
    error('Failed to load configurations: %s', ME.message);
end

%% Initialize Data Structures
try
    % Initialize global logging
    global analysis_log log_counter;
    analysis_log = {};
    log_counter = 1;
    
    % Master log entry
    master_log = struct();
    master_log.analysis_type = 'SodiumChannelDynamics';
    master_log.timestamp = datetime('now');
    master_log.purpose = 'Elementary analysis on highest epoch count SD levels';
    master_log.selection_strategy = 'Most epochs within frequency groups';
    analysis_log{log_counter} = master_log;
    log_counter = log_counter + 1;
    
    % Storage for analysis results
    analysis_results = struct();
    analysis_results.successful = {};
    analysis_results.failed = {};
    analysis_results.metadata = struct();
    
catch ME
    error('Failed to initialize data structures: %s', ME.message);
end

%% Load and Process Epoch Data
try
    % jauimodel setup
    loader = edu.washington.rieke.Analysis.getEntityLoader();
    treeFactory = edu.washington.rieke.Analysis.getEpochTreeFactory();
    
    % Load epoch list
    list = loader.loadEpochList([pathConfig.dataFolder 'currinjt250503.mat'], pathConfig.dataFolder);
    
    % Build tree structure
    dateSplit = @(list)splitOnExperimentDate(list);
    dateSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(list, dateSplit);
    
    tree = riekesuite.analysis.buildTree(list, {
        'protocolSettings(source:type)',        % Level 1: Cell Type
        dateSplit_java,                         % Level 2: Date
        'cell.label',                          % Level 3: Cell (individual cell)
        'protocolSettings(epochGroup:label)',   % Level 4: Group
        'protocolSettings(frequencyCutoff)',    % Level 5: Frequency
        'protocolSettings(currentSD)'           % Level 6: Current SD
    });
    
    % Launch GUI for selection
    gui = epochTreeGUI(tree);
    fprintf('Please select the top-level node in the GUI, then run the next section\n');
    
catch ME
    error('Failed to load epoch data: %s', ME.message);
end

%% Tree Navigation and Analysis
try
    fprintf('\n=== STARTING TREE NAVIGATION AND ANALYSIS ===\n');
    
    % Get the selected node from GUI
    selectedNodes = getSelectedEpochTreeNodes(gui);
    if length(selectedNodes) ~= 1
        error('Please select exactly one top-level node in the GUI');
    end
    
    CurrentNode = selectedNodes{1};
    fprintf('Selected node: %s\n', CurrentNode.splitValue);
    fprintf('Number of children: %d\n', CurrentNode.children.length);
    
    % Create tree navigator
    treeNavigator = TreeNavigator(CurrentNode);
    treeNavigator.displayTreeStructure();
    
    % Get all frequency nodes
    [frequencyNodes, metadata] = treeNavigator.getAllFrequencyNodes();
    analysis_results.metadata = metadata;
    
    % Create analysis executor
    analysisExecutor = AnalysisExecutor(analysisConfig, spikeConfig, true);
    
    % Create frequency analyzer
    frequencyAnalyzer = FrequencyAnalyzer(true);
    
    % Process each frequency node
    fprintf('\n=== PROCESSING FREQUENCY NODES ===\n');
    for i = 1:length(frequencyNodes)
        frequencyNode = frequencyNodes(i);
        
        try
            fprintf('\nProcessing frequency node %d/%d: %s Hz\n', ...
                i, length(frequencyNodes), frequencyNode.frequency);
            
            % DEBUG: Test basic data extraction first
            fprintf('  Testing basic data extraction...\n');
            selected_sd = frequencyNode.sd_levels(1); % Use first SD level for testing
            if ~isempty(selected_sd)
                fprintf('    Selected SD level: %s with %d epochs\n', ...
                    selected_sd.value, selected_sd.n_epochs);
                
                % Test if we can access the node
                test_node = selected_sd.node;
                fprintf('    Node splitValue: %s\n', test_node.splitValue);
                fprintf('    Node has %d epochs\n', test_node.epochList.length);
                
                % Test if getSelectedData function exists
                if exist('getSelectedData', 'file') || exist('getSelectedData', 'builtin')
                    fprintf('    getSelectedData function is available\n');
                else
                    fprintf('    ERROR: getSelectedData function not found\n');
                end
                
                % Test if getNoiseStm function exists
                if exist('getNoiseStm', 'file') || exist('getNoiseStm', 'builtin')
                    fprintf('    getNoiseStm function is available\n');
                else
                    fprintf('    ERROR: getNoiseStm function not found\n');
                end
            else
                fprintf('    No SD levels found for this frequency\n');
            end
            
            % Perform basic analysis
            result = analysisExecutor.analyzeFrequencyNode(frequencyNode);
            
            % Perform frequency-specific analysis
            frequency_analysis = frequencyAnalyzer.analyzeFrequencyResponse(frequencyNode, result);
            
            % Store results
            if result.success
                analysis_results.successful{end+1} = result;
                fprintf('✓ Analysis successful for %s Hz\n', frequencyNode.frequency);
            else
                analysis_results.failed{end+1} = struct('node', frequencyNode, 'error', result.error);
                fprintf('✗ Analysis failed for %s Hz: %s\n', frequencyNode.frequency, result.error.message);
            end
            
        catch ME
            fprintf('✗ Error processing frequency node %d: %s\n', i, ME.message);
            analysis_results.failed{end+1} = struct('node', frequencyNode, 'error', ME);
        end
    end
    
catch ME
    error('Failed during tree navigation and analysis: %s', ME.message);
end

%% Generate Reports
try
    fprintf('\n=== GENERATING REPORTS ===\n');
    
    % Create reporters
    analysisReporter = AnalysisReporter(analysis_results, true);
    debugReporter = DebugReporter(true);
    statsReporter = StatisticsReporter(analysis_results, true);
    
    % Generate reports
    analysisReporter.generateSummary();
    debugReporter.generateDebugLog();
    statsReporter.generateStatistics();
    
catch ME
    warning(ME.identifier, 'Failed to generate reports: %s', ME.message);
end

%% Final Summary
fprintf('\n=== ANALYSIS COMPLETE ===\n');
fprintf('Total frequency nodes processed: %d\n', length(frequencyNodes));
fprintf('Successful analyses: %d\n', length(analysis_results.successful));
fprintf('Failed analyses: %d\n', length(analysis_results.failed));
fprintf('Success rate: %.1f%%\n', ...
    length(analysis_results.successful) / length(frequencyNodes) * 100);

fprintf('\nResults stored in: analysis_results\n');
fprintf('Use the reporting classes to generate detailed reports.\n');

%% Helper Functions

function dateSplit = splitOnExperimentDate(list)
    % SPLITONEXPERIMENTDATE - Split epoch list by experiment date
    % This function extracts the date from each epoch and groups them accordingly
    
    dates = {};
    for i = 1:list.length
        % Use the correct method to access epochs from the list
        epoch = list.get(i-1);  % Java lists are 0-indexed
        date = char(epoch.startDate);
        dates{end+1} = date(1:11); % Extract first 11 characters (date part)
    end
    
    dateSplit = dates;
end 