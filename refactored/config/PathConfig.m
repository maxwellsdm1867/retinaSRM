classdef PathConfig
    % PATHCONFIG - Configuration class for file paths and directories
    % This class manages all file paths, directories, and path-related
    % operations for the analysis pipeline.
    %
    % Properties:
    %   dataFolder - Base data directory
    %   exportFolder - Export directory for results
    %   logPath - Directory for log files
    %   figurePath - Directory for generated figures
    %   analysisPath - Directory for analysis results
    %
    % Author: Maxwell
    % Date: 2024
    
    properties
        dataFolder = '/Users/maxwellsdm/Library/CloudStorage/GoogleDrive-maxwellsdm1867@gmail.com/.shortcut-targets-by-id/1CbZnfugdi-p4jM3fe7t-NyGk5v2ZcPdk/ParasolCenterSurround/Data/'
        exportFolder = '/Users/maxwellsdm/Library/CloudStorage/GoogleDrive-maxwellsdm1867@gmail.com/.shortcut-targets-by-id/1CbZnfugdi-p4jM3fe7t-NyGk5v2ZcPdk/ParasolCenterSurround/Data/'
        logPath = '/Users/maxwellsdm/Library/CloudStorage/GoogleDrive-maxwellsdm1867@gmail.com/.shortcut-targets-by-id/1CbZnfugdi-p4jM3fe7t-NyGk5v2ZcPdk/ParasolCenterSurround/Analysis/SodiumDynamics/'
        figurePath = '/Users/maxwellsdm/Library/CloudStorage/GoogleDrive-maxwellsdm1867@gmail.com/.shortcut-targets-by-id/1CbZnfugdi-p4jM3fe7t-NyGk5v2ZcPdk/ParasolCenterSurround/figures/ScienceJuiceFactory/currinjt/overview/'
        analysisPath = '/Users/maxwellsdm/Library/CloudStorage/GoogleDrive-maxwellsdm1867@gmail.com/.shortcut-targets-by-id/1CbZnfugdi-p4jM3fe7t-NyGk5v2ZcPdk/ParasolCenterSurround/Analysis/'
    end
    
    methods
        function obj = PathConfig()
            % PATHCONFIG Constructor
            % Creates a new PathConfig instance and ensures all directories exist
            obj.ensureDirectoriesExist();
        end
        
        function ensureDirectoriesExist(obj)
            % ENSUREDIRECTORIESEXIST - Create directories if they don't exist
            % Creates all necessary directories for the analysis pipeline
            
            directories = {obj.logPath, obj.figurePath, obj.analysisPath};
            
            for i = 1:length(directories)
                dir_path = directories{i};
                if ~exist(dir_path, 'dir')
                    mkdir(dir_path);
                    fprintf('Created directory: %s\n', dir_path);
                end
            end
        end
        
        function full_path = getDataFilePath(obj, filename)
            % GETDATAFILEPATH - Get full path for data file
            % Input:
            %   filename - Name of the data file
            % Output:
            %   full_path - Complete path to the data file
            full_path = fullfile(obj.dataFolder, filename);
        end
        
        function full_path = getFigurePath(obj, cellType, filename)
            % GETFIGUREPATH - Get full path for figure file
            % Input:
            %   cellType - Cell type for subdirectory organization
            %   filename - Name of the figure file
            % Output:
            %   full_path - Complete path to the figure file
            
            % Clean cell type for path
            cell_type_clean = strrep(cellType, '\', '/');
            cell_type_clean = strrep(cell_type_clean, ' ', '_');
            
            % Create subdirectory
            subdir = fullfile(obj.figurePath, cell_type_clean);
            if ~exist(subdir, 'dir')
                mkdir(subdir);
            end
            
            full_path = fullfile(subdir, filename);
        end
        
        function full_path = getLogFilePath(obj, filename)
            % GETLOGFILEPATH - Get full path for log file
            % Input:
            %   filename - Name of the log file
            % Output:
            %   full_path - Complete path to the log file
            full_path = fullfile(obj.logPath, filename);
        end
        
        function full_path = getAnalysisFilePath(obj, filename)
            % GETANALYSISFILEPATH - Get full path for analysis file
            % Input:
            %   filename - Name of the analysis file
            % Output:
            %   full_path - Complete path to the analysis file
            full_path = fullfile(obj.analysisPath, filename);
        end
        
        function validate(obj)
            % VALIDATE - Validate path configuration
            % Checks that all paths are accessible and valid
            % Throws error if validation fails
            
            paths_to_check = {obj.dataFolder, obj.exportFolder, obj.logPath, obj.figurePath, obj.analysisPath};
            
            for i = 1:length(paths_to_check)
                path_to_check = paths_to_check{i};
                if ~exist(path_to_check, 'dir')
                    error('Directory does not exist: %s', path_to_check);
                end
                
                % Check write permissions for output directories
                if i >= 3  % logPath, figurePath, analysisPath
                    test_file = fullfile(path_to_check, '.test_write');
                    try
                        fid = fopen(test_file, 'w');
                        if fid == -1
                            error('No write permission for directory: %s', path_to_check);
                        end
                        fclose(fid);
                        delete(test_file);
                    catch ME
                        error('Cannot write to directory %s: %s', path_to_check, ME.message);
                    end
                end
            end
        end
        
        function display(obj)
            % DISPLAY - Display path configuration
            % Shows all configured paths in a formatted way
            fprintf('\n=== PATH CONFIGURATION ===\n');
            fprintf('Data Folder: %s\n', obj.dataFolder);
            fprintf('Export Folder: %s\n', obj.exportFolder);
            fprintf('Log Path: %s\n', obj.logPath);
            fprintf('Figure Path: %s\n', obj.figurePath);
            fprintf('Analysis Path: %s\n', obj.analysisPath);
            fprintf('==========================\n\n');
        end
    end
end 