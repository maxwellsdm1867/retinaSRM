classdef DebugUtils
    % DebugUtils - Utility class for debug logging functions
    %
    % This class provides static methods for debug logging and tracking throughout
    % the analysis process. It maintains the exact same debug logging functionality
    % as the original code, but organizes it into reusable functions.
    %
    % The debug utilities handle:
    % - Initialization of global debug log
    % - Tracking of indexing operations
    % - Recording success/failure of operations
    % - Storing detailed debug messages
    % - Variable checking and logging
    % - Debug log export and clipboard operations
    %
    % Example Usage:
    %   % Initialize debug logging:
    %   DebugUtils.initializeDebugLog();
    %
    %   % Log debug information:
    %   DebugUtils.debugLogIndexing(cell_name, freq_str, sd_str, 'SD_selection', true, ...
    %       sprintf('Selected SD %s from %d options with %d epochs', ...
    %       sd_str, length(sd_values), max_epochs));
    
    methods(Static)
        function initializeDebugLog()
            % INITIALIZEDEBUGLOG - Initialize global debug log
            %
            % This function initializes the global debug logging system with the
            % exact same structure as the original code.
            %
            % The function:
            % 1. Creates an empty global DEBUG_LOG cell array
            % 2. Initializes the global DEBUG_LOG_COUNTER to 1
            % 3. Prepares the system for debug logging
            
            global DEBUG_LOG;
            global DEBUG_LOG_COUNTER;
            DEBUG_LOG = {};
            DEBUG_LOG_COUNTER = 1;
        end
        
        function debugLogIndexing(cell_name, freq_str, sd_str, stage, success, message)
            % DEBUGLOGINDEXING - Log debug information about indexing operations
            %
            % This function creates debug log entries with the exact same structure
            % and information as the original code's debug logging system.
            %
            % Inputs:
            %   cell_name - Name of the cell being processed
            %   freq_str - Frequency value as string
            %   sd_str - SD value as string
            %   stage - Current processing stage (e.g., 'SD_selection')
            %   success - Boolean indicating operation success
            %   message - Detailed debug message
            %
            % The function creates a debug entry containing:
            %   - timestamp: When the operation occurred
            %   - cell: Cell identifier
            %   - frequency: Frequency being processed
            %   - sd: SD value being processed
            %   - stage: Current operation stage
            %   - success: Whether the operation succeeded
            %   - message: Detailed debug information
            %
            % The entry is stored in the global DEBUG_LOG array using the same
            % structure as the original code.
            
            global DEBUG_LOG;
            global DEBUG_LOG_COUNTER;
            
            % Create debug entry with same structure as original
            debug_entry = struct();
            debug_entry.timestamp = datetime('now');
            debug_entry.cell = cell_name;
            debug_entry.frequency = freq_str;
            debug_entry.sd = sd_str;
            debug_entry.stage = stage;
            debug_entry.success = success;
            debug_entry.message = message;
            
            % Store in global log using same indexing as original
            DEBUG_LOG{DEBUG_LOG_COUNTER} = debug_entry;
            DEBUG_LOG_COUNTER = DEBUG_LOG_COUNTER + 1;
        end
        
        function debugLog(level, category, cell_id, freq_str, sd_str, message, varargin)
            % DEBUGLOG - Log a debug message with specified level and category
            %
            % Inputs:
            %   level - Debug level (e.g., 'INFO', 'WARNING', 'ERROR')
            %   category - Category of the debug message
            %   cell_id - Cell identifier
            %   freq_str - Frequency value as string
            %   sd_str - SD value as string
            %   message - Debug message
            %   varargin - Additional details (optional)
            
            global DEBUG_LOG;
            global DEBUG_LOG_COUNTER;
            
            debug_entry = struct();
            debug_entry.timestamp = datetime('now');
            debug_entry.level = level;
            debug_entry.category = category;
            debug_entry.cell = cell_id;
            debug_entry.frequency = freq_str;
            debug_entry.sd = sd_str;
            debug_entry.message = message;
            
            if ~isempty(varargin)
                debug_entry.details = varargin{1};
            end
            
            DEBUG_LOG{DEBUG_LOG_COUNTER} = debug_entry;
            DEBUG_LOG_COUNTER = DEBUG_LOG_COUNTER + 1;
        end
        
        function debugLogEntry(cell_id, freq_str, sd_str, n_epochs)
            % DEBUGLOGENTRY - Log entry point of analysis for a specific configuration
            DebugUtils.debugLog('INFO', 'ENTRY', cell_id, freq_str, sd_str, ...
                sprintf('Starting analysis with %d epochs', n_epochs));
        end
        
        function debugLogVariable(cell_id, freq_str, sd_str, var_name, exists, details)
            % DEBUGLOGVARIABLE - Log variable existence and details
            status = exists;
            if isnumeric(exists)
                status = exists ~= 0;
            end
            
            DebugUtils.debugLog('INFO', 'VARIABLE', cell_id, freq_str, sd_str, ...
                sprintf('Variable %s: %s', var_name, mat2str(status)), details);
        end
        
        function debugLogResult(cell_id, freq_str, sd_str, success, vp_train, vp_holdout)
            % DEBUGLOGRESULT - Log analysis results including VP distances
            if success
                DebugUtils.debugLog('SUCCESS', 'RESULT', cell_id, freq_str, sd_str, ...
                    sprintf('VP Train: %.3f, VP Holdout: %.3f', vp_train, vp_holdout));
            else
                DebugUtils.debugLog('ERROR', 'RESULT', cell_id, freq_str, sd_str, ...
                    'Analysis failed');
            end
        end
        
        function [exists, value, details] = checkAndLogVariable(stored_results, var_name, cell_id, freq_str, sd_str)
            % CHECKANDLOGVARIABLE - Check variable existence and log result
            exists = false;
            value = [];
            details = '';
            
            if isfield(stored_results, var_name)
                exists = true;
                value = stored_results.(var_name);
                if isnumeric(value)
                    details = sprintf('Value: %s', mat2str(value));
                elseif ischar(value)
                    details = sprintf('Value: %s', value);
                else
                    details = sprintf('Type: %s', class(value));
                end
            end
            
            DebugUtils.debugLogVariable(cell_id, freq_str, sd_str, var_name, exists, details);
        end
        
        function displayDebugSummary()
            % DISPLAYDEBUGSUMMARY - Display summary of debug log
            global DEBUG_LOG;
            
            if isempty(DEBUG_LOG)
                fprintf('Debug log is empty\n');
                return;
            end
            
            % Count entries by level
            levels = {};
            level_counts = [];
            
            for i = 1:length(DEBUG_LOG)
                entry = DEBUG_LOG{i};
                if isfield(entry, 'level')
                    level = entry.level;
                    idx = find(strcmp(levels, level));
                    if isempty(idx)
                        levels{end+1} = level;
                        level_counts(end+1) = 1;
                    else
                        level_counts(idx) = level_counts(idx) + 1;
                    end
                end
            end
            
            % Display summary
            fprintf('\n=== DEBUG LOG SUMMARY ===\n');
            fprintf('Total entries: %d\n', length(DEBUG_LOG));
            for i = 1:length(levels)
                fprintf('%s: %d entries\n', levels{i}, level_counts(i));
            end
        end
        
        function copyableLog = exportDebugLogForSharing()
            % EXPORTDEBUGLOGFORSHARING - Create shareable version of debug log
            global DEBUG_LOG;
            
            if isempty(DEBUG_LOG)
                copyableLog = 'No debug log entries found.';
                fprintf('No debug log entries to copy.\n');
                return;
            end
            
            copyableLog = sprintf('=== DEBUG LOG EXPORT ===\n');
            copyableLog = [copyableLog sprintf('Generated: %s\n', char(datetime('now')))];
            copyableLog = [copyableLog sprintf('Total entries: %d\n\n', length(DEBUG_LOG))];

            for i = 1:length(DEBUG_LOG)
                entry = DEBUG_LOG{i};
                
                % Safely get field values with fallbacks
                if isfield(entry, 'timestamp')
                    timestamp = char(entry.timestamp);
                else
                    timestamp = 'UNKNOWN_TIME';
                end
                
                if isfield(entry, 'level')
                    level = entry.level;
                else
                    level = 'UNKNOWN';
                end
                
                if isfield(entry, 'category')
                    category = entry.category;
                else
                    category = 'UNKNOWN';
                end
                
                if isfield(entry, 'cell')
                    cell_id = entry.cell;
                elseif isfield(entry, 'cell_id')
                    cell_id = entry.cell_id;
                else
                    cell_id = 'UNKNOWN';
                end
                
                if isfield(entry, 'frequency')
                    freq_str = entry.frequency;
                else
                    freq_str = 'UNKNOWN';
                end
                
                if isfield(entry, 'sd')
                    sd_str = entry.sd;
                else
                    sd_str = 'UNKNOWN';
                end
                
                if isfield(entry, 'message')
                    message = entry.message;
                else
                    message = 'No message';
                end

                line = sprintf('[%s] %s-%s %s|freq=%s|SD=%s: %s\n', ...
                    timestamp, level, category, cell_id, freq_str, sd_str, message);

                copyableLog = [copyableLog line];

                % Add data if present (handle both 'data' and 'details' fields)
                if isfield(entry, 'data') && ~isempty(fieldnames(entry.data))
                    data_fields = fieldnames(entry.data);
                    for j = 1:length(data_fields)
                        field = data_fields{j};
                        value = entry.data.(field);
                        if isnumeric(value)
                            line = sprintf('    %s: %g\n', field, value);
                        else
                            line = sprintf('    %s: %s\n', field, char(value));
                        end
                        copyableLog = [copyableLog line];
                    end
                elseif isfield(entry, 'details')
                    line = sprintf('    details: %s\n', char(entry.details));
                    copyableLog = [copyableLog line];
                end
            end

            copyableLog = [copyableLog sprintf('\n=== END DEBUG LOG ===\n')];

            % Copy to clipboard with robust error handling
            fprintf('Attempting to copy debug log to clipboard...\n');
            try
                if ispc
                    % Windows
                    clipboard('copy', copyableLog);
                    fprintf('✓ DEBUG LOG COPIED TO CLIPBOARD (Windows)!\n');
                elseif ismac
                    % macOS
                    clipboard('copy', copyableLog);
                    fprintf('✓ DEBUG LOG COPIED TO CLIPBOARD (macOS)!\n');
                else
                    % Linux - try different methods
                    try
                        clipboard('copy', copyableLog);
                        fprintf('✓ DEBUG LOG COPIED TO CLIPBOARD (Linux)!\n');
                    catch
                        % Fallback for Linux systems without clipboard support
                        fprintf('⚠ Clipboard copy not available on this system.\n');
                        fprintf('📋 DEBUG LOG READY - Copy the text below:\n');
                        fprintf('==========================================\n');
                        fprintf('%s', copyableLog);
                        fprintf('==========================================\n');
                    end
                end
            catch ME
                fprintf('⚠ Clipboard copy failed: %s\n', ME.message);
                fprintf('📋 DEBUG LOG READY - Copy the text below:\n');
                fprintf('==========================================\n');
                fprintf('%s', copyableLog);
                fprintf('==========================================\n');
            end
        end
        
        function copyDebugToClipboard()
            % COPYDEBUGTOCLIPBOARD - Copy debug log to clipboard
            global DEBUG_LOG;

            if isempty(DEBUG_LOG)
                fprintf('No debug log entries to copy.\n');
                return;
            end

            % Create simplified log for clipboard
            clipboardText = sprintf('MATLAB DEBUG LOG - %s\n', char(datetime('now')));
            clipboardText = [clipboardText sprintf('Entries: %d\n', length(DEBUG_LOG))];
            clipboardText = [clipboardText sprintf('====================\n')];

            for i = 1:length(DEBUG_LOG)
                entry = DEBUG_LOG{i};
                
                % Safely get field values with fallbacks
                if isfield(entry, 'timestamp')
                    timestamp = char(entry.timestamp);
                else
                    timestamp = 'UNKNOWN_TIME';
                end
                
                if isfield(entry, 'level')
                    level = entry.level;
                else
                    level = 'UNKNOWN';
                end
                
                if isfield(entry, 'category')
                    category = entry.category;
                else
                    category = 'UNKNOWN';
                end
                
                if isfield(entry, 'cell')
                    cell_id = entry.cell;
                elseif isfield(entry, 'cell_id')
                    cell_id = entry.cell_id;
                else
                    cell_id = 'UNKNOWN';
                end
                
                if isfield(entry, 'frequency')
                    freq_str = entry.frequency;
                else
                    freq_str = 'UNKNOWN';
                end
                
                if isfield(entry, 'sd')
                    sd_str = entry.sd;
                else
                    sd_str = 'UNKNOWN';
                end
                
                if isfield(entry, 'message')
                    message = entry.message;
                else
                    message = 'No message';
                end
                
                line = sprintf('%s [%s-%s] %s|%s|%s: %s\n', ...
                    timestamp, level, category, cell_id, freq_str, sd_str, message);
                clipboardText = [clipboardText line];
            end

            clipboardText = [clipboardText sprintf('====================\n')];

            % Copy to clipboard
            try
                clipboard('copy', clipboardText);
                fprintf('✓ Debug log copied to clipboard! (%d entries)\n', length(DEBUG_LOG));
            catch
                fprintf('⚠ Clipboard copy failed. Use exportDebugLogForSharing() instead.\n');
            end
        end
        
        function clearDebugLog()
            % CLEARDEBUGLOG - Clear the debug log
            global DEBUG_LOG;
            global DEBUG_LOG_COUNTER;
            DEBUG_LOG = {};
            DEBUG_LOG_COUNTER = 1;
            fprintf('Debug log cleared\n');
        end
    end
end 