classdef DebugReporter
    % DEBUGREPORTER - Class for generating debug reports and logs
    % This class handles the generation of debug information, including
    % detailed logs, error tracking, and diagnostic information.
    %
    % The class provides methods to:
    % - Generate debug logs
    % - Track execution flow
    % - Monitor performance
    % - Export debug information
    % - Provide diagnostic tools
    %
    % Properties:
    %   verbose - Verbosity level for logging
    %   debug_log - Structure array containing debug entries
    %   log_counter - Counter for debug log entries
    %
    % Author: Maxwell
    % Date: 2024
    
    properties
        verbose = true
        debug_log = {}
        log_counter = 1
    end
    
    methods
        function obj = DebugReporter(verbose)
            % DEBUGREPORTER Constructor
            % Input:
            %   verbose - Verbosity level (default: true)
            % Output:
            %   obj - DebugReporter instance
            
            if nargin >= 1
                obj.verbose = verbose;
            end
            
            % Initialize debug log
            obj.debug_log = {};
            obj.log_counter = 1;
        end
        
        function generateDebugLog(obj)
            % GENERATEDEBUGLOG - Generate comprehensive debug log
            % Creates a detailed debug log with all available information
            % and exports it for analysis.
            
            if obj.verbose
                fprintf('\n=== GENERATING DEBUG LOG ===\n');
            end
            
            try
                % Generate debug summary
                debug_summary = obj.generateDebugSummary();
                
                % Display debug information
                obj.displayDebugSummary(debug_summary);
                
                % Export debug log
                debug_export = obj.exportDebugLog();
                
                % Save debug information
                obj.saveDebugLog(debug_export);
                
                if obj.verbose
                    fprintf('Debug log generated successfully\n');
                end
                
            catch ME
                if obj.verbose
                    fprintf('Failed to generate debug log: %s\n', ME.message);
                end
            end
        end
        
        function logEntry(obj, level, category, message, data)
            % LOGENTRY - Add entry to debug log
            % Input:
            %   level - Log level ('INFO', 'WARN', 'ERROR', 'SUCCESS')
            %   category - Category of the log entry
            %   message - Log message
            %   data - Additional data (optional)
            
            if nargin < 5
                data = struct();
            end
            
            % Create log entry
            log_entry = struct();
            log_entry.timestamp = datetime('now');
            log_entry.level = level;
            log_entry.category = category;
            log_entry.message = message;
            log_entry.data = data;
            
            % Add to debug log
            obj.debug_log{obj.log_counter} = log_entry;
            obj.log_counter = obj.log_counter + 1;
            
            % Display if verbose
            if obj.verbose
                obj.displayLogEntry(log_entry);
            end
        end
        
        function displayLogEntry(obj, log_entry)
            % DISPLAYLOGENTRY - Display a single log entry
            % Input:
            %   log_entry - Log entry structure
            
            timestamp_str = char(log_entry.timestamp);
            level_prefix = sprintf('[%s-%s]', log_entry.level, log_entry.category);
            
            fprintf('%s %s: %s\n', timestamp_str, level_prefix, log_entry.message);
            
            % Display data if present
            if ~isempty(fieldnames(log_entry.data))
                data_fields = fieldnames(log_entry.data);
                for i = 1:length(data_fields)
                    field = data_fields{i};
                    value = log_entry.data.(field);
                    if isnumeric(value)
                        fprintf('    %s: %g\n', field, value);
                    else
                        fprintf('    %s: %s\n', field, char(value));
                    end
                end
            end
        end
        
        function debug_summary = generateDebugSummary(obj)
            % GENERATEDEBUGSUMMARY - Generate summary of debug information
            % Output:
            %   debug_summary - Structure containing debug summary
            
            debug_summary = struct();
            debug_summary.timestamp = datetime('now');
            debug_summary.total_entries = length(obj.debug_log);
            
            if debug_summary.total_entries == 0
                debug_summary.success = false;
                debug_summary.message = 'No debug entries found';
                return;
            end
            
            % Analyze log entries by level
            levels = {};
            categories = {};
            
            for i = 1:length(obj.debug_log)
                entry = obj.debug_log{i};
                levels{end+1} = entry.level;
                categories{end+1} = entry.category;
            end
            
            % Count by level
            unique_levels = unique(levels);
            level_counts = zeros(length(unique_levels), 1);
            
            for i = 1:length(unique_levels)
                level_counts(i) = sum(strcmp(levels, unique_levels{i}));
            end
            
            debug_summary.level_analysis = struct(...
                'levels', unique_levels, ...
                'counts', level_counts, ...
                'frequencies', level_counts / debug_summary.total_entries ...
            );
            
            % Count by category
            unique_categories = unique(categories);
            category_counts = zeros(length(unique_categories), 1);
            
            for i = 1:length(unique_categories)
                category_counts(i) = sum(strcmp(categories, unique_categories{i}));
            end
            
            debug_summary.category_analysis = struct(...
                'categories', unique_categories, ...
                'counts', category_counts, ...
                'frequencies', category_counts / debug_summary.total_entries ...
            );
            
            % Analyze time distribution
            timestamps = zeros(length(obj.debug_log), 1);
            for i = 1:length(obj.debug_log)
                timestamps(i) = datenum(obj.debug_log{i}.timestamp);
            end
            
            debug_summary.time_analysis = struct(...
                'start_time', obj.debug_log{1}.timestamp, ...
                'end_time', obj.debug_log{end}.timestamp, ...
                'duration_minutes', (timestamps(end) - timestamps(1)) * 24 * 60, ...
                'entries_per_minute', debug_summary.total_entries / ...
                    max(1, (timestamps(end) - timestamps(1)) * 24 * 60) ...
            );
            
            debug_summary.success = true;
        end
        
        function displayDebugSummary(obj, debug_summary)
            % DISPLAYDEBUGSUMMARY - Display debug summary
            % Input:
            %   debug_summary - Debug summary structure
            
            fprintf('\n===============================================\n');
            fprintf('DEBUG LOG SUMMARY\n');
            fprintf('===============================================\n');
            fprintf('Generated: %s\n', char(debug_summary.timestamp));
            
            if ~debug_summary.success
                fprintf('No debug entries found.\n');
                fprintf('===============================================\n');
                return;
            end
            
            % Basic statistics
            fprintf('\nBasic Statistics:\n');
            fprintf('  Total Entries: %d\n', debug_summary.total_entries);
            fprintf('  Start Time: %s\n', char(debug_summary.time_analysis.start_time));
            fprintf('  End Time: %s\n', char(debug_summary.time_analysis.end_time));
            fprintf('  Duration: %.1f minutes\n', debug_summary.time_analysis.duration_minutes);
            fprintf('  Entries per Minute: %.2f\n', debug_summary.time_analysis.entries_per_minute);
            
            % Level analysis
            fprintf('\nBy Level:\n');
            la = debug_summary.level_analysis;
            for i = 1:length(la.levels)
                fprintf('  %s: %d (%.1f%%)\n', ...
                    la.levels{i}, la.counts(i), la.frequencies(i) * 100);
            end
            
            % Category analysis
            fprintf('\nBy Category:\n');
            ca = debug_summary.category_analysis;
            for i = 1:length(ca.categories)
                fprintf('  %s: %d (%.1f%%)\n', ...
                    ca.categories{i}, ca.counts(i), ca.frequencies(i) * 100);
            end
            
            fprintf('\n===============================================\n');
        end
        
        function debug_export = exportDebugLog(obj)
            % EXPORTDEBUGLOG - Export debug log for sharing
            % Output:
            %   debug_export - Formatted debug log string
            
            if isempty(obj.debug_log)
                debug_export = 'No debug log entries found.';
                return;
            end
            
            debug_export = sprintf('=== DEBUG LOG EXPORT ===\n');
            debug_export = [debug_export sprintf('Generated: %s\n', char(datetime('now')))];
            debug_export = [debug_export sprintf('Total entries: %d\n\n', length(obj.debug_log))];
            
            for i = 1:length(obj.debug_log)
                entry = obj.debug_log{i};
                
                line = sprintf('[%s] %s-%s: %s\n', ...
                    char(entry.timestamp), entry.level, entry.category, entry.message);
                debug_export = [debug_export line];
                
                % Add data if present
                if ~isempty(fieldnames(entry.data))
                    data_fields = fieldnames(entry.data);
                    for j = 1:length(data_fields)
                        field = data_fields{j};
                        value = entry.data.(field);
                        if isnumeric(value)
                            line = sprintf('    %s: %g\n', field, value);
                        else
                            line = sprintf('    %s: %s\n', field, char(value));
                        end
                        debug_export = [debug_export line];
                    end
                end
            end
            
            debug_export = [debug_export sprintf('\n=== END DEBUG LOG ===\n')];
        end
        
        function saveDebugLog(obj, debug_export)
            % SAVEDEBUGLOG - Save debug log to file
            % Input:
            %   debug_export - Formatted debug log string
            
            try
                % Create filename with timestamp
                timestamp_str = datestr(datetime('now'), 'yyyymmdd_HHMMSS');
                filename = sprintf('debug_log_%s.txt', timestamp_str);
                
                % Write to file
                fid = fopen(filename, 'w');
                if fid == -1
                    error('Could not open file for writing: %s', filename);
                end
                
                fprintf(fid, '%s', debug_export);
                fclose(fid);
                
                if obj.verbose
                    fprintf('Debug log saved to: %s\n', filename);
                end
                
            catch ME
                if obj.verbose
                    fprintf('Failed to save debug log: %s\n', ME.message);
                end
            end
        end
        
        function copyToClipboard(obj)
            % COPYTOCLIPBOARD - Copy debug log to clipboard
            
            if isempty(obj.debug_log)
                if obj.verbose
                    fprintf('No debug log entries to copy.\n');
                end
                return;
            end
            
            try
                % Create simplified log for clipboard
                clipboard_text = sprintf('MATLAB DEBUG LOG - %s\n', char(datetime('now')));
                clipboard_text = [clipboard_text sprintf('Entries: %d\n', length(obj.debug_log))];
                clipboard_text = [clipboard_text sprintf('====================\n')];
                
                for i = 1:length(obj.debug_log)
                    entry = obj.debug_log{i};
                    line = sprintf('%s [%s-%s]: %s\n', ...
                        char(entry.timestamp), entry.level, entry.category, entry.message);
                    clipboard_text = [clipboard_text line];
                end
                
                clipboard_text = [clipboard_text sprintf('====================\n')];
                
                % Copy to clipboard
                clipboard('copy', clipboard_text);
                
                if obj.verbose
                    fprintf('Debug log copied to clipboard! (%d entries)\n', length(obj.debug_log));
                end
                
            catch ME
                if obj.verbose
                    fprintf('Failed to copy debug log to clipboard: %s\n', ME.message);
                end
            end
        end
        
        function clearLog(obj)
            % CLEARLOG - Clear the debug log
            
            obj.debug_log = {};
            obj.log_counter = 1;
            
            if obj.verbose
                fprintf('Debug log cleared.\n');
            end
        end
        
        function filtered_log = filterLog(obj, level, category)
            % FILTERLOG - Filter debug log by level and/or category
            % Input:
            %   level - Level to filter by (optional)
            %   category - Category to filter by (optional)
            % Output:
            %   filtered_log - Filtered log entries
            
            filtered_log = {};
            
            for i = 1:length(obj.debug_log)
                entry = obj.debug_log{i};
                include_entry = true;
                
                % Filter by level
                if nargin >= 2 && ~isempty(level)
                    if ~strcmp(entry.level, level)
                        include_entry = false;
                    end
                end
                
                % Filter by category
                if nargin >= 3 && ~isempty(category)
                    if ~strcmp(entry.category, category)
                        include_entry = false;
                    end
                end
                
                if include_entry
                    filtered_log{end+1} = entry;
                end
            end
        end
        
        function displayFilteredLog(obj, level, category)
            % DISPLAYFILTEREDLOG - Display filtered debug log
            % Input:
            %   level - Level to filter by (optional)
            %   category - Category to filter by (optional)
            
            filtered_log = obj.filterLog(level, category);
            
            if isempty(filtered_log)
                fprintf('No entries found matching the filter criteria.\n');
                return;
            end
            
            fprintf('\n=== FILTERED DEBUG LOG ===\n');
            if nargin >= 2 && ~isempty(level)
                fprintf('Level: %s\n', level);
            end
            if nargin >= 3 && ~isempty(category)
                fprintf('Category: %s\n', category);
            end
            fprintf('Entries: %d\n\n', length(filtered_log));
            
            for i = 1:length(filtered_log)
                obj.displayLogEntry(filtered_log{i});
            end
            
            fprintf('=== END FILTERED LOG ===\n');
        end
    end
end 