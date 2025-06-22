classdef ClipboardUtils
    % ClipboardUtils - Utility class for handling clipboard operations and log formatting
    %
    % This class provides static methods for formatting and copying analysis logs
    % to the clipboard. It maintains the exact same formatting and behavior as the
    % original inline code, but organizes it into reusable functions.
    %
    % The clipboard utilities handle:
    % - Formatting log entries with consistent structure
    % - Converting analysis results into human-readable messages
    % - Copying formatted logs to system clipboard
    % - Fallback to console output if clipboard operations fail
    %
    % Example Usage:
    %   % Copy analysis log to clipboard:
    %   ClipboardUtils.copyAnalysisLogToClipboard(analysis_log);
    
    methods(Static)
        function copyAnalysisLogToClipboard(analysis_log)
            % COPYANALYSISLOGTOCLIPBOARD - Format and copy analysis log to clipboard
            %
            % This function maintains the exact same formatting and behavior as the
            % original code, creating a formatted log text and copying it to the
            % system clipboard.
            %
            % Inputs:
            %   analysis_log - Cell array containing analysis log entries, each entry
            %                 being a structure with fields like timestamp, level,
            %                 cell_id, etc.
            %
            % The function will:
            % 1. Create a formatted header with timestamp and entry count
            % 2. Format each log entry with consistent structure
            % 3. Attempt to copy to system clipboard
            % 4. Provide console output as fallback if clipboard copy fails
            
            fprintf('\n=== LOG MANAGEMENT ===\n');
            fprintf('Automatically copying analysis log to clipboard...\n');

            % Create log text for clipboard (same format as original)
            log_text = sprintf('=== SECTION 1 ANALYSIS LOG ===\n');
            log_text = [log_text sprintf('Generated: %s\n', char(datetime('now')))];
            log_text = [log_text sprintf('Total entries: %d\n\n', length(analysis_log))];

            % Format each entry using the same structure as original code
            for i = 1:length(analysis_log)
                entry = analysis_log{i};
                log_text = [log_text ClipboardUtils.formatLogEntry(entry)];
            end

            % Try clipboard copy with same error handling as original
            try
                clipboard('copy', log_text);
                fprintf('✓ Analysis log copied to clipboard!\n');
                fprintf('Total entries: %d\n', length(analysis_log));
            catch ME
                fprintf('⚠ Clipboard copy failed: %s\n', ME.message);
                fprintf('Log ready for manual copy:\n');
                fprintf('==========================================\n');
                fprintf('%s', log_text);
                fprintf('==========================================\n');
            end
        end
        
        function formatted_text = formatLogEntry(entry)
            % FORMATLOGENTRY - Format a single log entry
            %
            % This function formats a single log entry with the exact same structure
            % and appearance as the original code.
            %
            % Inputs:
            %   entry - Structure containing log entry data with fields like:
            %          timestamp, level, cell_id, frequency, selected_sd, etc.
            %
            % Returns:
            %   formatted_text - String containing the formatted log entry with:
            %     - Timestamp in brackets
            %     - Level (SUCCESS/ERROR)
            %     - Cell ID
            %     - Frequency and SD values
            %     - Appropriate message based on entry type
            
            if isfield(entry, 'level') && isfield(entry, 'cell_id') && ...
               isfield(entry, 'frequency') && isfield(entry, 'selected_sd')
                % Standard log entry format (same as original)
                formatted_text = sprintf('[%s] %s | %s | Freq:%s | SD:%s | %s\n', ...
                    char(entry.timestamp), entry.level, entry.cell_id, ...
                    num2str(entry.frequency), num2str(entry.selected_sd), ...
                    ClipboardUtils.getLogMessage(entry));
            else
                % Master log entry or other format (same as original)
                formatted_text = sprintf('[%s] %s\n', ...
                    char(entry.timestamp), ClipboardUtils.getLogMessage(entry));
            end
        end
        
        function message = getLogMessage(entry)
            % GETLOGMESSAGE - Get appropriate message for log entry
            %
            % This function generates the appropriate message for each log entry type,
            % maintaining the exact same message format as the original code.
            %
            % Inputs:
            %   entry - Structure containing log entry data
            %
            % Returns:
            %   message - Formatted message string based on entry type:
            %     - For analysis entries: Analysis type and purpose
            %     - For spike data: Spike count and firing rate
            %     - For errors: Error message
            %     - For other types: Generic log entry message
            
            if isfield(entry, 'message')
                message = entry.message;
            elseif isfield(entry, 'analysis_type')
                message = sprintf('Analysis: %s - %s', entry.analysis_type, entry.purpose);
            elseif isfield(entry, 'n_spikes') && isfield(entry, 'firing_rate_Hz')
                message = sprintf('Spikes: %d (%.2f Hz)', entry.n_spikes, entry.firing_rate_Hz);
            elseif isfield(entry, 'error_message')
                message = sprintf('Error: %s', entry.error_message);
            else
                message = 'Log entry';
            end
        end
    end
end 