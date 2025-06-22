classdef LoggingUtils
    % LoggingUtils - Utility class for handling analysis logging functions
    %
    % This class provides static methods for consistent logging across the project.
    % It maintains the exact same logging structure as the original inline code,
    % but organizes it into reusable functions.
    %
    % The logging system tracks:
    % - Successful analyses with spike counts, firing rates, and filter correlations
    % - Failed analyses with error messages and debugging information
    % - Timestamps and metadata for each analysis attempt
    %
    % Example Usage:
    %   % For successful analysis:
    %   log_entry = LoggingUtils.logSuccessfulAnalysis(cell_name, protocol_name, ...
    %       freq_value, selected_sd_value, max_epochs, results);
    %
    %   % For failed analysis:
    %   [log_entry, failed_analysis] = LoggingUtils.logFailedAnalysis(cell_name, ...
    %       protocol_name, freq_value, selected_sd_value, max_epochs, ME);
    
    methods(Static)
        function log_entry = logSuccessfulAnalysis(cell_name, protocol_name, freq_value, selected_sd_value, max_epochs, results)
            % LOGSUCCESSFULANALYSIS - Create a log entry for successful analysis
            %
            % This function creates a structured log entry for a successful analysis,
            % maintaining the exact same fields and format as the original code.
            %
            % Inputs:
            %   cell_name - Name/ID of the cell being analyzed
            %   protocol_name - Name of the protocol used
            %   freq_value - Frequency value used in the analysis
            %   selected_sd_value - Selected SD value for the analysis
            %   max_epochs - Number of epochs in the analysis
            %   results - Structure containing analysis results with fields:
            %     - n_spikes: Number of spikes detected
            %     - firing_rate_Hz: Firing rate in Hz
            %     - filter_correlation: Linear filter correlation value
            %     - spike_width_ms: Spike width in milliseconds
            %
            % Returns:
            %   log_entry - Structured log entry containing:
            %     - timestamp: Current date/time
            %     - level: 'SUCCESS'
            %     - cell_id: Cell identifier
            %     - protocol: Protocol name
            %     - frequency: Analysis frequency
            %     - selected_sd: Selected SD value
            %     - n_epochs: Number of epochs
            %     - n_spikes: Number of detected spikes
            %     - firing_rate_Hz: Calculated firing rate
            %     - filter_correlation: Filter correlation value
            %     - spike_width_ms: Measured spike width
            %     - success: true
            
            log_entry = struct();
            log_entry.timestamp = datetime('now');
            log_entry.level = 'SUCCESS';
            log_entry.cell_id = cell_name;
            log_entry.protocol = protocol_name;
            log_entry.frequency = freq_value;
            log_entry.selected_sd = selected_sd_value;
            log_entry.n_epochs = max_epochs;
            log_entry.n_spikes = results.n_spikes;
            log_entry.firing_rate_Hz = results.firing_rate_Hz;
            log_entry.filter_correlation = results.filter_correlation;
            log_entry.spike_width_ms = results.spike_width_ms;
            log_entry.success = true;
        end
        
        function [log_entry, failed_analysis] = logFailedAnalysis(cell_name, protocol_name, freq_value, selected_sd_value, max_epochs, error_info)
            % LOGFAILEDANALYSIS - Create log entries for failed analysis
            %
            % This function creates both a log entry and a detailed failed analysis
            % record when an analysis fails, maintaining the exact structure and
            % information as the original code.
            %
            % Inputs:
            %   cell_name - Name/ID of the cell being analyzed
            %   protocol_name - Name of the protocol used
            %   freq_value - Frequency value that was being analyzed
            %   selected_sd_value - Selected SD value for the analysis
            %   max_epochs - Number of epochs in the attempted analysis
            %   error_info - Structure containing error information with fields:
            %     - message: Error message string
            %     - identifier: Error identifier
            %
            % Returns:
            %   log_entry - Structured log entry containing:
            %     - timestamp: Current date/time
            %     - level: 'ERROR'
            %     - cell_id: Cell identifier
            %     - protocol: Protocol name
            %     - frequency: Analysis frequency
            %     - selected_sd: Selected SD value
            %     - error_message: Description of what went wrong
            %     - error_id: Error identifier
            %     - success: false
            %
            %   failed_analysis - Detailed failed analysis record containing:
            %     - cell_id: Cell identifier
            %     - protocol: Protocol name
            %     - frequency: Analysis frequency
            %     - selected_sd: Selected SD value
            %     - n_epochs: Number of epochs
            %     - error_message: Description of what went wrong
            %     - error_id: Error identifier
            %     - timestamp: When the error occurred
            
            % Create failed analysis record with the same structure as original
            failed_analysis = struct();
            failed_analysis.cell_id = cell_name;
            failed_analysis.protocol = protocol_name;
            failed_analysis.frequency = freq_value;
            failed_analysis.selected_sd = selected_sd_value;
            failed_analysis.n_epochs = max_epochs;
            failed_analysis.error_message = error_info.message;
            failed_analysis.error_id = error_info.identifier;
            failed_analysis.timestamp = datetime('now');

            % Create log entry with the same structure as original
            log_entry = struct();
            log_entry.timestamp = datetime('now');
            log_entry.level = 'ERROR';
            log_entry.cell_id = cell_name;
            log_entry.protocol = protocol_name;
            log_entry.frequency = freq_value;
            log_entry.selected_sd = selected_sd_value;
            log_entry.error_message = error_info.message;
            log_entry.error_id = error_info.identifier;
            log_entry.success = false;
        end
    end
end 