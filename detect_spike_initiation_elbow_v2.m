function [elbow_indices, spike_peaks, isi, avg_spike, diagnostic_info] = detect_spike_initiation_elbow_v2(Vm_all, dt, vm_thresh, d2v_thresh, search_back_ms, visualize, varargin)
% Enhanced spike detection with false positive reduction and performance optimization
% 
% INPUTS:
%   Vm_all - voltage trace (mV)
%   dt - time step (s)
%   vm_thresh - minimum peak threshold for spike detection (mV), e.g., -30
%   d2v_thresh - second derivative threshold for elbow detection (mV/ms^2)
%   search_back_ms - time window to search back from peak (ms)
%   visualize - boolean, whether to create plots
%   varargin - optional parameter-value pairs:
%     'min_peak_height' - minimum voltage for valid spikes (default: -60 mV)
%     'min_dv_thresh' - minimum dV/dt at elbow point (default: 1 mV/ms)
%     'debug_low_peaks' - show cases where peak < min_peak_height (default: false)
%     'elbow_thresh' - elbow voltage threshold for artifact detection (default: -65 mV)
%     'spike_thresh' - peak threshold for real spikes (default: -10 mV)
%     'time_to_peak_thresh' - time-to-peak variability threshold (default: 0.8 ms)
%     'max_plot_spikes' - maximum spikes to plot for speed (default: 50)
%     'plot_outliers_only' - show only outlier waveforms (default: false)
%
% OUTPUTS:
%   elbow_indices - spike initiation points (indices)
%   spike_peaks - spike peak indices
%   isi - inter-spike intervals (ms)
%   avg_spike - average spike waveform
%   diagnostic_info - structure with additional diagnostic information

% START TIMING
tic; % Start overall timer
t_start = tic;

% Parse optional inputs
p = inputParser;
addParameter(p, 'min_peak_height', -60, @isnumeric);  % Voltage threshold for real spikes
addParameter(p, 'min_dv_thresh', 1, @isnumeric);     % Minimum dV/dt at elbow
addParameter(p, 'debug_low_peaks', false, @islogical); % Debug mode for low peaks
addParameter(p, 'elbow_thresh', -65, @isnumeric);    % Elbow voltage threshold for artifact detection
addParameter(p, 'spike_thresh', -10, @isnumeric);    % Peak threshold for real spikes
addParameter(p, 'time_to_peak_thresh', 0.8, @isnumeric); % Time-to-peak variability threshold (ms)
addParameter(p, 'max_plot_spikes', 50, @isnumeric);   % Maximum spikes to plot for speed
addParameter(p, 'plot_outliers_only', false, @islogical); % Show only outlier waveforms
parse(p, varargin{:});

min_peak_height = p.Results.min_peak_height;
min_dv_thresh = p.Results.min_dv_thresh;
debug_low_peaks = p.Results.debug_low_peaks;
elbow_thresh = p.Results.elbow_thresh;
spike_thresh = p.Results.spike_thresh;
time_to_peak_thresh = p.Results.time_to_peak_thresh;
max_plot_spikes = p.Results.max_plot_spikes;
plot_outliers_only = p.Results.plot_outliers_only;

% --- Parameters ---
search_back_pts = round(search_back_ms / 1000 / dt);

fprintf('\n=== SPIKE DETECTION STARTED ===\n');
fprintf('Recording: %.2f sec, %.1f kHz sampling, %d points\n', ...
    length(Vm_all)*dt, 1/dt/1000, length(Vm_all));

% --- Derivatives ---
t_deriv = tic;
dVm = [0; diff(Vm_all)] / dt / 1000; % mV/ms
ddVm = [0; diff(dVm)] / dt / 1000; % mV/ms^2
t_deriv_elapsed = toc(t_deriv);
fprintf('Derivatives computed: %.1f ms\n', t_deriv_elapsed*1000);

% --- Spike peaks with enhanced filtering ---
t_peaks = tic;
is_peak = (Vm_all(2:end-1) > Vm_all(1:end-2)) & (Vm_all(2:end-1) > Vm_all(3:end));
spike_peaks_all = find(is_peak) + 1;

% Apply voltage threshold
spike_peaks_thresh = spike_peaks_all(Vm_all(spike_peaks_all) > vm_thresh);

% NEW: Apply peak height filter to reduce false positives
spike_peaks = spike_peaks_thresh(Vm_all(spike_peaks_thresh) > min_peak_height);

% Store rejected peaks for diagnostics
rejected_peaks = spike_peaks_thresh(Vm_all(spike_peaks_thresh) <= min_peak_height);
t_peaks_elapsed = toc(t_peaks);
fprintf('Peak detection: %.1f ms (%d total → %d above thresh → %d above height)\n', ...
    t_peaks_elapsed*1000, length(spike_peaks_all), length(spike_peaks_thresh), length(spike_peaks));

% --- Enhanced Elbow detection ---
t_elbow = tic;
elbow_indices = [];
elbow_peak_heights = [];
elbow_dv_values = [];
elbow_time_to_peak = [];  % NEW: Track time from elbow to peak
rejected_elbows = [];
rejection_reasons = {};

for i = 1:length(spike_peaks)
    t_peak = spike_peaks(i);
    window = max(1, t_peak - search_back_pts):t_peak;
    d2v = ddVm(window);
    
    % Find elbow point (last crossing of d2v threshold)
    crossing = find(d2v(1:end-1) < d2v_thresh & d2v(2:end) >= d2v_thresh, 1, 'last');
    
    if ~isempty(crossing)
        elbow_idx = window(crossing);
        peak_height = Vm_all(t_peak);
        dv_at_elbow = dVm(elbow_idx);
        time_to_peak_ms = (t_peak - elbow_idx) * dt * 1000;  % Convert to ms
        
        % NEW: Additional validation criteria
        is_valid = true;
        reason = '';
        
        % Check if dV/dt at elbow is sufficiently high
        if dv_at_elbow < min_dv_thresh
            is_valid = false;
            reason = sprintf('Low dV/dt: %.2f < %.2f', dv_at_elbow, min_dv_thresh);
        end
        
        % Check if peak reaches minimum height (redundant but explicit)
        if peak_height <= min_peak_height
            is_valid = false;
            reason = sprintf('Low peak: %.1f <= %.1f', peak_height, min_peak_height);
        end
        
        % NEW: Smart filtering - catch current injection artifacts
        % Reject if elbow is very hyperpolarized AND peak doesn't reach spike threshold
        elbow_voltage = Vm_all(elbow_idx);
        if elbow_voltage < elbow_thresh && peak_height < spike_thresh
            is_valid = false;
            reason = sprintf('Current artifact: elbow %.1fmV < %.1fmV AND peak %.1fmV < %.1fmV', ...
                elbow_voltage, elbow_thresh, peak_height, spike_thresh);
        end
        
        % NEW: Reject unrealistically fast time-to-peak (< 0.2ms)
        if time_to_peak_ms < 0.2
            is_valid = false;
            reason = sprintf('Unrealistic time-to-peak: %.2fms < 0.2ms', time_to_peak_ms);
        end
        
        if is_valid
            elbow_indices(end+1) = elbow_idx;
            elbow_peak_heights(end+1) = peak_height;
            elbow_dv_values(end+1) = dv_at_elbow;
            elbow_time_to_peak(end+1) = time_to_peak_ms;
        else
            rejected_elbows(end+1) = elbow_idx;
            rejection_reasons{end+1} = reason;
        end
    end
end
t_elbow_elapsed = toc(t_elbow);
fprintf('Elbow detection: %.1f ms (%d valid, %d rejected)\n', ...
    t_elbow_elapsed*1000, length(elbow_indices), length(rejected_elbows));

% NEW: Secondary filtering based on time-to-peak consistency
t_ttp = tic;
time_to_peak_outliers = [];
if length(elbow_time_to_peak) > 3  % Need at least 4 spikes for statistics
    mean_time_to_peak = mean(elbow_time_to_peak);
    std_time_to_peak = std(elbow_time_to_peak);
    
    % Smart threshold: not smaller than half the mean value
    effective_thresh = max(time_to_peak_thresh, mean_time_to_peak * 0.5);
    
    % Find outliers (spikes with very different time-to-peak)
    outlier_indices = [];
    for i = 1:length(elbow_time_to_peak)
        if abs(elbow_time_to_peak(i) - mean_time_to_peak) > effective_thresh
            outlier_indices(end+1) = i;
            time_to_peak_outliers(end+1) = elbow_time_to_peak(i);
        end
    end
    
    % Move outliers to rejected category
    if ~isempty(outlier_indices)
        for i = length(outlier_indices):-1:1  % Reverse order to maintain indices
            idx = outlier_indices(i);
            rejected_elbows(end+1) = elbow_indices(idx);
            rejection_reasons{end+1} = sprintf('Time-to-peak outlier: %.2fms (mean: %.2f, thresh: %.2f)', ...
                elbow_time_to_peak(idx), mean_time_to_peak, effective_thresh);
            
            % Remove from valid arrays
            elbow_indices(idx) = [];
            elbow_peak_heights(idx) = [];
            elbow_dv_values(idx) = [];
            elbow_time_to_peak(idx) = [];
        end
    end
end
t_ttp_elapsed = toc(t_ttp);
if ~isempty(time_to_peak_outliers)
    fprintf('Time-to-peak filtering: %.1f ms (%d outliers removed)\n', ...
        t_ttp_elapsed*1000, length(time_to_peak_outliers));
end

% --- Extract waveforms with speed optimization ---
t_waveforms = tic;
win_ms = 4;
win_pts = round(win_ms / 1000 / dt);
pre_elbow_ms = 3;  % Start 3ms before elbow for rejected spikes
pre_elbow_pts = round(pre_elbow_ms / 1000 / dt);

% Valid spike waveforms (subsample for speed if many spikes)
valid_elbows = elbow_indices(elbow_indices + win_pts - 1 <= length(Vm_all));
n_valid = length(valid_elbows);
n_plot_valid = min(n_valid, max_plot_spikes);

if n_valid > 0
    if n_plot_valid < n_valid
        % Subsample evenly distributed spikes
        plot_indices = round(linspace(1, n_valid, n_plot_valid));
        plot_valid_elbows = valid_elbows(plot_indices);
    else
        plot_valid_elbows = valid_elbows;
    end
    
    spike_matrix = zeros(length(plot_valid_elbows), win_pts);
    for i = 1:length(plot_valid_elbows)
        idx_start = plot_valid_elbows(i);
        idx_end = idx_start + win_pts - 1;
        spike_matrix(i, :) = Vm_all(idx_start:idx_end);
    end
    
    % Always compute average from ALL spikes, not just plotted ones
    all_spike_matrix = zeros(n_valid, win_pts);
    for i = 1:n_valid
        idx_start = valid_elbows(i);
        idx_end = idx_start + win_pts - 1;
        all_spike_matrix(i, :) = Vm_all(idx_start:idx_end);
    end
    avg_spike = mean(all_spike_matrix, 1);
else
    spike_matrix = [];
    avg_spike = zeros(1, win_pts);
end

% Rejected spike waveforms (optimized extraction)
all_rejected = [rejected_peaks; rejected_elbows'];
rejected_matrix = [];
rejected_elbow_starts = [];
outlier_matrix = [];  % NEW: Separate matrix for outliers

if ~isempty(all_rejected)
    n_rejected = length(all_rejected);
    n_plot_rejected = min(n_rejected, max_plot_spikes);
    
    if plot_outliers_only && ~isempty(time_to_peak_outliers)
        % Plot only time-to-peak outliers
        outlier_elbows = rejected_elbows(end-length(time_to_peak_outliers)+1:end);
        plot_rejected = outlier_elbows;
    else
        % Subsample rejected spikes
        if n_plot_rejected < n_rejected
            plot_indices = round(linspace(1, n_rejected, n_plot_rejected));
            plot_rejected = all_rejected(plot_indices);
        else
            plot_rejected = all_rejected;
        end
    end
    
    % Extract rejected waveforms
    for i = 1:length(plot_rejected)
        elbow_idx = plot_rejected(i);
        idx_start = max(1, elbow_idx - pre_elbow_pts);
        idx_end = min(idx_start + win_pts - 1, length(Vm_all));
        
        if idx_end - idx_start + 1 == win_pts
            if isempty(rejected_matrix)
                rejected_matrix = zeros(1, win_pts);
                rejected_matrix(1, :) = Vm_all(idx_start:idx_end);
                rejected_elbow_starts = pre_elbow_pts + 1;
            else
                rejected_matrix(end+1, :) = Vm_all(idx_start:idx_end);
                rejected_elbow_starts(end+1) = pre_elbow_pts + 1;
            end
        end
    end
end

t_wave = (0:win_pts-1) * dt * 1000; % in ms
t_wave_rejected = t_wave - pre_elbow_ms; % Adjust time axis for rejected
t_waveforms_elapsed = toc(t_waveforms);
fprintf('Waveform extraction: %.1f ms (valid: %d/%d plotted, rejected: %d)\n', ...
    t_waveforms_elapsed*1000, n_plot_valid, n_valid, size(rejected_matrix, 1));

% --- Compute ISI (in ms) ---
if length(elbow_indices) > 1
    spike_times_ms = elbow_indices * dt * 1000;
    isi = diff(spike_times_ms);
else
    isi = [];
end

% --- Store timing information ---
total_time = toc(t_start);

% --- Diagnostic Information ---
diagnostic_info = struct();
diagnostic_info.total_peaks_found = length(spike_peaks_all);
diagnostic_info.peaks_above_thresh = length(spike_peaks_thresh);
diagnostic_info.peaks_above_height = length(spike_peaks);
diagnostic_info.rejected_low_peaks = rejected_peaks;
diagnostic_info.rejected_elbows = rejected_elbows;
diagnostic_info.rejection_reasons = rejection_reasons;
diagnostic_info.elbow_peak_heights = elbow_peak_heights;
diagnostic_info.elbow_dv_values = elbow_dv_values;
diagnostic_info.elbow_time_to_peak = elbow_time_to_peak;  
diagnostic_info.time_to_peak_outliers = time_to_peak_outliers;  % NEW
diagnostic_info.n_plotted_valid = n_plot_valid;  % NEW: Track how many plotted
diagnostic_info.n_plotted_rejected = size(rejected_matrix, 1);  % NEW
diagnostic_info.timing = struct('total', total_time, 'derivatives', t_deriv_elapsed, ...
                               'peaks', t_peaks_elapsed, 'elbows', t_elbow_elapsed, ...
                               'time_to_peak', t_ttp_elapsed, 'waveforms', t_waveforms_elapsed);
diagnostic_info.parameters = struct('min_peak_height', min_peak_height, ...
                                   'min_dv_thresh', min_dv_thresh, ...
                                   'vm_thresh', vm_thresh, ...
                                   'd2v_thresh', d2v_thresh, ...
                                   'elbow_thresh', elbow_thresh, ...
                                   'spike_thresh', spike_thresh, ...
                                   'time_to_peak_thresh', time_to_peak_thresh);

% Print final summary with timing
fprintf('\n=== SPIKE DETECTION COMPLETED ===\n');
fprintf('TOTAL TIME: %.0f ms (%.2f sec)\n', total_time*1000, total_time);
fprintf('Final result: %d valid spikes detected\n', length(elbow_indices));
if length(elbow_indices) > 1
    fprintf('Firing rate: %.2f Hz\n', length(elbow_indices)/(length(Vm_all)*dt));
end
fprintf('Performance: %.1f spikes/sec processing rate\n', length(spike_peaks_all)/total_time);
fprintf('=====================================\n');

% --- Enhanced Visualization ---
if visualize || debug_low_peaks
    t_plot = tic;
    time_axis = (1:length(Vm_all)) * dt;
    
    if ~isempty(elbow_indices)
        zoom_start_s = time_axis(elbow_indices(1));
        zoom_end_s = time_axis(elbow_indices(1)) + 0.03;
    elseif ~isempty(rejected_peaks)
        zoom_start_s = time_axis(rejected_peaks(1));
        zoom_end_s = time_axis(rejected_peaks(1)) + 0.03;
    else
        zoom_start_s = 0;
        zoom_end_s = min(0.1, time_axis(end));
    end
    
    % --- Main plots ---
    if visualize
        figure('Position', [100, 100, 1800, 900]);
        
        % Row 1: Main traces (full width)
        subplot(3,4,[1 2]);
        plot(time_axis, Vm_all, 'k', 'LineWidth', 1.2); hold on;
        if ~isempty(elbow_indices)
            plot(time_axis(elbow_indices), Vm_all(elbow_indices), 'bo', ...
                'MarkerSize', 6, 'LineWidth', 1.5, 'DisplayName', 'Valid Elbows');
        end
        if ~isempty(rejected_peaks)
            plot(time_axis(rejected_peaks), Vm_all(rejected_peaks), 'rx', ...
                'MarkerSize', 8, 'LineWidth', 1.5, 'DisplayName', 'Rejected Peaks');
        end
        if ~isempty(spike_peaks)
            plot(time_axis(spike_peaks), Vm_all(spike_peaks), 'go', ...
                'MarkerSize', 4, 'LineWidth', 1, 'DisplayName', 'Valid Peaks');
        end
        yline(min_peak_height, 'g--', 'LineWidth', 1, 'DisplayName', 'Min peak height');
        yline(vm_thresh, 'm--', 'LineWidth', 1, 'DisplayName', 'Peak thresh');
        if exist('spike_thresh', 'var')
            yline(spike_thresh, 'b--', 'LineWidth', 1, 'DisplayName', 'Spike thresh');
        end
        xlim([zoom_start_s zoom_end_s]);
        ylabel('Vm (mV)');
        title(sprintf('Spike Detection: %d valid, %d rejected peaks, %d rejected elbows (%.0fms total)', ...
            length(elbow_indices), length(rejected_peaks), length(rejected_elbows), total_time*1000));
        legend('Location', 'best');
        grid on;
        
        % Row 1: dV/dt trace 
        subplot(3,4,[3 4]);
        plot(time_axis, dVm, 'r', 'LineWidth', 1.2); hold on;
        if ~isempty(elbow_indices)
            plot(time_axis(elbow_indices), dVm(elbow_indices), 'bo', ...
                'MarkerSize', 6, 'LineWidth', 1.5, 'DisplayName', 'Valid Elbows');
        end
        if ~isempty(rejected_elbows)
            plot(time_axis(rejected_elbows), dVm(rejected_elbows), 'rx', ...
                'MarkerSize', 8, 'LineWidth', 1.5, 'DisplayName', 'Rejected Elbows');
        end
        yline(min_dv_thresh, 'g--', 'LineWidth', 1, 'DisplayName', 'Min dV/dt');
        xlim([zoom_start_s zoom_end_s]);
        ylabel('dV/dt (mV/ms)');
        title(['dV/dt (d²V/dt² ≥ ' num2str(d2v_thresh) ', dV/dt ≥ ' num2str(min_dv_thresh) ')']);
        legend('Location', 'best');
        grid on;
        
        % Row 2: Waveforms
        subplot(3,4,5);
        if ~isempty(spike_matrix)
            % Plot subset with transparency for speed
            plot(t_wave, spike_matrix', 'Color', [0.7 0.7 0.7 0.3]); hold on;
            plot(t_wave, avg_spike, 'k', 'LineWidth', 2);
            if n_plot_valid < n_valid
                legend(sprintf('Subset (%d/%d)', n_plot_valid, n_valid), 'Average (all)', 'Location', 'best');
            else
                legend('Individual', 'Average', 'Location', 'best');
            end
        end
        xlabel('Time (ms)');
        ylabel('Vm (mV)');
        title(['Valid Waveforms (n=' num2str(n_valid) ')']);
        grid on;
        
        subplot(3,4,6);
        
        if ~isempty(rejected_matrix)
            % Plot individual rejected waveforms without average
            plot(t_wave_rejected, rejected_matrix', 'Color', [1 0.5 0.5 0.6]); hold on;
            
            % Mark elbow points on each waveform
            if ~isempty(rejected_elbow_starts)
                for i = 1:length(rejected_elbow_starts)
                    if i <= size(rejected_matrix, 1)
                        elbow_time = t_wave_rejected(rejected_elbow_starts(i));
                        elbow_voltage = rejected_matrix(i, rejected_elbow_starts(i));
                        plot(elbow_time, elbow_voltage, 'ko', 'MarkerSize', 4, 'LineWidth', 1);
                    end
                end
            end
            
            if plot_outliers_only && ~isempty(time_to_peak_outliers)
                legend('Outliers', 'Elbows', 'Location', 'best');
            else
                legend('Individual', 'Elbows', 'Location', 'best');
            end
        end
        xline(0, 'k--', 'LineWidth', 1, 'Alpha', 0.5);  % Mark elbow time
        xlabel('Time from elbow (ms)');
        ylabel('Vm (mV)');
        
        if plot_outliers_only && ~isempty(time_to_peak_outliers)
            title(['Outlier Waveforms (n=' num2str(size(rejected_matrix, 1)) ')']);
        else
            title(['Rejected Waveforms (n=' num2str(size(rejected_matrix, 1)) ')']);
        end
        grid on;
        
        % Row 2: Elbow voltage distribution
        subplot(3,4,7);
        if ~isempty(elbow_indices)
            valid_elbow_voltages = Vm_all(elbow_indices);
            histogram(valid_elbow_voltages, 'BinWidth', 2, 'FaceColor', [0 0.8 0], ...
                'FaceAlpha', 0.7, 'DisplayName', 'Valid'); hold on;
        end
        
        if ~isempty(rejected_elbows)
            rejected_elbow_voltages = Vm_all(rejected_elbows);
            histogram(rejected_elbow_voltages, 'BinWidth', 2, 'FaceColor', [0.8 0 0], ...
                'FaceAlpha', 0.7, 'DisplayName', 'Rejected');
        end
        
        if exist('elbow_thresh', 'var')
            xline(elbow_thresh, 'k--', 'LineWidth', 2);
        end
        
        xlabel('Elbow Voltage (mV)');
        ylabel('Count');
        title('Elbow Distribution');
        legend('Location', 'best');
        grid on;
        
        % Row 2: Time-to-peak distribution
        subplot(3,4,8);
        if ~isempty(elbow_time_to_peak)
            % Fine-grained histogram with 0-1ms range
            histogram(elbow_time_to_peak, 'BinWidth', 0.05, 'BinLimits', [0, 1], ...
                'FaceColor', [0 0.6 0.8], 'FaceAlpha', 0.7); hold on;
            
            mean_ttp = mean(elbow_time_to_peak);
            effective_thresh = max(time_to_peak_thresh, mean_ttp * 0.5);
            min_ttp = min(elbow_time_to_peak);
            max_ttp = max(elbow_time_to_peak);
            
            % Threshold lines
            xline(mean_ttp, 'k-', 'LineWidth', 2);
            xline(mean_ttp + effective_thresh, 'r--', 'LineWidth', 1);
            xline(mean_ttp - effective_thresh, 'r--', 'LineWidth', 1);
            
            % Min/max markers
            xline(min_ttp, 'g:', 'LineWidth', 1.5);
            xline(max_ttp, 'g:', 'LineWidth', 1.5);
            
            % Hard minimum line
            xline(0.2, 'm-', 'LineWidth', 1.5);
            
            % Mark outliers if any
            if ~isempty(time_to_peak_outliers)
                for i = 1:length(time_to_peak_outliers)
                    if time_to_peak_outliers(i) <= 1  % Only mark if in range
                        xline(time_to_peak_outliers(i), 'r:', 'LineWidth', 2, 'Alpha', 0.7);
                    end
                end
            end
            
            legend('Data', 'Mean', '±Thresh', '', 'Min', 'Max', '0.2ms limit', 'Location', 'best');
        end
        xlabel('Time to Peak (ms)');
        ylabel('Count');
        title(['Time-to-Peak (0-1ms, thresh: ' num2str(effective_thresh, '%.2f') 'ms)']);
        xlim([0, 1]);
        grid on;
        
        % Row 3: ISI and Statistics
        subplot(3,4,9);
        if ~isempty(isi)
            histogram(isi, 'BinWidth', 2, 'BinLimits', [0 50], 'FaceColor', [0.3 0.3 0.3]);
        end
        xlabel('ISI (ms)');
        ylabel('Count');
        title('Inter-Spike Intervals');
        grid on;
        
        % Combined statistics in remaining space
        subplot(3,4,[10 11 12]);
        axis off;
        
        % Create comprehensive text summary
        stats_text = {
            sprintf('DETECTION SUMMARY');
            sprintf('Total peaks: %d → Above thresh: %d → Height filter: %d → Valid elbows: %d', ...
                diagnostic_info.total_peaks_found, diagnostic_info.peaks_above_thresh, ...
                diagnostic_info.peaks_above_height, length(elbow_indices));
            sprintf('Rejected: %d peaks + %d elbows = %d total', ...
                length(rejected_peaks), length(rejected_elbows), ...
                length(rejected_peaks) + length(rejected_elbows));
            sprintf('');
            sprintf('PERFORMANCE');
            sprintf('Total time: %.0f ms (%.2f sec)', total_time*1000, total_time);
            sprintf('  • Derivatives: %.1f ms', t_deriv_elapsed*1000);
            sprintf('  • Peak detection: %.1f ms', t_peaks_elapsed*1000);
            sprintf('  • Elbow detection: %.1f ms', t_elbow_elapsed*1000);
            sprintf('  • Waveform extraction: %.1f ms', t_waveforms_elapsed*1000);
            sprintf('Processing rate: %.1f spikes/sec', length(spike_peaks_all)/total_time);
            sprintf('');
            sprintf('PARAMETERS');
            sprintf('vm_thresh: %.0f mV  |  d2v_thresh: %.0f mV/ms²  |  min_dv: %.1f mV/ms', ...
                vm_thresh, d2v_thresh, min_dv_thresh);
        };
        
        if exist('elbow_thresh', 'var') && exist('spike_thresh', 'var')
            stats_text{end+1} = sprintf('elbow_thresh: %.0f mV  |  spike_thresh: %.0f mV', elbow_thresh, spike_thresh);
        end
        
        if ~isempty(elbow_indices)
            valid_elbow_voltages = Vm_all(elbow_indices);
            stats_text{end+1} = sprintf('');
            stats_text{end+1} = sprintf('ELBOW ANALYSIS');
            stats_text{end+1} = sprintf('Valid elbows: %.1f ± %.1f mV (range: %.1f to %.1f)', ...
                mean(valid_elbow_voltages), std(valid_elbow_voltages), ...
                min(valid_elbow_voltages), max(valid_elbow_voltages));
            
            if ~isempty(isi)
                stats_text{end+1} = sprintf('Firing rate: %.1f Hz  |  Mean ISI: %.1f ± %.1f ms', ...
                    length(elbow_indices)/(length(Vm_all)*dt), mean(isi), std(isi));
            end
            
            % NEW: Time-to-peak statistics
            if ~isempty(elbow_time_to_peak)
                mean_ttp = mean(elbow_time_to_peak);
                effective_thresh = max(time_to_peak_thresh, mean_ttp * 0.5);
                stats_text{end+1} = sprintf('Time-to-peak: %.2f ± %.2f ms (eff. thresh: %.2f)', ...
                    mean_ttp, std(elbow_time_to_peak), effective_thresh);
                if ~isempty(time_to_peak_outliers)
                    stats_text{end+1} = sprintf('Time-to-peak outliers: %d (%.1f%% of spikes)', ...
                        length(time_to_peak_outliers), 100*length(time_to_peak_outliers)/length(elbow_indices));
                end
            end
        end
        
        if ~isempty(rejected_elbows)
            rejected_elbow_voltages = Vm_all(rejected_elbows);
            stats_text{end+1} = sprintf('Rejected elbows: %.1f to %.1f mV', ...
                min(rejected_elbow_voltages), max(rejected_elbow_voltages));
        end
        
        % Display rejection reasons
        if ~isempty(rejection_reasons)
            stats_text{end+1} = sprintf('');
            stats_text{end+1} = sprintf('REJECTION REASONS');
            unique_reasons = unique(rejection_reasons);
            for i = 1:length(unique_reasons)
                count = sum(strcmp(rejection_reasons, unique_reasons{i}));
                stats_text{end+1} = sprintf('• %s (%d cases)', unique_reasons{i}, count);
            end
        end
        
        % Add performance info
        if n_valid > max_plot_spikes || size(rejected_matrix, 1) > 0
            stats_text{end+1} = sprintf('');
            stats_text{end+1} = sprintf('VISUALIZATION');
            if n_valid > max_plot_spikes
                stats_text{end+1} = sprintf('Valid waveforms: plotted %d/%d for speed', n_plot_valid, n_valid);
            end
            if ~isempty(rejected_matrix)
                stats_text{end+1} = sprintf('Rejected waveforms: plotted %d', size(rejected_matrix, 1));
            end
        end
        
        text(0.02, 0.98, stats_text, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
            'FontSize', 10, 'FontName', 'FixedWidth', 'Interpreter', 'none');
        title('Statistics & Performance', 'FontSize', 12);
        
        t_plot_elapsed = toc(t_plot);
        fprintf('Plotting time: %.0f ms\n', t_plot_elapsed*1000);
    end
    
    % --- Debug plot for low peaks ---
    if debug_low_peaks && ~isempty(rejected_peaks)
        figure('Position', [200, 200, 1000, 600]);
        
        subplot(2,1,1);
        plot(time_axis, Vm_all, 'k', 'LineWidth', 1.2); hold on;
        plot(time_axis(rejected_peaks), Vm_all(rejected_peaks), 'rx', ...
            'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Rejected Low Peaks');
        yline(min_peak_height, 'g--', 'LineWidth', 2, 'DisplayName', 'Min peak height');
        yline(vm_thresh, 'm--', 'LineWidth', 1, 'DisplayName', 'Basic thresh');
        xlabel('Time (s)');
        ylabel('Vm (mV)');
        title('Rejected Peaks Due to Low Voltage');
        legend;
        grid on;
        
        subplot(2,1,2);
        rejected_heights = Vm_all(rejected_peaks);
        histogram(rejected_heights, 'BinWidth', 2, 'FaceColor', 'r', 'EdgeColor', 'k');
        xline(min_peak_height, 'g--', 'LineWidth', 2, 'DisplayName', 'Min peak height');
        xlabel('Peak Height (mV)');
        ylabel('Count');
        title('Distribution of Rejected Peak Heights');
        legend;
        grid on;
        
        % Print some statistics
        fprintf('\n=== DEBUG: Rejected Peaks Analysis ===\n');
        fprintf('Total rejected peaks: %d\n', length(rejected_peaks));
        fprintf('Rejected peak heights: %.1f to %.1f mV\n', min(rejected_heights), max(rejected_heights));
        fprintf('Current min_peak_height threshold: %.1f mV\n', min_peak_height);
        fprintf('Suggestion: Consider adjusting min_peak_height based on the histogram\n');
    end
end

% --- Print summary ---
fprintf('\n=== Spike Detection Summary ===\n');
fprintf('Total voltage peaks found: %d\n', diagnostic_info.total_peaks_found);
fprintf('Peaks above basic threshold (%.1f mV): %d\n', vm_thresh, diagnostic_info.peaks_above_thresh);
fprintf('Peaks above height filter (%.1f mV): %d\n', min_peak_height, diagnostic_info.peaks_above_height);
fprintf('Valid elbow points detected: %d\n', length(elbow_indices));
fprintf('Rejected due to low peaks: %d\n', length(rejected_peaks));
fprintf('Rejected due to elbow criteria: %d\n', length(rejected_elbows));

if ~isempty(rejection_reasons)
    fprintf('\nRejection reasons:\n');
    for i = 1:length(rejection_reasons)
        fprintf('  %s\n', rejection_reasons{i});
    end
end
end