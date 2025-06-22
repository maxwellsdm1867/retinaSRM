classdef VPLossUtils
    % VPLossUtils - Utility class for Victor-Purpura loss calculations
    %
    % This class provides static methods for computing Victor-Purpura loss
    % functions used in spike train analysis.
    
    methods(Static)
        function loss = compute_fast_vp_loss_exponential(params, model, q)
            % COMPUTE_FAST_VP_LOSS_EXPONENTIAL - Compute VP loss with exponential model
            %
            % Inputs:
            %   params - Model parameters
            %   model - SRM model instance
            %   q - VP distance parameter
            %
            % Returns:
            %   loss - Computed VP loss value
            
            % Extract parameters
            tau_m = params(1);
            eta = params(2);
            v_reset = params(3);
            
            % Update model parameters
            model.tau_m = tau_m;
            model.eta = eta;
            model.v_reset = v_reset;
            
            % Predict spike train
            [predicted_spikes, ~] = model.predict_spike_train_exponential();
            
            % Compute VP distance
            loss = spkd_c(model.true_spike_times, predicted_spikes, length(model.true_spike_times), length(predicted_spikes), q);
        end
        
        function loss = compute_fast_vp_loss_linexp(params, model, q)
            % COMPUTE_FAST_VP_LOSS_LINEXP - Compute VP loss with linear-exponential model
            %
            % Inputs:
            %   params - Model parameters
            %   model - SRM model instance
            %   q - VP distance parameter
            %
            % Returns:
            %   loss - Computed VP loss value
            
            % Extract parameters
            tau_m = params(1);
            eta = params(2);
            v_reset = params(3);
            alpha = params(4);
            
            % Update model parameters
            model.tau_m = tau_m;
            model.eta = eta;
            model.v_reset = v_reset;
            model.alpha = alpha;
            
            % Predict spike train
            [predicted_spikes, ~] = model.predict_spike_train_linexp();
            
            % Compute VP distance
            loss = spkd_c(model.true_spike_times, predicted_spikes, q);
        end
    end
end 