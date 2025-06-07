function err = SpikeGenWrapper(coef)
    
    global fitParameters
    
    params.threshold = coef(1);
    params.absRefract = fitParameters.absRefract;
    params.refractHeight = coef(2);
    params.refractTau = abs(coef(3));
    params.refractHeightS = coef(4);
    params.refractTauS = abs(coef(5));
    params.refractHeightS2 = coef(6);
    params.refractTauS2 = abs(coef(7));
        
    predSpikeCount = 0;
    measSpikeCount = 0;
    
    for epoch = 1:size(fitParameters.Prediction, 1)
        subThreshVolt = fitParameters.Prediction(epoch, :);
        predictTimes = predictSpikeTimes(subThreshVolt, length(subThreshVolt), params.threshold, params.absRefract, params.refractHeight, params.refractTau, params.refractHeightS, params.refractTauS, params.refractHeightS2, params.refractTauS2); 
%        predictTimes = predictSpikeTimes(params); 
        fitParameters.predSpikeTimes{epoch} = predictTimes;
        measSpikeTimes = fitParameters.DecimatedSpikeTimes{epoch};
        if (fitParameters.Verbose)
            randSpikeTimes = randi(size(fitParameters.Prediction, 2), length(predictTimes));
            SpikeDistRand(epoch) = spkd_c(randSpikeTimes, measSpikeTimes, length(randSpikeTimes), length(measSpikeTimes), fitParameters.cost);
%            SpikeDistRand(epoch) = spkd(randSpikeTimes, measSpikeTimes, fitParameters.cost);
        end
        SpikeDist(epoch) = spkd_c(predictTimes, measSpikeTimes, length(predictTimes), length(measSpikeTimes), fitParameters.cost);
%        SpikeDist(epoch) = spkd(predictTimes, measSpikeTimes, fitParameters.cost);
        predSpikeCount = predSpikeCount + length(predictTimes);
        measSpikeCount = measSpikeCount + length(measSpikeTimes);
    end
    err = mean(SpikeDist);
    if (fitParameters.Verbose)
        errRand = mean(SpikeDistRand);
        fprintf(1, 'count = %d %d distance = %d %d (%d)\n', measSpikeCount, predSpikeCount, err, errRand, err/errRand);
    else
%        fprintf(1, '%d %d %d\n', measSpikeCount, predSpikeCount, err);
    end
end
        
