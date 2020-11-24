
% Chloe Quinto 
% CS 583 Deep Learning 
% HW 2 Evaluation and Error Analysis  


function main()
    clear all;
    
    diary diaryFile

    [xtrain,~,ytrain] = digitTrain4DArrayData; 
    [xvalid, ~, yvalid] = digitTest4DArrayData; 
    
    checkNormalization(ytrain); 

    layers = neuralNetwork();
    
    net = trainNN(xtrain, ytrain, layers, xvalid, yvalid);
    
    testNN(net, xvalid, yvalid);
    
    worstPredicted(net, xvalid, yvalid);
    
    diary off
end 


function checkNormalization(ytrain)
    % Plots rotation angle of ytrain samples to find 
    fig = figure; 
    histogram(ytrain)
    axis tight 
    ylabel("Counts")
    xlabel("Rotation Angle")
    saveas(fig, "reports/normalization.png")
    
end


function layers =  neuralNetwork()
    layers = [
        imageInputLayer([28 28 1]) % 28 x 28 images with 1 channel color 

        convolution2dLayer(3,8,'Padding','same')
        batchNormalizationLayer
        reluLayer

        averagePooling2dLayer(2,'Stride',2)

        convolution2dLayer(3,16,'Padding','same')
        batchNormalizationLayer
        reluLayer

        averagePooling2dLayer(2,'Stride',2)

        convolution2dLayer(3,32,'Padding','same')
        batchNormalizationLayer
        reluLayer

        convolution2dLayer(3,32,'Padding','same')
        batchNormalizationLayer
        reluLayer

        dropoutLayer(0.2)
        fullyConnectedLayer(1)
        regressionLayer
    ];
end 

function net = trainNN(xtrain, ytrain, layers, xvalid, yvalid)
    miniBatchSize = 128; 
    validationFrequency = floor(numel(ytrain)/miniBatchSize); 
    options = trainingOptions('sgdm', ...
        'MiniBatchSize',miniBatchSize, ...
        'MaxEpochs',30, ...
        'InitialLearnRate',1e-3, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropFactor',0.1, ...
        'LearnRateDropPeriod',20, ...
        'Shuffle','every-epoch', ...
        'ValidationData',{xvalid,yvalid}, ...
        'ValidationFrequency',validationFrequency, ...
        'Plots','training-progress', ...
        'Verbose',false);
    net = trainNetwork(xtrain,ytrain,layers,options);
end 


function testNN(net, xvalid, yvalid)
    ypredict = predict(net, xvalid); 
    predictionErr = yvalid - ypredict; 

    thr = 10; 
    numCorrect = sum(abs(predictionErr) < thr); 
    numValidationImages = numel(yvalid); 
    
    accuracy = numCorrect/numValidationImages; 
    display(accuracy); 
end 

function worstPredicted(net, xvalid, yvalid)
    ypredict = predict(net, xvalid); 
    predictionErr = yvalid - ypredict; 

    [~, sortIDX] = sort(predictionErr(:), 'descend'); % sort errors in descending order 
    maxIDX = sortIDX(1:5); % get top 5 
    display(maxIDX);
    
    fig2 = figure;
    for i = 1:numel(maxIDX)
 
        subplot(1, 5, i)
        imshow(xvalid(:,:,:,maxIDX(i)))
        drawnow 
    end 
    saveas(fig2, "reports/worst5predictions.png")
    
end 