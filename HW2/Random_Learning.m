[XTrain,~,YTrain] = digitTrain4DArrayData;
[XValidation,~,YValidation] = digitTest4DArrayData;


figure
histogram(YTrain)
axis tight
ylabel('Counts')
xlabel('Rotation Angle')
saveas(figure, "nornalization.png")

layers = [
    imageInputLayer([28 28 1])
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

while(true) 
    YPredicted = predict(net,XValidation);
    predictionError = YValidation - YPredicted;
    if predictionError < 0.05
        break;
    else
        net = replaceWeights(net);
    end
end


thr = 10;
numCorrect = sum(abs(predictionError) < thr);
numValidationImages = numel(YValidation);

accuracy = numCorrect/numValidationImages;
display(accuracy);

squares = predictionError.^2;
rmse = sqrt(mean(squares));





% this function is referenced from MATBAL Answers.
function newNet = replaceWeights(oldNet)
oldLgraph = layerGraph(oldNet);
layers = oldLgraph.Layers;
connections = oldLgraph.Connections;
% Set new weights
layerID = 4.*rand(1);
newWeights = rand(1);
layers(layerID).Weights = newWeights;
% Freeze weights, from the Matlab transfer learning example
for ii = 1:size(layers,1)
    props = properties(layers(ii));
    for p = 1:numel(props)
        propName = props{p};
        if ~isempty(regexp(propName, 'LearnRateFactor$', 'once'))
            layers(ii).(propName) = 0;
        end
    end
end
% Build new lgraph, from the Matlab transfer learning example
newLgraph = layerGraph();
for i = 1:numel(layers)
    newLgraph = addLayers(newLgraph,layers(i));
end
for c = 1:size(connections,1)
    newLgraph = connectLayers(newLgraph,connections.Source{c},connections.Destination{c});
end
% Very basic options
options = trainingOptions('sgdm','MaxEpochs', 1);
% Note that you might need to change the label here depending on your
% network in my case '1' is a valid label.
newNet = trainNetwork(zeros(oldNet.Layers(1).InputSize),1,newLgraph,options);
end