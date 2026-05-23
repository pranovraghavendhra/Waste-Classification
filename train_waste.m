
clear; close all; clc;


datasetFolder = fullfile(pwd,'dataset'); 
imgSize = [224 224];                     
valSplit = 0.15;                          
testSplit = 0.10;                        
miniBatchSize = 32;
maxEpochs = 10;                          
initialLearnRate = 1e-4;


imds = imageDatastore(datasetFolder, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

tbl = countEachLabel(imds)

numImages = numel(imds.Files);

[imdsTrain, imdsRemain] = splitEachLabel(imds, 1 - (valSplit+testSplit), 'randomized');
[imdsVal, imdsTest] = splitEachLabel(imdsRemain, valSplit/(valSplit+testSplit), 'randomized');

imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-20 20], ...
    'RandXTranslation',[-10 10], ...
    'RandYTranslation',[-10 10], ...
    'RandXScale',[0.9 1.1], ...
    'RandYScale',[0.9 1.1], ...
    'RandXReflection',true);

augTrain = augmentedImageDatastore(imgSize, imdsTrain, 'DataAugmentation', imageAugmenter);
augVal   = augmentedImageDatastore(imgSize, imdsVal);
augTest  = augmentedImageDatastore(imgSize, imdsTest);

net = resnet18();                       
lgraph = layerGraph(net);

numClasses = numel(categories(imdsTrain.Labels));

if ismember('fc1000', {lgraph.Layers.Name})
    lgraph = removeLayers(lgraph, {'fc1000','prob','ClassificationLayer_predictions'});
else
   
    n = numel(lgraph.Layers);
    removeNames = {lgraph.Layers(n-2).Name, lgraph.Layers(n-1).Name, lgraph.Layers(n).Name};
    lgraph = removeLayers(lgraph, removeNames);
end

newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc_custom','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax_custom')
    classificationLayer('Name','classoutput')];
lgraph = addLayers(lgraph, newLayers);


lastLayer = lgraph.Layers(end-numel(newLayers)).Name; 
lgraph = connectLayers(lgraph, lastLayer, 'fc_custom');

options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',maxEpochs, ...
    'InitialLearnRate',initialLearnRate, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augVal, ...
    'ValidationFrequency',floor(numel(imdsTrain.Files)/miniBatchSize), ...
    'Verbose',true, ...
    'Plots','training-progress');


[trainedNet, info] = trainNetwork(augTrain, lgraph, options);


preds = classify(trainedNet, augTest);
YTest = imdsTest.Labels;
acc = mean(preds == YTest);
fprintf('Test accuracy: %.2f%%\n', acc*100);

figure; plotconfusion(YTest, preds); title('Test Confusion Matrix');

modelFile = fullfile(pwd, 'wasteNet_resnet18.mat');
save(modelFile, 'trainedNet', 'info', 'imds', 'imdsTrain','imdsVal','imdsTest');
fprintf('Model saved to %s\n', modelFile);
