%% This script will help you test out your single layer neural network code
clear all;
%% Select which data to use:

% 1 = dot cloud 1
% 2 = dot cloud 2
% 3 = dot cloud 3
% 4 = OCR data

dataSetNr = 1; % Change this to load new data 

[X, D, L] = loadDataSet( dataSetNr );

%% Select a subset of the training features

numBins = 2; % Number of Bins you want to devide your data into
numSamplesPerLabelPerBin = 100; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
selectAtRandom = true; % true = select features at random, false = select the first features

[ Xt, Dt, Lt ] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom );

% Note: Xt, Dt, Lt will be cell arrays, to extract a bin from them use i.e.
XBin1 = Xt{1};
XBin2 = Xt{2};
%% Modify the X Matrices so that a bias is added

% The Training Data
Xtraining = [ones(1,length(XBin1)); XBin1];

% The Test Data
Xtest = [ones(1,length(XBin1)); XBin2];


%% Train your single layer network
% Note: You nned to modify trainSingleLayer() in order to train the network
[n_rows, n_cols] = size(Xtraining); % number of input variables, (I+1)*O = (2+1)*2
[Dn_rows, Dn_cols] = size(D);
numIterations = 5000; % Change this, Numner of iterations (Epochs)
learningRate = 0.001; % Change this, Your learningrate
W0 = -0.1 + (0.1-(-0.1)).*rand(Dn_rows, n_rows); % Initiate the weight matrix W, values in range [0,0.01]
%%
[W, trainingError, testError ] = trainSingleLayer(Xtraining,Dt{1},Xtest,Dt{2}, W0,numIterations, learningRate );

% Plot errors
figure(1101)
clf
[mErr, mErrInd] = min(testError);

plot(trainingError,'k','linewidth',1.5)
%axis([0 10 0 10])
hold on
plot(testError,'r','linewidth',1.5)
plot(mErrInd,mErr,'bo','linewidth',1.5)
hold off
title('Training and Test Errors, Single Layer')
legend('Training Error','Test Error','Min Test Error')

%% Calculate The Confusion Matrix and the Accuracy of the Evaluation Data
% Note: you have to modify the calcConfusionMatrix() function yourselfs.

[ Y, LSingleLayerTraining ] = runSingleLayer(Xtraining, W);
[ Y, LSingleLayerTest ] = runSingleLayer(Xtest, W);

% The confucionMatrix
cM = calcConfusionMatrix( LSingleLayerTest, Lt{2})

% The accuracy
acc = calcAccuracy(cM)

%% Plot classifications
% Note: You do not need to change this code.

if dataSetNr < 4
    plotResultSingleLayer(W,Xtraining,Lt{1},LSingleLayerTraining,Xtest,Lt{2},LSingleLayerTest)
else
    plotResultsOCR( Xtest, Lt{2}, LSingleLayerTest)
end
