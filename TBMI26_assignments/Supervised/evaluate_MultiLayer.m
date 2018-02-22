%% This script will help you test out your single layer neural network code

%% Select which data to use:

% 1 = dot cloud 1
% 2 = dot cloud 2
% 3 = dot cloud 3
% 4 = OCR data

dataSetNr = 3; % Change this to load new data 

[X, D, L] = loadDataSet( dataSetNr );

%% Select a subset of the training features
% select few training samples to get a non-generalized model 

numBins = 2; % Number of Bins you want to devide your data into
numSamplesPerLabelPerBin = 5; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
selectAtRandom = true; % true = select features at random, false = select the first features

[ Xt, Dt, Lt ] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom );

% Note: Xt, Dt, Lt will be cell arrays, to extract a bin from them use i.e.
trainBin = Xt{1};
testBin = Xt{2};

% Pick maximum number of test samples 
numSamplesPerLabelPerBin = inf; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)

[ Xt2, Dt2, Lt2 ] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom );

trainBin2 = Xt2{1};
testBin2 = Xt2{2};
%% Modify the X Matrices so that a bias is added

% The Training Data
Xtraining = [ones(1,length(trainBin)); trainBin];

% The Test Data
Xtest = [ones(1,length(testBin2)); testBin2];


%% Train your single layer network
% Note: You need to modify trainSingleLayer() in order to train the network

% Numbers resulting in satisfactory accuracy
% Dataset 1: numHidden = 5, numIterations = 20000, learningRate = 0.005 
% Accuracy = 0.995
% Dataset 2: numHidden = 5, numIterations = 20000, learningRate = 0.005
% Accuracy = 0.998
% Dataset 3: numHidden = 10, numIterations = 20000, learningRate = 0.005
% Accuracy = 0.99667
% Dataset 4: numHidden = 50, numIterations = 13000, learningRate = 0.01
% Accuracy = 0.96606
[n_rows, ~] = size(Xtraining);
[Dn_rows, ~] = size(D);
numHidden = 2; % Number of hidden neurons 
numIterations = 1000; % Change this, Number of iterations (Epochs)
learningRate = 0.05; % Change this, Your learningrate
W0 = -0.1 + (0.1-(-0.1)).*rand(numHidden, n_rows); % Change this, Initiate your weight matrix W, (I+1)*H1 + (H1+1)*O 
V0 = -0.1 + (0.1-(-0.1)).*rand(Dn_rows, numHidden); % Change this, Initiate your weight matrix V
%%
%
tic
[W,V, trainingError, testError ] = trainMultiLayer(Xtraining,Dt{1},Xtest,Dt2{2}, W0,V0,numIterations, learningRate );
trainingTime = toc;
%% Plot errors
figure(1101)
clf
[mErr, mErrInd] = min(testError);
plot(trainingError,'k','linewidth',1.5)
hold on
plot(testError,'r','linewidth',1.5)
plot(mErrInd,mErr,'bo','linewidth',1.5)
hold off
title('Training and Test Errors, Multi-Layer')
legend('Training Error','Test Error','Min Test Error')

%% Calculate The Confusion Matrix and the Accuracy of the Evaluation Data
% Note: you have to modify the calcConfusionMatrix() function yourselfs.

[ Y, LMultiLayerTraining ] = runMultiLayer(Xtraining, W, V);
tic
[ Y, LMultiLayerTest ] = runMultiLayer(Xtest, W,V);
classificationTime = toc/length(Xtest);
% The confucionMatrix
cM = calcConfusionMatrix( LMultiLayerTest, Lt2{2});

% The accuracy
acc = calcAccuracy(cM);

display(['Time spent training: ' num2str(trainingTime) ' sec'])
display(['Time spent classifying 1 feature vector: ' num2str(classificationTime) ' sec'])
display(['Accuracy: ' num2str(acc)])

%% Plot classifications
% Note: You do not need to change this code.

if dataSetNr < 4
    plotResultMultiLayer(W,V,Xtraining,Lt{1},LMultiLayerTraining,Xtest,Lt2{2},LMultiLayerTest)
else
    plotResultsOCR( Xtest, Lt{2}, LMultiLayerTest )
end
