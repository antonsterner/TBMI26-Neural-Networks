%% This script will help you test out your kNN code
clear all; 
%% Select which data to use:

% 1 = dot cloud 1
% 2 = dot cloud 2
% 3 = dot cloud 3
% 4 = OCR data

dataSetNr = 1; % Change this to load new data 

[X, D, L] = loadDataSet( dataSetNr );

% You can plot and study dataset 1 to 3 by running:
 plotCase(X,D)

%% Select a subset of the training features

numBins = 2; % Number of Bins you want to devide your data into
numSamplesPerLabelPerBin = 100; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
selectAtRandom = true; % true = select features at random, false = select the first features

[ Xt, Dt, Lt ] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom );

% Note: Xt, Dt, Lt will be cell arrays, to extract a bin from them use i.e.
Xtrain = Xt{1};
Xtest = Xt{2};
Ltrain = Lt{1};
Ltest = Lt{2};

%% Use kNN to classify data
% Note: you have to modify the kNN() function yourselfs.

% Set the number of neighbors
k = 3;     
LkNN = kNN(Xtest, k, Xtrain, Ltrain);
    

%% Calculate The Confusion Matrix and the Accuracy
% Note: you have to modify the calcConfusionMatrix() function yourselves.

% The confusionMatrix
cM = calcConfusionMatrix( LkNN, Ltest)

% The accuracy
acc = calcAccuracy(cM)

%% Plot classifications
% Note: You do not need to change this code. 
k = 3
if dataSetNr < 4
    plotkNNResultDots(Xtest,LkNN,k,Ltest,Xtrain,Ltrain);
else
    plotResultsOCR( Xtest, Ltest, LkNN )
end
%% Cross validation

maxk = 50;
acc = zeros(1,maxk);
maxacc = zeros(1,4); 
bestk = zeros(1,4); 

for i = 1:4
    dataSetNr = i; % Change this to load new data 

    [X, D, L] = loadDataSet( dataSetNr );
    
    numBins = 2; % Number of Bins you want to devide your data into
    numSamplesPerLabelPerBin = inf; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
    selectAtRandom = true; % true = select features at random, false = select the first features

    [ Xt, Dt, Lt ] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom );

    % Note: Xt, Dt, Lt will be cell arrays, to extract a bin from them use i.e.
    Xtrain = Xt{1};
    Xtest = Xt{2};
    Ltrain = Lt{1};
    Ltest = Lt{2};
    
    for k = 1:200

        LkNN = kNN(Xtest, k, Xtrain, Ltrain);

        % Calculate The Confusion Matrix and the Accuracy
        % The confusionMatrix
        cM = calcConfusionMatrix( LkNN, Ltest);

        % The accuracy
        acc(i,k) = calcAccuracy(cM);
        if(acc(i,k) > maxacc(i)) % save best accuracy and best k
            maxacc(i) = acc(i,k);
            bestk(i) = k;
        end
    end
    
end
figure(i)
plot(1:200, acc)
legend('Dataset 1', 'Dataset 2','Dataset 3','Dataset 4')
bestk
maxacc