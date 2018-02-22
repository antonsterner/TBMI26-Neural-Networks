function [Wout, trainingError, testError ] = trainSingleLayer(Xtrain,Dtrain,Xtest,Dtest, W0,numIterations, learningRate )
%TRAINSINGLELAYER Trains the network (Learning)
%   Inputs:
%               X* - Training/test features (matrix)
%               D* - Training/test desired output of net (matrix)
%               W0 - Weights of the neurons (matrix)
%
%   Output:
%               Wout - Weights after training (matrix)
%               trainingError - The training error for each iteration
%                               (vector)
%               testError - The test error for each iteration
%                               (vector)

% Initiate variables
trainingError = nan(numIterations+1,1);
testError = nan(numIterations+1,1);
Ntrain = size(Xtrain,2);
Ntest = size(Xtest,2);
Wout = W0;
% Calculate initial error
Ytrain = runSingleLayer(Xtrain, W0);
Ytest = runSingleLayer(Xtest, W0);
trainingError(1) = sum(sum((Ytrain - Dtrain).^2))/Ntrain; 
testError(1) = sum(sum((Ytest - Dtest).^2))/Ntest;

for n = 1:numIterations

    Y = Wout*Xtrain;
    
    grad_w = 2/Ntrain * (Y-Dtrain) * Xtrain'; % Minimization algorithm
    
    Wout = Wout - learningRate * grad_w; % gradient descent
    trainingError(1+n) = sum(sum((Wout*Xtrain - Dtrain).^2))/Ntrain;
    testError(1+n) = sum(sum((Wout*Xtest - Dtest).^2))/Ntest;
end
end

