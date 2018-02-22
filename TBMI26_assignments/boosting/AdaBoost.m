% Load face and non-face data and plot a few examples
load faces;
load nonfaces;
faces = double(faces);
nonfaces = double(nonfaces);

figure(1);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(faces(:,:,10*k));
    axis image;
    axis off;
end

figure(2);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(nonfaces(:,:,10*k));
    axis image;
    axis off;
end

% Generate Haar feature masks
nbrHaarFeatures = 25;
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);

figure(3);
colormap gray;
for k = 1:25
    subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2]);
    axis image;
    axis off;
end

% Create a training data set with a number of training data examples
% from each class. Non-faces = class label y=-1, faces = class label y=1
nbrTrainExamples = 1000;
trainImages = cat(3,faces(:,:,1:nbrTrainExamples),nonfaces(:,:,1:nbrTrainExamples));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks); % result of Haar features applied to images
yTrain = [ones(1,nbrTrainExamples), -ones(1,nbrTrainExamples)]; % correct labels, 1 for face, -1 for non face

%% Implement the AdaBoost training here
% Use your implementation of WeakClassifier and WeakClassifierError
tic
p = 1; % polarity
T = 50; % number of weak classifiers 
nbrTrainImg = nbrTrainExamples*2;
classified = zeros(1,nbrTrainImg);
alpha = zeros(1,T);
error = ones(nbrHaarFeatures,nbrTrainImg);
weights = ones(1,nbrTrainImg)./(nbrTrainImg);
minError = ones(1,T);
bestThreshold = ones(1,T);
P = ones(1,T);
bestFeature = ones(1,T);
bestClassifier = ones(T,nbrTrainImg);
wthreshold = 5/(nbrTrainImg);

% Find the best weak classifier among all features and all thresholds,
% update weights and repeat
% for each weak classifier
%   for each Haar feature
%       for each threshold tau
for k = 1:T
    for i = 1:nbrHaarFeatures
        x = xTrain(i,:);
        for j = 1:length(xTrain)
            tau = x(j);    
            % classify images for each threshold
            classified = WeakClassifier(tau, p, x);
            
            % calculate error of classification
            %wsize = size(weights(i,:,k));
            error(i,j) = WeakClassifierError(classified, weights, yTrain);
            
            % switch polarity if error > 0.5
            if(error(i,j) > 0.5)
               error(i,j) = 1-error(i,j); 
               p = -p;
               classified = -classified;
            end
            
            % save best classifier
            if(error(i,j) < minError(k))
               bestClassifier(k,:) = classified;
               minError(k) = error(i,j); 
               bestThreshold(k) = tau;
               bestFeature(k) = i;
               P(k) = p;
            end
        end
    end
    % calculate alpha
    alpha(1,k) = 0.5 * log((1-minError(1,k))/minError(1,k));
    
    % update weights, 
    % if statement to not exceed matrix dimensions of weights
    if(k < T)
        weights = weights.*exp(alpha(k)*-(yTrain.*bestClassifier(k,:)));
        % trim weights exceeding 5 times original weight
        %size(weights)
        weights(weights > wthreshold) = wthreshold; 
        % renormalize weights
        %size(weights)
        weights = weights./sum(weights);
    end
end
trainingTime = toc
%% Extract test data

nbrTestExamples = 2000;
nbrTestImg = nbrTestExamples*2;

testImages  = cat(3,faces(:,:,(nbrTrainExamples+1):(nbrTrainExamples+nbrTestExamples)),...
                    nonfaces(:,:,(nbrTrainExamples+1):(nbrTrainExamples+nbrTestExamples)));
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks);
yTest = [ones(1,nbrTestExamples), -ones(1,nbrTestExamples)];

%% Evaluate your strong classifier here
%  You can evaluate on the training data if you want, but you CANNOT use
%  this as a performance metric since it is biased. You MUST use the test
%  data to truly evaluate the strong classifier.
tic
finalTrainClassifier = zeros(T, nbrTrainImg);
finalTrainClassification = zeros(1, nbrTrainImg);
trainError = zeros(1,T);
trainAccuracy = zeros(1,T);

finalTestClassifier = zeros(T, nbrTestImg);
finalTestClassification = zeros(1, nbrTestImg);
testError = zeros(1,T);
testAccuracy = zeros(1,T);

for j = 1:T % test using different amount of classifiers
    for k = 1:j % for each classifier
        % use only best feature for classification
        featureIndex = bestFeature(k);
        x = xTest(featureIndex,:);
        x2 = xTrain(featureIndex,:);
        finalTestClassifier(k,:) = alpha(k)*WeakClassifier(bestThreshold(k), P(k), x);
        finalTrainClassifier(k,:) = alpha(k)*WeakClassifier(bestThreshold(k), P(k), x2); % use previously saved classifications
    end
    % sum the columns of the classifications(each image)
    for i = 1:nbrTestImg 
        finalTestClassification(i) = sign(sum(finalTestClassifier(:,i)));
        
    end
    for i = 1:nbrTrainImg
        finalTrainClassification(i) = sign(sum(finalTrainClassifier(:,i)));
    end

    testError(j) = sum(abs(finalTestClassification(finalTestClassification ~= yTest)/nbrTestImg));
    testAccuracy(j) = 1 - testError(j);
    trainError(j) = sum(abs(finalTrainClassification(finalTrainClassification ~= yTrain)/nbrTrainImg));
    trainAccuracy(j) = 1 - trainError(j);
end
testTime = toc
%% Plot the error of the strong classifier as function of the number of weak classifiers.
%  Note: you can find this error without re-training with a different
%  number of weak classifiers.
nc = 1:T;
figure(4);
plot(nc, testAccuracy);
hold on
plot(nc, trainAccuracy);

