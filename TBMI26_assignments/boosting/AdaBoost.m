% antst719, rasst403
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
nbrHaarFeatures = 100;
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
nbrTrainExamples = 500;
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



for k = 1:T % for each weak classifier
    for i = 1:nbrHaarFeatures % for each Haar feature
        x = xTrain(i,:); 
        for j = 1:length(xTrain) % for each threshold tau
            tao = x(j); % threshold
            % classify images for each threshold
            classified = WeakClassifier(tao, p, x);
            
            % calculate weighted error of classification
            error(i,j) = WeakClassifierError(classified, weights, yTrain);
            
            % switch polarity if error > 0.5
            if(error(i,j) > 0.5)
               error(i,j) = 1-error(i,j); 
               p = -p;
               classified = -classified;
            end
            
            % save best classifier, smallest error, best threshold,
            % feature and polarity
            if(error(i,j) < minError(k))
               bestClassifier(k,:) = classified;
               minError(k) = error(i,j); 
               bestThreshold(k) = tao;
               bestFeature(k) = i;
               P(k) = p;
            end
        end
    end
    % calculate alpha
    alpha(1,k) = 0.5 * log((1-minError(1,k))/minError(1,k));
    
    % update weights, 
    % if statement to not update weights in the last iteration
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

nbrTestExamples = 3000;
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
rawClassification = zeros(T, nbrTestImg); % to be used to count the most commonly misclassified images 


for j = 1:T % test using different amount of classifiers
    for k = 1:j % for each classifier
        % use only best feature for classification
        featureIndex = bestFeature(k);
        x = xTest(featureIndex,:);
        x2 = xTrain(featureIndex,:);
        finalTestClassifier(k,:) = alpha(k)*WeakClassifier(bestThreshold(k), P(k), x);
        rawClassification(k,:) = WeakClassifier(bestThreshold(k), P(k), x); % to be used to count the most commonly misclassified images 
        finalTrainClassifier(k,:) = alpha(k)*WeakClassifier(bestThreshold(k), P(k), x2); % use previously saved classifications
    end
    % sum the columns of the classifications(each image)
    for i = 1:nbrTestImg 
        finalTestClassification(i) = sign(sum(finalTestClassifier(:,i))); % strong classifier
    end
    for i = 1:nbrTrainImg
        finalTrainClassification(i) = sign(sum(finalTrainClassifier(:,i)));
    end
    
    testWrongClass = (finalTestClassification ~= yTest); % all indeces that differ, returns 1 if diff
    testError(j) = sum(testWrongClass/nbrTestImg);
    testAccuracy(j) = 1 - testError(j);
        
    trainWrongClass = (finalTrainClassification ~= yTrain);
    trainError(j) = sum(trainWrongClass/nbrTrainImg);
    trainAccuracy(j) = 1 - trainError(j);
    
end
testTime = toc

% count the most commonly misclassified images
wrong = zeros(T,nbrTestImg);
for i = 1:T
   wrong(i,:) = (rawClassification(i,:) ~= yTest); 
end

[nbrWrongClassifications, wrongID] = sort(sum(wrong),'descend'); 

% save max accuracy, and the number of weak classifiers used
[max_accuracy, nbrWeakClassifiers] = max(testAccuracy);

%% Plot the accuracy of the strong classifier as function of the number of weak classifiers.
%  Note: you can find this accuracy without re-training with a different
%  number of weak classifiers.
nc = 1:T;
figure(4);
plot(nc, testAccuracy);
hold on
plot(nc, trainAccuracy);
legend('Test Accuracy', 'Training Accuracy')
xlabel('Number of classifiers')
ylabel('Accuracy')

% plot 25 most commonly misclassified images 
figure(5);
colormap gray;
for k=1:25
    if(wrongID(k) < nbrTestExamples)
        subplot(5,5,k), imagesc(faces(:,:,wrongID(k)));
    else
        subplot(5,5,k), imagesc(nonfaces(:,:,wrongID(k)));
    end
    axis image;
    axis off;
end