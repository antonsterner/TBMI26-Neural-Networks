function [ labelsOut ] = kNN(X, k, Xt, Lt)
%KNN Your implementation of the kNN algorithm
%   Inputs:
%               X  - Features to be classified
%               k  - Number of neighbors
%               Xt - Training features
%               LT - Correct labels of each feature vector [1 2 ...]'
%
%   Output:
%               LabelsOut = Vector with the classified labels

labelsOut  = zeros(size(X,2),1);
classes = unique(Lt);
numClasses = length(classes);
%distance = zeros(size(X,2),1);
%dist1 = dist(X,Xt)

% Find k nearest neighbors of new points X, euclidean distance to the k-nearest points
for i = 1:length(X(1,:))
    
   distance = sqrt(sum(Xt(1,:)-X(1,i)).^2 + (Xt(2,:) - X(2,i)).^2); 
   
   [~,I] = sort(distance); % sort by distance
   % if the vote is even between classes, choose the class of the nearest
   % point 
   labelsOut(i) = mode(Lt(I(1:k)));
end

end

