function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_values = [0.001 0.003 0.01 0.03 0.1 0.3 1 3 10 30 100 300];
sigma_values = [0.001 0.003 0.01 0.03 0.1 0.3 1 3 10 30 100 300];

val = [-1 -1];
err = Inf(1);
for i=1:length(C_values)
    for j=1:length(sigma_values)
        C = C_values(i);
        sigma = sigma_values(j);
        model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma), 1e-3, 20);
        predictions = svmPredict(model, Xval);
        present_error = mean(double(predictions ~= yval));
        if(present_error<=err)
            val(1) = i;
            val(2) = j;
            err = present_error;
        end
    end
end

C = C_values(val(1));
sigma = sigma_values(val(2));




% =========================================================================

end
