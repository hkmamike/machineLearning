function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

sumVal = 0;
for i=1:m
    xRow = X(i, :);
    thetaTrans = theta';
    h = sum(xRow .* thetaTrans);
    sumVal = sumVal + ( h - y(i) ) ^ 2;
end

J = (0.5 / m) * sumVal;

% adding regularization, not including const parameter
for i=2:n
    J = J + (lambda/(2*m)) * theta(i)^2;
end

% gradient
for i=1:n
    for j=1:m
        xRow = X(j,:);
        thetaTrans = theta';
        h = sum(xRow .* thetaTrans);
        grad(i) = grad(i) + (1/m) * (h - y(j)) * X(j, i);
    end

    if (i != 1)
        grad(i) = grad(i) + lambda * theta(i) / m;
    end
end

% =========================================================================

grad = grad(:);

end
