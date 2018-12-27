function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% cost function
for i=1:m
    xRow = X(i,:);
    thetaTrans = theta';
    h = sigmoid(sum(xRow .* thetaTrans));
    firstTerm = -1* y(i) * log(h);
    secondTerm = (1 - y(i)) * log(1-h);
    J = J + (1/m) * (firstTerm - secondTerm);
end;

% adding regularization, not including const parameter
for i=2:n
    J = J + (lambda/(2*m)) * theta(i)^2;
end



% gradient
for i=1:n
    for j=1:m
        xRow = X(j,:);
        thetaTrans = theta';
        h = sigmoid(sum(xRow .* thetaTrans));
        grad(i) = grad(i) + (1/m) * (h - y(j)) * X(j, i);
    end

    if (i != 1)
        grad(i) = grad(i) + lambda * theta(i) / m;
    end
end;

% adding regularization

% =============================================================

end
