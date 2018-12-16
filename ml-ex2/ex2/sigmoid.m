function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
height = size(z)(1);
width = size(z)(2);
g = zeros([height, width]);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

for i=1:height,
    for j=1:width,
        g(i,j) = 1 / (1 + exp(-1*z(i,j)));
end;

% =============================================================

end
