function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
J = (0.5/m)*((X*theta-y)'*(X*theta-y))+(lambda/(2*m) )*(sum(theta.*theta)-theta(1)*theta(1));
[o p]=size(theta);
grad = zeros(o-1,p);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
a=X(:,2:end);
[row1 col1]=size(a);
[row col]=size(X);
lol1=reshape(a',[1,col1,m,1]);
lol=reshape(X',[1,col,m,1]);
grad0=0;
for i=1:m,
 grad = grad+(1/m)*((lol(:,:,i)*theta)-y(i))*(lol1(:,:,i))';
 grad0=grad0+(1/m)*((lol(:,:,i)*theta)-y(i))*ones(1);
end;

grad=grad+(lambda/m)*theta(2:end,:);
grad=[grad0 ;grad];


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
