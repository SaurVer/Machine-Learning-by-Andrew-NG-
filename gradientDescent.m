function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
lol=reshape(X',[1,2,m,1]);

for iter = 1:num_iters,
  summ=[0;0];
  for i=1:m,
    
    summ=summ+((lol(:,:,i)*theta)-y(i))*(lol(:,:,i)');
    %disp(i);
    %disp(summ);
    %disp((lol(:,:,i)'));
    
  end;
 
    theta=theta-(alpha/m)*summ;
    %disp(theta);
    
 
  
 
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %







    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    

end;

end;
