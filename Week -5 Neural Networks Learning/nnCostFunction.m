function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
X=[ones(m, 1) X];
a=[1:size(Theta2,1)];
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

for i=1:m,
  for K=1:size(Theta2,1),
    o=a==y(i);
    O=o(K);
    a2=sigmoid(Theta1*X(i,:)');
    
    a2=[1;a2];
    
    a3=sigmoid(Theta2*a2);
    
    J=J+(-O*log(a3(K))-(1-O)*log(1-a3(K)));
  end;
end;
J=(J/m);
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%with regularized terms
[j k]=size(Theta1);
T1=0;
for Z=1:j,
  for K=1:k-1,
    T1=T1+Theta1(Z,K+1)^2;
  end;
end;

T2=0;
[j k]=size(Theta2);

for Z=1:j,
  for K=1:k-1,
    T2=T2+Theta2(Z,K+1)^2;
  end;
end;
J=J+(lambda/(2*m))*(T1+T2);
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
Z2= Theta1*X';
A2=sigmoid(Z2);
A2=[(ones(size(A2,2), 1))'; A2];
Z3= Theta2*A2;
A3=sigmoid(Z3);
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%             first time.
big=zeros(size(Theta2,1),m);
for i=1:m,
  big(:,i)=a==y(i);
  
end;
DELTA3=A3-big;
Theta22=Theta2(:,2:end);
DELTA2=(Theta22'*DELTA3).*sigmoidGradient(Z2);

  BIG_DEL2=zeros(size(Theta2,1), size(Theta2,2));
  %A2=A2(2:end,:);
for i=1:size(X,1);
BIG_DEL2=BIG_DEL2+DELTA3(:,i)*A2(:,i)';
end;



BIG_DEL1=zeros(size(Theta1,1), size(Theta1,2));
  
for i=1:size(X,1);
BIG_DEL1=BIG_DEL1+DELTA2(:,i)*X(i,:);
end;
Theta1_grad=BIG_DEL1./m;
Theta2_grad=BIG_DEL2./m;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad=[BIG_DEL1(:,1)./m  (BIG_DEL1(:,2:end)./m+lambda*Theta1(:,2:end)./m)];


Theta2_grad=[BIG_DEL2(:,1)./m  (BIG_DEL2(:,2:end)./m+lambda*Theta2(:,2:end)./m)];

%A2=A2';
%A3=A3';
%DELTA2=DELTA2';
%DELTA3=DELTA3';












% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
