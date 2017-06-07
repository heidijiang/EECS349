function pred = logreg(trainX,trainY,testX)


lambdas = 0:.05:1;
for i = 1:length(lambdas)
    [model, llh] = logitBin(trainX, trainY, lambdas(i));
    [y, p] = logitBinPred(model,trainX);
    precision(i) = sum(and(y==1,trainY==1))/(sum(and(y==1,trainY==0))+sum(and(y==1,trainY==1)));
end

[~,id] = max(precision);
%%
[model, llh] = logitBin(trainX, trainY, lambdas(id));
[pred, p] = logitBinPred(model,testX);



%%
function [model, llh] = logitBin(X, y, lambda, eta)
% Logistic regression for binary classification optimized by Newton-Raphson method.
% Input:
%   X: d x n data matrix
%   z: 1 x n label (0/1)
%   lambda: regularization parameter
%   eta: step size
% Output:
%   model: trained model structure
%   llh: loglikelihood
% Written by Mo Chen (sth4nth@gmail.com).
if nargin < 4
    eta = 1e-1;
end
if nargin < 3
    lambda = 1e-4;
end
X = [X; ones(1,size(X,2))];
[d,n] = size(X);
tol = 1e-4;
epoch = 200;
llh = -inf(1,epoch);
h = 2*y-1;
w = rand(d,1);
for t = 2:epoch
    a = w'*X;
    llh(t) = -(sum(log1pexp(-h.*a))+0.5*lambda*dot(w,w))/n; % 4.89
    if llh(t)-llh(t-1) < tol; break; end
    z = sigmoid(a);                     % 4.87
    g = X*(z-y)'+lambda*w;              % 4.96
    r = z.*(1-z);                       % 4.98
    Xw = bsxfun(@times, X, sqrt(r));
    H = Xw*Xw'+lambda*eye(d);           % 4.97
    w = w-eta*(H\g); 
end
llh = llh(2:t);
model.w = w;
end

function [y, p] = logitBinPred(model, X)
% Prediction of binary logistic regression model
% Input:
%   model: trained model structure
%   X: d x n testing data
% Output:
%   y: 1 x n predict label (0/1)
%   p: 1 x n predict probability [0,1]
% Written by Mo Chen (sth4nth@gmail.com).
X = [X;ones(1,size(X,2))];
w = model.w;
p = sigmoid(w'*X);
y = round(p);
end

function y = log1pexp(x)
% Accurately compute y = log(1+exp(x))
% reference: Accurately Computing log(1-exp(|a|)) Martin Machler
seed = 33.3;
y = x;
idx = x<seed;
y(idx) = log1p(exp(x(idx)));
end

end
