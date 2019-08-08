function [Z,E] = LRR_l1(X,A,lambda,display)
% This routine uses Inexact ALM algorithm to solve the following nuclear-norm optimization problem:
% min |Z|_*+lambda*|E|_1
% s.t., X = AZ+E
% inputs:
%        X -- D*N data matrix, D is the data dimension, and N is the number
%             of data vectors.
%        A -- D*M matrix of a dictionary, M is the size of the dictionary
global optdata
if nargin<4
    display = 1;
end

tol = 5e-6;
maxIter = 1e3;
[d n] = size(X);
m = size(A,2);
rho = 1.1;
max_mu = 1e8;
mu = 0.1*lambda;  
atx = A'*X;
inv_a = inv(A'*A+eye(m));
%% Initializing optimization variables
% intialize
J = zeros(m,n); if optdata.gpu ==1; J = gpu(J); end
Z = zeros(m,n); if optdata.gpu ==1; Z = gpu(Z); end
E = zeros(d,n);if optdata.gpu ==1; E = gpu(E); end
theta = 1e-3*rand(m,n); if optdata.gpu == 1; theta = gpu(theta); end
Y1 = zeros(d,n); if optdata.gpu ==1; Y1 = gpu(Y1); end
Y2 = zeros(m,n); if optdata.gpu ==1; Y2 = gpu(Y2); end
%% Start main loop
iter = 0;
if display
    disp(['initial,rank=' num2str(rank(Z))]);
end
diff = zeros(1, maxIter);if optdata.gpu ==1; diff = gpu(diff); end
while iter<maxIter
    iter = iter + 1;
    %update J
    temp = Z + Y2/mu;
    try
        [U,sigma,V] = svd(temp,'econ');
    catch
        [U,sigma,V] = svd(temp+theta,'econ');
    end
    sigma = diag(sigma);
    svp = length(find(sigma>1/mu));
    if svp>=1
        sigma = sigma(1:svp)-1/mu;
    else
        svp = 1;
        sigma = 0;
    end
    J = U(:,1:svp)*diag(sigma)*V(:,1:svp)';
    
    %udpate Z
    Z = inv_a*(atx-A'*E+J+(A'*Y1-Y2)/mu);
    
    %update E
    xmaz = X-A*Z;
    temp = xmaz+Y1/mu;
    E = max(0,temp - lambda/mu)+min(0,temp + lambda/mu);
    
    leq1 = xmaz-E;
    leq2 = Z-J;
    stopC = max(max(max(abs(leq1))),max(max(abs(leq2))));
    diff(iter) = stopC;
    if display && (iter==1 || mod(iter,30)==0 || stopC<tol)
        disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
            ',rank=' num2str(rank(Z,1e-3*norm(Z,2))) ',stopALM=' num2str(stopC,'%2.3e')]);
    end
    if iter>10 &&   (abs(diff(iter) - diff(iter-10))/abs(diff(iter)) <1e-3)
        break;
    end
    if stopC<tol 
        break;
    else
        Y1 = Y1 + mu*leq1;
        Y2 = Y2 + mu*leq2;
        mu = min(max_mu,mu*rho);
    end
end
