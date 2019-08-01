close all
clear
clc
addpath(genpath('../../SDRPCA'))
addpath(genpath('../../data_img'))


% init data & and settings
try
gpu(1)
optdata.gpu = 1;
fprintf('GPU is used \n')
catch 
fprintf('GPU is not available, calculating on cpu \n')
end
optdata.ind_dataset = 1;% 1 is Extended Yale B, 0 is toy data
optdata.add_outlier = true; % adding outlier or not
optdata.o_per = 0.2;% outlier percentage
optdata.outlier_type = 'l1'; % l1 is l1 norm, l21 is l21 norm, no other options
optdata.rng = 0; % random seed
[X0,X0cv,X0test,T] = datgen(optdata); 
[X,Xcv,Xtest,E] = out_norm(X0, X0cv, X0test, optdata);
cv_fold = 5; % 3 folds cross-validation


addpath('lrr')
m = 4;
n = 5;
i=[1  2  m];j=[1  3  n];s = [6  7  8]; 
A = sparse(i,j,s);
B=full(A);
c1 = randn(m,1);
c2 = randn(1,n);
C = c1*c2;
% C=ones(m, n);

X=1.0*(B+C)

lambda = 1/sqrt(n); % try 0.05 0.1 0.5

[Z1,E1] = solve_lrr(X,X,lambda,1,1);
L1 = X*Z1;
norm(L1- C, 'fro')
E1


[Z2,E2] = solve_lrr(X,eye(m,n),lambda,1,1);
L2 = eye(m,n)*Z2;
norm(L2- C, 'fro')
E2

[Z3,E3] = solve_lrr(X,C,lambda,1,1);
L3 = C*Z3;
norm(L3- C, 'fro')
E3