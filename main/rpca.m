% run initcode % one can do -- run initcode_l21
close all
clear
clc
addpath(genpath('../../SDRPCA'))
addpath(genpath('../../data_img'))

% init data & and settings
try
    gpuDevice(1);
    optdata.gpu = 1;
    fprintf('GPU is used \n')
catch 
    optdata.gpu = 0;
    fprintf('GPU is not available, calculating on cpu \n')
end
optdata.ind_dataset = 1;% 1 is Extended Yale B, 0 is toy data
optdata.add_outlier = true; % adding outlier or not
optdata.outlier_type = 'l1'; % l1 is l1 norm, l21 is l21 norm, no other options
optdata.rng = 0;
[X0,X0cv,X0test,T] = datgen(optdata); 
M = size(X0.data,1); % data dimension
cv_fold = 5; % 3 folds cross-validation
d_set = floor(2/3*M/20): floor(2/3*M/20): floor(2/3*M);
o_per_set = 0:0.1:0.5;
acc = 0;


i=[1  2  4];j=[1  3  5];s = [6  7  8]; 
A = sparse(i,j,s)
B=full(A)
C=ones(4,5)
D=1.0*(B+C);
D = [D, D*2, D*3]

[m,n]=size(D)
lambda=1.0/sqrt(max(m,n))
mu = 10*lambda
tol = 1e-6
max_iter = 1000
[L, S] = RPCA(D, lambda, mu, tol, max_iter)


