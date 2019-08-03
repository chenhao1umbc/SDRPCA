% run initcode % one can do -- run initcode_l21
close all
clear
clc
addpath(genpath('../../SDRPCA'))
addpath(genpath('../../data_img'))

% init data & and settings
try
    gpuDevice(2);
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


lambda = 1/sqrt(n); % try 0.05 0.1 0.5
[Z,E] = solve_lrr(X,X,lambda,1,1);
