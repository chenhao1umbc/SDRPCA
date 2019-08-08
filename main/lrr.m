% run initcode % one can do -- run initcode_l21
close all
clear
clc
addpath(genpath('../../SDRPCA'))
addpath(genpath('../../data_img'))

% init data & and settings
global optdata % to cooperate with existing code for GPU 
try
    gpuDevice(3);
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
lamb_set = 2.^(-(-3:12));
o_per_set = 0:0.1:0.5;
acc = 0;

%%
tic
for s = 1:3
    optdata.ind_dataset = s;% 1 is Extended Yale B, 0 is toy data
    acc_all = zeros(length(o_per_set), length(lamb_set));
    if optdata.gpu,  acc_all = gpu(zeros(length(o_per_set), length(lamb_set))); end
for o_per = 1:length(o_per_set)
for l = 1:length(lamb_set)
for i = 1:cv_fold
optdata.o_per = o_per_set(o_per);% outlier percentage
optdata.rng = i; % random seed
[X0,X0cv,X0test,T] = datgen(optdata); 
[X,Xcv,Xtest,E] = out_norm(X0, X0cv, X0test, optdata);
Xcv = Xtest;

lambda = lamb_set(l); % try 0.05 0.1 0.5
[Z, E_hat] = LRR_l1(X.data,X.data,lambda);
Proj = get_proj(X.data*Z);
Xtr = Proj*X.data;
% KNN classifier
acc = acc + myknn(Xtr, X.label(1,:), Xtest, Proj); % k = 5
end
acc_all(o_per, l) = acc/cv_fold
disp('dataset'); disp(s); 
acc = 0;
end
end
end
toc


