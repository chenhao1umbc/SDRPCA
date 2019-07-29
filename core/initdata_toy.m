close all
clear
clc
addpath(genpath('../../1_irpcafishercleanl1'))
addpath(genpath('../../data'))

% init data & and settings
optdata.ind_dataset = 0;% 1 is Extended Yale B, 0 is toy data
optdata.add_outlier = true; % adding outlier or not
optdata.o_per = 0.5;% outlier percentage
optdata.outlier_type = 'l21'; % l1 is l1 norm, l21 is l21 norm, no other options
optdata.rng = 0; % random seed
[X0,X0cv,X0test,T] = datgen(optdata); 
optdata.T = T;
[X,Xcv,Xtest,E] = out_norm(X0, X0cv, X0test, optdata);
cv_fold = 3; % 3 folds cross-validation