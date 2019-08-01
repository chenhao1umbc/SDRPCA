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
[X0,X0cv,X0test,T] = datgen(optdata); 
M = size(X0.data,1); % data dimension
cv_fold = 5; % 3 folds cross-validation
d_set = floor(2/3*M/20): floor(2/3*M/20): floor(2/3*M);
o_per_set = 0:5;
acc = 0;

%%
tic
for o_per = 1:length(o_per_set)
for d = 1:length(d_set)
for i = 1:cv_fold
optdata.o_per = 0.1*o_per_set(o_per);% outlier percentage
optdata.rng = 0; % random seed
[X0,X0cv,X0test,T] = datgen(optdata); 
[X,Xtest,Xcv,E] = out_norm(X0, X0cv, X0test, optdata);
dim = d_set(d);
x = X.data;
x0 = x-mean(x,2);
[u,s,v] = svd(x0*x0');
Prj = u(:, 1:dim)'; 
Xtr = Prj*X.data;
% KNN classifier
acc = acc + myknn(Xtr, X.label(1,:), Xtest, Prj); % k = 5
end
acc = acc/cv_fold;

end
end

toc