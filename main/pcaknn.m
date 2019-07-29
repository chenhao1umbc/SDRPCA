% init data, clear, close all, adding pathes
addpath(genpath('../../1_irpcafishercleanl1'))
run initcode
% run initcode_l1

tic
% dim = 30; % for l1
dim = 144; % for l21
x = X.data;
x0 = x-mean(x,2);
[u,s,v] = svd(x0*x0');
Prj = u(:, 1:dim)'; 
Xtr = Prj*X.data;
% KNN classifier
acc_packnn = myknn(Xtr, X.label(1,:), Xtest, Prj); % k = 5
toc