% init data, clear, close all, adding pathes
addpath(genpath('.././1_irpcafishercleanl1'))
addpath(genpath('.././data'))
clear 
clc
close all
for ii = 0.1:0.1:0.5
% init data & and settings
run initcode 

%% FDDL 
lambda1 = 0.12;%  sparse
lambda2 = 0.08; %  fisher
k = 8; % How many atoms % 0.89 cv 10% outlier
[acc_fddl, rt_fddl, res_fddl] = LRSDL_wrapper(X.data, X.label(1,:), Xtest.data , Xtest.label(1,:), ...
                            k, 0, lambda1, lambda2, 0)
                        
%% LRSDL
lambda1 =  0.01;
lambda2 = 0.1;
lambda3 = 1;
k = 8;
k0 = 5; % 0.7125 cv 10% outlier
[acc_lrsdl,rt_lrsdl,res_lrsdl] = LRSDL_wrapper(X.data, X.label(1,:), Xtest.data , Xtest.label(1,:), ...
                            k, k0, lambda1, lambda2, lambda3)

                        
%% f1
tic
nu = 0.036;
lam = 1e-4; % 0.9350cv 10% outlier

[Var0, opt] = initdata(X, optdata);
opt.nu = nu;
opt.lam = lam;
% opt.rank_Ptilde = rank_Ptilde;
%train and crossvalidation
Var = traincdp(X.data, Var0, opt);
Prj = Var.Ptilde'*Var.P;
Xtr = Prj*X.data;
% KNN classifier
acc_f1 = myknn(Xtr, X.label(1,:), Xtest, Prj) % k = 5
toc

%% f2
tic
nu = 0.07;
lam = 1e-4;
alpha = 300;% 0.9550cv 10% outlier

[Var0, opt] = initdata2(X, optdata);
opt.nu = nu;
opt.lam = lam;
opt.alpha = alpha;
% opt.rank_Ptilde = rank_Ptilde;
Var = traincdp2(X.data, Var0, opt);
Prj = Var.Ptilde'*Var.P;
Xtr = Prj*Var.L;
% KNN classifier
acc_f2 = myknn(Xtr, X.label(1,:), Xtest, Prj) % k = 5
toc

%% srrs1
tic
nu = 0.05;
lam = 1e-4;
% rank_Ptilde = 4.5 * C;% acc = 0.9567

[Var0, opt] = init_srrs1(X, optdata);
opt.nu = nu;
opt.lam = lam;
% opt.rank_Ptilde = rank_Ptilde;

%train and crossvalidation
Var = trainsrrs(X.data, Var0, opt);
Prj = Var.Ptilde';
Xtr = Prj*X.data*Var.Z;
% KNN classifier

acc_srrs1 = myknn(Xtr, X.label(1,:), Xtest, Prj) % k = 5
toc

%% srrs2
tic
nu = 0.06;
lam = 1e-3;
alpha = 100;
% rank_Ptilde = 4.5 * C;% acc = 0.9892

[Var0, opt] = init_srrs2(X, optdata);
opt.nu = nu;
opt.lam = lam;
opt.alpha = alpha;
% opt.rank_Ptilde = rank_Ptilde;
Var = trainsrrs2(X.data, Var0, opt);
Prj = Var.Ptilde';
Xtr = Var.Ptilde'*Var.L*Var.Z;
        
% KNN classifier
acc_srrs2 = myknn(Xtr, X.label(1,:), Xtest, Prj) % k = 5
toc

end
