% run initcode % one can do -- run initcode_l21
close all
clear
clc
addpath(genpath('../../SDRPCA'))
addpath(genpath('../../data_img'))

% init data & and settings
global optdata % to cooperate with existing code for GPU 
try
    gpuDevice(4);
    optdata.gpu = 1;
    fprintf('GPU is used \n')
catch 
    optdata.gpu = 0;
    fprintf('GPU is not available, calculating on cpu \n')
end
optdata.ind_dataset = 1;% 1 is Extended Yale B, 0 is toy data
optdata.add_outlier = true; % adding outlier or not
optdata.o_per = 0.2;% outlier percentage
optdata.outlier_type = 'l1'; % l1 is l1 norm, l21 is l21 norm, no other options
optdata.rng = 0; % random seed
[X0,X0cv,X0test,T] = datgen(optdata); 
[X,Xcv,Xtest,E] = out_norm(X0, X0cv, X0test, optdata);
[Var0, opt] = initdata(X, optdata);
cv_fold = 5; % 3 folds cross-validation
nu_set = 2.^(-(-3:16));
lam_set = 2.^(-(-3:16));
o_per_set = 0:0.1:0.5;
acc = 0;
%%
tic
for s = 1:3
optdata.ind_dataset = s;% 1 is Extended Yale B, 0 is toy data
acc_all = zeros(length(o_per_set), length(nu_set));
if optdata.gpu,  acc_all = gpu(zeros(length(o_per_set), length(nu_set))); end
    for o_per = 1:length(o_per_set)
    for n = 1:length(nu_set)
    for l = 1: length(lam_set)
        for i = 1:cv_fold    
        optdata.o_per = o_per_set(o_per);% outlier percentage
        optdata.rng = i; % random seed
        [X0,X0cv,X0test,T] = datgen(optdata); 
        [X,Xcv,Xtest,E] = out_norm(X0, X0cv, X0test, optdata);
        Xcv = Xtest;

        [Var0, opt] = initdata(X, optdata);
        opt.nu = nu_set(n); % try 0.05 0.1 0.5
        opt.lam = lam_set(l);% for fishe
        opt.percentage = 0.9;
        Var = traincdp(X.data, Var0, opt);
        Proj = Var.Ptilde'*Var.P;
        Xtr = Proj*X.data;

        % KNN classifier
        acc = acc + myknn(Xtr, X.label(1,:), Xtest, Proj); % k = 5
        end
    acc_all(o_per, n) = acc/cv_fold
    disp('dataset'); disp(s); 
    acc = 0;
    end
    end    
    end
end
toc







% {
for ii = 1:6
optdata.o_per = 0.1*(ii -1);% outlier percentage
optdata.outlier_type = 'l1'; % l1 is l1 norm, l21 is l21 norm, no other options
optdata.rng = 0; % random seed
[X0,X0cv,X0test,T] = datgen(optdata);  % T is Truth for toy data
[X,Xcv,Xtest,E] = out_norm(X0, X0cv, X0test, optdata);
[Var0, opt] = initdata(X, optdata);

% opt.rank_Ptilde = 68;
nu = [1e-3, 1e-2, 1e-4, 0.1, 1, 1e-6, 1e-5];% for error
lam = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1];% for fisher
av_acc = zeros(length(nu), length(lam)); % >0.9408
opt.calcost = true*0;
cv_fold = 5;

for ind1 = 1:length(nu)    
for ind2 = 1:length(lam)
    
    opt.nu = nu(ind1);
    opt.lam = lam(ind2);
    acc = zeros(1,cv_fold);
    cv_fold = 5;
    
    for f = 1: cv_fold
        tic
        %train and crossvalidation        
        Var = traincdp(X.data, Var0, opt);
        
        % testing
        if ~isempty(Var.Ptilde) && size(Var.Ptilde,2)<300
        rate_tmp = zeros(1,size(Var.Ptilde,2));
        for dim = 1:1:size(Var.Ptilde,2)
            Prj0 = Var.Ptilde(:,1:dim)';
%             Prj = Var.Ptilde(:,end-dim+1:end)';
            Prj = Prj0*Var.P;
            Xtr = Prj*X.data;
            % KNN classifier
            acc0 = myknn(Xtr, X.label(1,:), Xcv, Prj);% k = 5
            rate_tmp(dim) = acc0;           
        end
        acc(f) = max(rate_tmp)
        % shuffle Xtrain and Xcv
        [X, Xcv, E] = myshffle(X0,X0cv, f*10, optdata); % f for shuffle ind
        [Var0, ~] = initdata(X, optdata);
        else
            isemp = ~isempty(Var.Ptilde)
            sizePtilde = size(Var.Ptilde,2)
            cv_fold = f-1;
            break
        end  
        toc
    end    
    av_acc(ind1, ind2) = sum(acc)/cv_fold;
    if opt.saveresult 
        dt = datestr(datetime);
        dt((datestr(dt) == ':')) = '_'; % for windows computer
        filenamedt = ['../../result/f1_255',dt];
        save(filenamedt, 'nu', 'lam', 'ind1','opt',...
            'ind2', 'cv_fold','optdata', 'acc', 'av_acc');
    end
end    
end
end
rank(Var.P, 1e-4)
rank(Var.Ptilde)
if strcmp('l1', optdata.outlier_type)
    run plotEP_l1
else
    run plotEP_l21    
end
%}