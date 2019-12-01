close all
clear
clc
addpath(genpath('../../SDRPCA'))
addpath(genpath('../../data_img'))


% init data & and settings
global optdata % to cooperate with existing code for GPU 
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
optdata.o_per = 0.2;% outlier percentage
optdata.outlier_type = 'l1'; % l1 is l1 norm, l21 is l21 norm, no other options
optdata.rng = 0; % random seed
[X0,X0cv,X0test,T] = datgen(optdata); 
[X,Xcv,Xtest,E] = out_norm(X0, X0cv, X0test, optdata);
cv_fold = 5; % 3 folds cross-validation
[Var0, opt] = initdata2(X, optdata);
acc = 0;

nu_set = 2.^(-(-3:16));
lam_set = 2.^(-(-3:16));
o_per_set = 0:0.1:0.5;
alpha = [0.001 0.01 0.1 1 10 100 1000]; 

%%
tic
for s = 2:3
optdata.ind_dataset = s;% 1 is Extended Yale B, 0 is toy data
acc_all = zeros(length(o_per_set), length(nu_set), length(lam_set), length(alpha));
if optdata.gpu,  acc_all = gpu(zeros(6, length(nu_set), length(lam_set), 7)); end
    for o_per = 1:length(o_per_set)
    for n = 1:length(nu_set)
    for l = 1: length(lam_set)
    for a = 1:7
        for i = 1:cv_fold    
        optdata.o_per = o_per_set(o_per);% outlier percentage
        optdata.rng = i; % random seed
        [X0,X0cv,X0test,T] = datgen(optdata); 
        [X,Xcv,Xtest,E] = out_norm(X0, X0cv, X0test, optdata);
        Xcv = Xtest;

        [Var0, opt] = initdata2(X, optdata);
        opt.nu = nu_set(n); % try 0.05 0.1 0.5
        opt.lam = lam_set(l);% for fisher
        opt.alpha = alpha(a);
        opt.percentage = 0.9;
        Var = traincdp2(X.data, Var0, opt);
        Proj = Var.Ptilde'*Var.P;
        Xtr = Proj*X.data;

        % KNN classifier
        acc = acc + myknn(Xtr, X.label(1,:), Xtest, Proj); % k = 5
        end
    acc_all(o_per, n, l, a) = acc/cv_fold
    disp('dataset'); disp(s); 
    acc = 0;
    if opt.saveresult 
    dt = datestr(datetime);
    dt((datestr(dt) == ':')) = '_'; % for windows computer
    filenamedt = ['sdrpca2_acc','dataset_',num2str(s),'_', dt];
    save(filenamedt, 'acc_all');
    end
    end
    end    
    end
    end
end



%{
for ind1 = 1:length(nu)
for ind2 = 1:length(lam)
for ind3 = 1:length(alpha) 

    opt.nu = nu(ind1);
    opt.lam = lam(ind2);
    opt.alpha = alpha(ind3);
    acc = zeros(1,cv_fold);
    cv_fold = 5;
    
    for f = 1: cv_fold
        tic
        %train and crossvalidation
        Var = traincdp2(X.data, Var0, opt);

        % testing
        if ~isempty(Var.Ptilde) && size(Var.Ptilde,2)<300
        rate_tmp = zeros(1,size(Var.Ptilde,2));
        for dim = 1:1:size(Var.Ptilde,2)
            Prj0 = Var.Ptilde(:,1:dim)';
            Prj = Prj0*Var.P;
            Xtr = Prj*Var.L;
            % KNN classifier
%             acc0 = myknn(Xtr, X.label(1,:), Xcv, Prj);% k = 5
            acc0 = mysvm(Xtr, X.label(1,:), Xcv, Prj);% k = 5
            rate_tmp(dim) = acc0;  
            acc(f) = max(rate_tmp);
        end
        % shuffle Xtrain and Xcv
        [X, Xcv, E] = myshffle(X0, X0cv, f*10, optdata); % f for shuffle ind
        [Var, ~] = initdata2(X, optdata);
        else
            isemp = ~isempty(Var.Ptilde)
            sizePtilde = size(Var.Ptilde,2)
            cv_fold = f-1;
            break
        end        
        toc
    end
    av_acc(ind1, ind2, ind3) = sum(acc)/cv_fold;
    if opt.saveresult     
        dt = datestr(datetime);
        dt((datestr(dt) == ':')) = '_'; % for windows computer
        filenamedt = ['../../result/f2_',dt];
        save(filenamedt, 'nu','ind1','ind2','ind3','opt',...
             'cv_fold','optdata', 'acc', 'av_acc');
    end
end        
end
end
end

if strcmp('l1', optdata.outlier_type)
    run plotEP_l1
else
    run plotEP_l21    
end
%}