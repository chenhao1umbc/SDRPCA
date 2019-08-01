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
[Var0, opt] = initdata2(X, optdata);

nu = [1 0.5 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001 5.e-05 1.e-05 1.e-06];
lam = [1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1 1];
alpha = [0.001 0.01 0.1 1 10 100 1000]; 
av_acc = zeros(length(nu), length(lam), length(alpha)); % >0.9542
cv_fold = 5 ;

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
