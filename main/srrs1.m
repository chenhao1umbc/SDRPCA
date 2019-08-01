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
[Var0, opt] = init_srrs1(X, optdata);

nu = [1 0.12 0.1 0.08 0.012 0.01 0.008  0.0012 0.001 0.0008]; % sparse
lam = [1 0.12 0.1 0.08 0.012 0.01 0.008  0.0012 0.001 0.0008]; %  fisher
av_acc = zeros(length(nu), length(lam)); % >0.9408
opt.saveresult =1;
cv_fold = 5;
opt.calcost = true*0;

for ind1 = 1:length(nu)    
for ind2 = 1:length(lam)
    
    opt.nu = nu(ind1);
    opt.lam = lam(ind2);
    acc = zeros(1,cv_fold);
    
    for f = 1: cv_fold
        tic
        %train and crossvalidation
        Var = trainsrrs(X.data, Var0, opt);
        
        % testing
        rate_tmp = [];
        for dim = 1:1:size(Var.Ptilde,2)
            Prj = Var.Ptilde(:,1:dim)';
%             Prj = Var.Ptilde(:,end-dim+1:end)';
            Xtr = Prj*X.data*Var.Z;
            % KNN classifier
            acc0 = myknn(Xtr, X.label(1,:), Xcv, Prj);% k = 5
%             acc0 = mysvm(Xtr, X.label(1,:), Xcv, Prj);% k = 5
            rate_tmp = [rate_tmp; acc0];            
        end
        acc(f) = max(rate_tmp);
        % shuffle Xtrain and Xcv
        [X, Xcv, E] = myshffle(X0,X0cv, f*10, optdata); % f for shuffle ind
        [Var0, opt] = init_srrs1(X, optdata);
        opt.nu = nu(ind1);
        opt.lam = lam(ind2);
        toc
    end    
    av_acc(ind1, ind2) = sum(acc)/cv_fold
    if opt.saveresult
        dt = datestr(datetime);
        dt((datestr(dt) == ':')) = '_'; % for windows computer
        filenamedt = ['../../result/srrs1_',dt];
        save(filenamedt, 'nu', 'lam', 'ind1',...
            'ind2', 'cv_fold','optdata', 'acc', 'av_acc');
    end
end    
end

if strcmp('l1', optdata.outlier_type)
    run plotEZ_l1
else
    run plotEZ_l21    
end
