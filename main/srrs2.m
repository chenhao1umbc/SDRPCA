 % init data, clear, close all, adding pathes
addpath(genpath('../../1_irpcafishercleanl1'))
run initcode % one can do -- run initcode_l21
[Var0, opt] = init_srrs2(X, optdata,X0.data);

nu = 0.05;
lam = 1e-5; %1e-4
alpha = 1000;
av_acc = zeros(length(nu), length(lam), length(alpha)); % >0.9542
opt.saveresult = 1;
cv_fold = 1 ;

for ind1 = 1:length(nu)
for ind2 = 1:length(lam)
for ind3 = 1:length(alpha) 

    opt.nu = nu(ind1);
    opt.lam = lam(ind2);
    opt.alpha = alpha(ind3);
    acc = zeros(1,cv_fold);
    
    for f = 1: cv_fold
        tic
        %train and crossvalidation
        Var = trainsrrs2(X.data, Var0, opt);
        
        % testing
        rate_tmp = [];
        for dim = 1:1:size(Var.Ptilde,2)
            Prj = Var.Ptilde(:,1:dim)';
%             Prj = Var.Ptilde(:,end-dim+1:end)';
            Xtr = Prj*Var.L*Var.Z;
            % KNN classifier
            acc = myknn(Xtr, X.label(1,:), Xcv, Prj);% k = 5
            rate_tmp = [rate_tmp; acc];            
        end
        max(rate_tmp)
%         % shuffle Xtrain and Xcv
%         [X0, X0cv, E] = myshffle(X0, X0cv, f*10, optdata); % f for shuffle ind
%         [Var0, opt] = initdata2(X, optdata);
%         opt.nu = nu(ind1);
%         opt.lam = lam(ind2);
%         opt.alpha = alpha(ind3);
        toc
    end    
    av_acc(ind1, ind2, ind3) = sum(acc)/cv_fold;
%     if opt.saveresult
%         dt = datestr(datetime);
%         dt((datestr(dt) == ':')) = '_'; % for windows computer
%         filenamedt = ['../../result/srrs2_',dt];
%         save(filenamedt, 'nu','ind1','ind2','ind3',...
%              'cv_fold','optdata', 'acc', 'av_acc');
%     end
end
end
end

if strcmp('l1', optdata.outlier_type)
    run plotEZ_l1
else
    run plotEZ_l21    
end