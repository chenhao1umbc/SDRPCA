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
cv_fold = 5; % 3 folds cross-validation
optdata.o_per = 0.0;% outlier percentage
optdata.rng = 0; % random seed
[X0,X0cv,X0test,T] = datgen(optdata); 
[X,Xtest,Xcv,E] = out_norm(X0, X0cv, X0test, optdata);


tic
P = trainmypca(X, 30);
acc = fitpca(P, Xcv, optdata);
toc

function P = trainmypca(Xin, dim)
    if nargin <2
        dim = 120;
    end
    X = Xin.data;
    label = Xin.label;
    C = max(label(end-1, :)); % how many classes
    P = cell(1, cpu(C)); %Save all the projections per class
    X_sort = [X; label];

    for i = 1:C
        [~, startpoint] =  find(X_sort(end-1,:) == i,1); % index of first sample in current class
        [~, endpoint] = find(X_sort(end-1,:) == i,1,'last'); % sample in each class could be different

        x = X(: , startpoint : endpoint); %permute the current samples
        x0 = x-mean(x,2);
        [u,s,v] = svd(x0*x0');
        P{i} = u(:, 1:dim);    
    end

end % end of function



