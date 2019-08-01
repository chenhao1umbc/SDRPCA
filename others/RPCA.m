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
function [L, S] = RPCA(X, lambda, mu, tol, max_iter)

    % - X is a data matrix (of the size N x M) to be decomposed
    %   X can also contain NaN's for unobserved values
    % - lambda - regularization parameter, default = 1/sqrt(max(N,M))
    % - mu - the augmented lagrangian parameter, default = 10*lambda
    % - tol - reconstruction error tolerance, default = 1e-6
    % - max_iter - maximum number of iterations, default = 1000

    % this one is using the ADM with inexact ALM to solve it
    % asdfasdfe0234psdf
    [M, N] = size(X);
    unobserved = isnan(X);
    %在使用Matlab�?�仿真的时候难�?会出现数�?��?是数字的情况，就是NaN的情况，这些数�?�是�?能使用的,用isnan函数解决。
    %tf=isnan(A)：返回一个与A相�?�维数的数组，若A的元素为NaN（�?�数值），在对应�?置上返回逻辑1（真），�?�则返回逻辑0（�?�）。
    %对虚数z，如果z的实部或虚部都是NaN，那么isnan函数返回逻辑1，如果实部和虚部都是inf，则返回逻辑0。
    X(unobserved) = 0;
    normX = norm(X, 'fro');%n=norm(A),返回A的最大奇异值，�?�max(svd(A))

    % default arguments
    if nargin < 2%matalb �??供两个获�?�函数�?�数数目的函数，nargin返回函数输入�?�数的数�?
        lambda = 1 / sqrt(max(M,N));
    end
    if nargin < 3
        mu = 10*lambda;
    end
    if nargin < 4
        tol = 1e-6;
    end
    if nargin < 5
        max_iter = 1000;
    end

    % initial solution
    L = zeros(M, N);
    S = zeros(M, N);
    Y = zeros(M, N);

    for iter = (1:max_iter)
        % ADMM step: update L and S
        L = Do(1/mu, X - S + (1/mu)*Y);%更新低秩矩阵
        S = So(lambda/mu, X - L + (1/mu)*Y);%更新稀�?矩阵
        % and augmented lagrangian multiplier
        Z = X - L - S;
        Z(unobserved) = 0; % skip missing values
        Y = Y + mu*Z;

        err = norm(Z, 'fro') / normX;
        if (iter == 1) || (mod(iter, 10) == 0) || (err < tol)
            fprintf(1, 'iter: %04d\terr: %f\trank(L): %d\tcard(S): %d\n', ...
                    iter, err, rank(L), nnz(S(~unobserved)));
        end
        if (err < tol) break; end
    end
end

function r = So(tau, X)
    % shrinkage operator
    r = sign(X) .* max(abs(X) - tau, 0);
end

function r = Do(tau, X)
    % shrinkage operator for singular values
    [U, S, V] = svd(X, 'econ');
    r = U*So(tau, S)*V';
end
