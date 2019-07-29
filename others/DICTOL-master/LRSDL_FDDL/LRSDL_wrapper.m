function [acc, rt,res] = LRSDL_wrapper(Y_train, label_train, Y_test , label_test, ...
                            k, k0, lambda1, lambda2, lambda3)
% function [acc, rt] = LRSDL_wrapper(Y_train, label_train, Y_test , label_test, ...
%                             k, k0, lambda1, lambda2, lambda3)
% -----------------------------------------------
% Author: Tiep Vu, thv102@psu.edu, 5/11/2016
%         (http://www.personal.psu.edu/thv102/)
% -----------------------------------------------
    if nargin == 0 % test mode
        dataset = 'myYaleB';
        N_train = 10;        
        [~, Y_train, Y_test, label_train, label_test] = ...
            train_test_split(dataset, N_train);        
        k = 8;
        k0 = 5;
        lambda1 = 0.001;
        lambda2 = 0.01;
        lambda3 = 0.01;
    end 
     C = max(label_train);
    opts.k           = k;
    opts.k0          = k0;
    opts.show_cost   = 0;
    opts.lambda1     = lambda1;
    opts.lambda2     = lambda2;
    opts.lambda3     = lambda3;
    opts.D_range     = k*(0:C);
    opts.D_range_ext = [opts.D_range k*C+k0];
    opts.initmode    = 'normal';   
    opts.max_iter    = 100;
    opts             = initOpts(opts);
    opts.verbose      = true*0;
    opts.tol         = 1e-8;
    %% Train 
    [D, D0, X, X0, CoefM, coefM0, opts, rt] = ...
                    LRSDL(Y_train, label_train, opts);
    X1 = [X; X0];
    Y_range = label_to_range(label_train);
    C = max(label_train);
    CoefMM0 = zeros(size(X1,1), C);
    for c = 1: C 
        X1c = get_block_col(X1, c, Y_range);
        CoefMM0(:,c) = mean(X1c,2);
    end    
    opts.verbose = 0;
    acc0 = [];
    if numel(D0) ~= 0
%         fprintf('GC:\n');
        acc0 = LRSDL_pred_GC(Y_test, D, D0, CoefM, coefM0, opts, label_test);
%         fprintf('maximum acc: %4f\n', max(acc));
    else             
%         fprintf('GC:\n');
        opts.weight = 0.5;
        for vgamma = 0.01%[0.0001, 0.001, 0.01, 0.1]
            opts.gamma = vgamma;
            pred = FDDL_pred(Y_test, D, CoefM, opts);
            acc0 = [acc0 calc_acc(pred, label_test)];
%             fprintf('gamma = %.4f, acc = %.4f\n', vgamma, acc(end));
        end 

    end
%     fprintf('\n');
    % added code
    acc = max(acc0);
%     acc.max = max(acc0);
%     acc.mean = mean(acc0);    
    res.D = D;
    res.D0 = D0;
    res.X = X;
    res.X0 = X0;
    
end 
