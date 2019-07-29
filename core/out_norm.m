function [X, Xcv, Xtest, E] = out_norm(X0, X0cv, X0test, optdata)
% adding outliers and nomalization
% X0 X0cv X0test are raw data withouth normalization
% X Xcv Xtest are normalized

if optdata.add_outlier  
    if strcmp('l1', optdata.outlier_type)
        % adding outliers(corruption) to the whole dataset, normalization
        [X,Xcv,Xtest,E] = l1outlier(X0, X0cv, X0test, optdata);            
    end      
    if strcmp('l21', optdata.outlier_type)
        % adding outliers(corruption) to the training dataset, normalization
        [X,Xcv,Xtest,E] = l21outlier(X0, X0cv, X0test, optdata);                  
    end
%     X.data = removemean(X.data,  X.label(1,:));
    
else % if not adding outlier , just nomalize the data
    E = zeros(size(X0.data));
    if optdata.ind_dataset == 0
        X.data = X0.data;
        X.label = X0.label;
%         X.data = removemean(X.data,  X.label(1,:));
        
        Xcv.data = X0cv.data;
        Xcv.label = X0cv.label;
        
        if ~isempty(X0test)
            Xtest.data = X0test.data;
            Xtest.label = X0test.label;
        end
    else     
        X.data = norm_data(X0.data);
        X.label = X0.label;
%         X.data = removemean(X.data,  X.label(1,:));        
        Xcv = X0cv;
        if ~isempty(X0cv)
            Xcv.data = norm_data(X0cv.data);
            Xcv.label = X0cv.label;
        end
        
        Xtest = X0test;
        if ~isempty(X0test)
            Xtest.data = norm_data(X0test.data);
            Xtest.label = X0test.label;
        end    
    end
end

end %end of this file