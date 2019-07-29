function [X_new, Xcv_new, E] = myshffle(X0,X0cv, s_ind, optdata)
% s_ind is shuffle ind

X1 = [X0.data;X0.label]; % bind label to data
X2 = [X0cv.data;X0cv.label];
XX = [X1, X2]; % concat. training and cv data
[~, Ind] = sort(XX(end,:), 'ascend'); % class 1 has the smallest prime keys 
X_sort = XX(:,Ind);

Ntrain = size(X0.data, 2);
C = max(X0cv.label(1,:)); % how many classes
ind_train_all = zeros(1, Ntrain);
Ntr_per = Ntrain/C; % this should be interger
for i = 1:C
    [~, startpoint] =  find(X_sort(end-1,:) == i,1); % index of first sample in current class
    [~, endpoint] = find(X_sort(end-1,:) == i,1,'last'); % sample in each class could be different
    if nargin >2
        rng(s_ind)
    end
    ind_train = randperm(endpoint - startpoint+1); %permute the current samples
    ind_train_all(1+Ntr_per*(i-1):Ntr_per*i) = startpoint-1+ind_train(1:Ntr_per); 
end
X_new_temp = X_sort(:,ind_train_all);
X_sort(:,ind_train_all)= [];
Xcv_new_temp = X_sort;

X0_new.data = X_new_temp(1:end-2, :);
X0_new.label = X_new_temp(end-1:end, :);
X0cv_new.data = Xcv_new_temp(1:end-2, :);
X0cv_new.label = Xcv_new_temp(end-1:end, :);

[X_new,Xcv_new, ~, E] = out_norm(X0_new, X0cv_new, [], optdata); % adding outlier

end % end of this file
