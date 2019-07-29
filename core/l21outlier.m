function [X,Xcv,Xtest,E] = l21outlier(X0, X0cv, X0test, optdata)

o_per = optdata.o_per;% outlier percentage
rng(optdata.rng)

data = X0.data;
[d, ntrain] = size(X0.data);
indx = randperm(ntrain);
corrupt_ind = indx(1:floor(ntrain*o_per));
E = zeros(d, ntrain);

% only adding to train data, cv and test is clean
if optdata.ind_dataset == 1 % YaleB
    addE = rand(size(data,1),floor(ntrain*o_per));
    data(:, corrupt_ind) = addE; % uniform distribution   
    % store the added outlier indexes
    E(:,corrupt_ind) = addE./sqrt(sum(addE.^2,1));    
    normdata = data./sqrt(sum(data.^2,1)); % normalize data 
    
elseif optdata.ind_dataset ==0 % toy data
    addE = rand(size(data,1),floor(ntrain*o_per));
    data(:, corrupt_ind) = addE; % uniform distribution   
    % store the added outlier indexes
    E(:,corrupt_ind) = addE./sqrt(sum(addE.^2,1));   
    E = E./sqrt(sum(E.^2,1)); % normalize E 
    normdata = data./sqrt(sum(data.^2,1)); % normalize data 
end   

X.data = normdata;
X.label = X0.label;
Xcv.data = X0cv.data./sqrt(sum(X0cv.data.^2,1));
Xcv.label = X0cv.label;
Xtest = X0test;
if ~isempty(X0test)
    Xtest.data = X0test.data./sqrt(sum(X0test.data.^2,1));
    Xtest.label = X0test.label;
end

end% end of this file