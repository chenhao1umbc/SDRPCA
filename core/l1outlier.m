function [X,Xcv,Xtest,E] = l1outlier(X0, X0cv, X0test, optdata)

o_per = optdata.o_per;% outlier percentage
opr = -10 - 10*log10(o_per/0.1);   % outier power ratio
rng(optdata.rng)

if ~isempty(X0test)
    data = [X0.data, X0cv.data, X0test.data];
else
    data = [X0.data, X0cv.data];
end
[~, ntrain] = size(X0.data);
[~, ncv] = size(X0cv.data);
n = numel(data); % numbder of total entries(pixels)
indx = randperm(n);
corrupt_ind = indx(1:floor(n*o_per));


if optdata.ind_dataset ==0 % toy data
    E = zeros(size(data));
    E(corrupt_ind) = min(min(data)) +  (max(max(data))-min(min(data)))*rand(floor(n*o_per), 1);
    data(corrupt_ind) = E(corrupt_ind);
    normdata = norm_data(data); % without normalization is to check the E_est
else    
    E = randi([0,255], size(data)); 
    temp = data;
    data(corrupt_ind) = E(corrupt_ind); % 0 or 255, SNR= 1, still working but much lower acc
    normdata = norm_data(data); % normalize data     
    E = normdata - norm_data(temp); % store the added outlier   
end    

X.data = normdata(:,1:ntrain); % subtract mean for each class
X.label = X0.label;
Xcv.data = normdata(:,ntrain+1:ntrain+ncv);
Xcv.label = X0cv.label;
Xtest = X0test;
if ~isempty(X0test)
    Xtest.data = normdata(:, ntrain+ncv+1:end);
    Xtest.label = X0test.label;
end


end% end of this file