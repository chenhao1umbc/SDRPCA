function [data, av] = removemean(data_in,  classlabels)
% this function is to remove the mean per class

C = max(classlabels); % how many classes
data = data_in;
av = zeros(size(data, 1), C); % mean of each class
for i = 1:C
    ind = find(classlabels == i); % index of data in the same class
    tp = mean(data_in(:, ind), 2);
    data(:, ind) = data_in(:, ind) - tp;
    av(:, i) = tp;
end

end %end of the file