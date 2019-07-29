% init data, clear, close all, adding pathes
addpath(genpath('../../1_irpcafishercleanl1'))
run initcode % one can do -- run initcode_l21

tic
P = trainmypca(X, 3);
acc = fitpca(P, Xcv);
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



