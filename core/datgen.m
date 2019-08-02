function [X,Xcv,Xtest,T] = datgen(optdata)
% return the data without outlier or nomalization

if nargin <1 || optdata.ind_dataset == 0
    [dat, datcv, dattest, T] = toy_datagen(optdata); % toy data
    X.data = dat;
    X.label = zeros(2, 900);
    X.label(1,1:900) = sum(kron(diag([1,2,3]), ones(1,300)),1);    
    X.label(2,1:900) = zeros(1,900);
    
    Xcv.data = dattest; % test is cv
    ltest = size(dattest,2);
    Xcv.label = zeros(2, ltest);
    Xcv.label(1,1:ltest) = sum(kron(diag([1,2,3]), ones(1,ltest/3)),1);
    Xcv.label(2,1:ltest) = zeros(1,ltest);
    
    Xtest.data = datcv; % cv is test, then cv has more data
    lcv = size(datcv,2);
    Xtest.label = zeros(2, lcv);
    Xtest.label(1,1:lcv) = sum(kron(diag([1,2,3]), ones(1,lcv/3)),1);
    Xtest.label(2,1:lcv) = zeros(1,lcv);

elseif optdata.ind_dataset == 1    
    load('ExtendedYaleB_front_32x32.mat');     
    ncv = 0;  % 13 samples per class -testing
    ntrain = 30; % 40 samples per class -training

    % adding outlider is moved outside this function
    
    % data partition
    Xcv.data = [];
    Xcv.label = [];
    X.data = [];
    X.label = [];
    Xtest.data = [];
    Xtest.label = [];
    Labels = [labels;1:length(labels)]; % adding index (primary key) for all the data    

    for i = 1:max(labels)              
        [~, startpoint] =  find(labels == i,1); % index of first sample in current class
        [~, endpoint] = find(labels == i,1,'last');
        
        rng(optdata.rng)
        ind = randperm(endpoint - startpoint+1); % some classes has 64 samples, as few as 59
        
        ind_test = startpoint-1+ ind(1:ncv);
        ind_train = startpoint-1+ ind(ncv+1: ncv+ntrain); 
        ind_cv = startpoint-1+ ind(ncv+ntrain+1: end);

        Xcv.data = [Xcv.data, data(:, ind_test)];
        Xcv.label = [Xcv.label, Labels(:,ind_test)];
        
        X.data = [X.data, data(:, ind_train)];
        X.label = [X.label, Labels(:,ind_train)];                
      
        Xtest.data = [Xtest.data, data(:, ind_cv)];
        Xtest.label = [Xtest.label, Labels(:,ind_cv)];        
    end 
    T = 0; 
    
elseif optdata.ind_dataset == 2   
    load('coil20');
    ncv = 0;  % 11 samples per class -testing
    ntrain = 40; % 50 samples per class -training

    % adding outlider is moved outside this function
    
    % data partition
    Xcv.data = [];
    Xcv.label = [];
    X.data = [];
    X.label = [];
    Xtest.data = [];
    Xtest.label = [];
    Labels = [label;1:length(label)]; % adding index (primary key) for all the data    

    for i = 1:max(label)              
        [~, startpoint] =  find(label == i,1); % index of first sample in current class
        [~, endpoint] = find(label == i,1,'last');
        
        rng(optdata.rng)
        ind = randperm(endpoint - startpoint+1); % some classes has 64 samples, as few as 59
        
        ind_test = startpoint-1+ ind(1:ncv);
        ind_train = startpoint-1+ ind(ncv+1: ncv+ntrain); 
        ind_cv = startpoint-1+ ind(ncv+ntrain+1: end);

        Xcv.data = [Xcv.data, data(:, ind_test)];
        Xcv.label = [Xcv.label, Labels(:,ind_test)];
        
        X.data = [X.data, data(:, ind_train)];
        X.label = [X.label, Labels(:,ind_train)];                
      
        Xtest.data = [Xtest.data, data(:, ind_cv)];
        Xtest.label = [Xtest.label, Labels(:,ind_cv)];        
    end 
    T = 0; 
        
elseif optdata.ind_dataset == 3 % This dataset contains outlier itself no outliter adding is needed.
    data = zeros(165*120/9, 100*26); % downsample 3*3
    for i = 1:50
        for ii = 1:26
            part1 = ['M-00',num2str(i)];
            if i > 9 
                part1 = ['M-0',num2str(i)]; 
            end                
            part2 = ['-0',num2str(ii),'.bmp'];
            if ii > 9 
                part2 = ['-',num2str(ii),'.bmp'];
            end
            a = mean(imread([part1, part2]),3);
            a = imresize(a, [165/3, 120/3]);
            data(:,(i-1)*26+ii) = a(:);    
        end
    end
    % FEMALE
    for i = 1:50
        for ii = 1:26
            part1 = ['W-00',num2str(i)];
            if i > 9 
                part1 = ['W-0',num2str(i)]; 
            end                
            part2 = ['-0',num2str(ii),'.bmp'];
            if ii > 9 
                part2 = ['-',num2str(ii),'.bmp'];
            end
            a = mean(imread([part1, part2]),3);
            a = imresize(a, [165/3, 120/3]);
            data(:,(50+i-1)*26+ii) = a(:);    
        end
    end
    % data partition
    Xcv.data = [];
    Xcv.label = [];
    X.data = [];
    X.label = [];
    Xtest.data = [];
    Xtest.label = [];   
    label = sum(kron(eye(100), 1:26));
    Labels = [label; 1:(100*26)];
    ncv = 0;  % 11 samples per class -testing
    ntrain = 16; % 50 samples per class -training 

    rng(optdata.rng)
    ind = randperm(26); % 
    ind_cv = sum(kron(eye(100), ind(1:ncv)) + kron(diag(0:99), 26*ones(1, ncv)));
    ind_train = sum(kron(eye(100), ind(ncv+1: ncv+ntrain)) + kron(diag(0:99), 26*ones(1, ntrain))); 
    ind_test = sum(kron(eye(100), ind(ncv+ntrain+1: end)) + kron(diag(0:99), 26*ones(1, length(ind(ncv+ntrain+1: end)))));
        
    Xcv.data = [Xcv.data, data(:, ind_cv)];
    Xcv.label = [Xcv.label, Labels(:,ind_cv)];

    X.data = [X.data, data(:, ind_train)];
    X.label = [X.label, Labels(:,ind_train)];                

    Xtest.data = [Xtest.data, data(:, ind_test)];
    Xtest.label = [Xtest.label, Labels(:,ind_test)];        
    T = 0; 
else
    fprintf('\nno dataset found\n \n')
end

if optdata.gpu
    X.data = gpu(X.data);
    X.label = gpu(X.label); 
    Xtest.data = gpu(Xtest.data);
    Xtest.label = gpu(Xtest.label); 
    Xcv.data = gpu(Xcv.data);
    Xcv.label = gpu(Xcv.label); 
end
    
end %end of the file
