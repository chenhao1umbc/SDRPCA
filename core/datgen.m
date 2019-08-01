function [X,Xcv,Xtest,T] = datgen(optdata)
% return the data without outlier or nomalization

if nargin <1 || optdata.ind_dataset == 0
    [dat, datcv, dattest, T] = toy_datagen(optdata); % toy data
    X.data = dat;
    X.label = zeros(2, 900);
    X.label(1,1:900) = sum(kron(diag([1,2,3]), ones(1,300)),1);    
    X.label(2,1:900) = zeros(1,900);
    
    Xtest.data = dattest; % test is cv
    ltest = size(dattest,2);
    Xtest.label = zeros(2, ltest);
    Xtest.label(1,1:ltest) = sum(kron(diag([1,2,3]), ones(1,ltest/3)),1);
    Xtest.label(2,1:ltest) = zeros(1,ltest);
    
    Xcv.data = datcv; % cv is test, then cv has more data
    lcv = size(datcv,2);
    Xcv.label = zeros(2, lcv);
    Xcv.label(1,1:lcv) = sum(kron(diag([1,2,3]), ones(1,lcv/3)),1);
    Xcv.label(2,1:lcv) = zeros(1,lcv);

elseif optdata.ind_dataset == 1    
    load('ExtendedYaleB_front_32x32.mat');     
    ntest = 13;  % 13 samples per class -testing
    ntrain = 30; % 40 samples per class -training

    % adding outlider is moved outside this function
    
    % data partition
    Xtest.data = [];
    Xtest.label = [];
    X.data = [];
    X.label = [];
    Xcv.data = [];
    Xcv.label = [];
    Labels = [labels;1:length(labels)]; % adding index (primary key) for all the data    

    for i = 1:max(labels)              
        [~, startpoint] =  find(labels == i,1); % index of first sample in current class
        [~, endpoint] = find(labels == i,1,'last');
        
        rng(optdata.rng)
        ind = randperm(endpoint - startpoint+1); % some classes has 64 samples, as few as 59
        
        ind_test = startpoint-1+ ind(1:ntest);
        ind_train = startpoint-1+ ind(ntest+1: ntest+ntrain); 
        ind_cv = startpoint-1+ ind(ntest+ntrain+1: end);

        Xtest.data = [Xtest.data, data(:, ind_test)];
        Xtest.label = [Xtest.label, Labels(:,ind_test)];
        
        X.data = [X.data, data(:, ind_train)];
        X.label = [X.label, Labels(:,ind_train)];                
      
        Xcv.data = [Xcv.data, data(:, ind_cv)];
        Xcv.label = [Xcv.label, Labels(:,ind_cv)];        
    end 
    T = 0; 
    
elseif optdata.ind_dataset == 2   
%     load('coil20un'); %%%%%%%%% What is this?
    load('coil20');
    ntest = 11;  % 11 samples per class -testing
    ntrain = 50; % 50 samples per class -training

    % adding outlider is moved outside this function
    
    % data partition
    Xtest.data = [];
    Xtest.label = [];
    X.data = [];
    X.label = [];
    Xcv.data = [];
    Xcv.label = [];
    Labels = [label;1:length(label)]; % adding index (primary key) for all the data    

    for i = 1:max(label)              
        [~, startpoint] =  find(label == i,1); % index of first sample in current class
        [~, endpoint] = find(label == i,1,'last');
        
        rng(optdata.rng)
        ind = randperm(endpoint - startpoint+1); % some classes has 64 samples, as few as 59
        
        ind_test = startpoint-1+ ind(1:ntest);
        ind_train = startpoint-1+ ind(ntest+1: ntest+ntrain); 
        ind_cv = startpoint-1+ ind(ntest+ntrain+1: end);

        Xtest.data = [Xtest.data, data(:, ind_test)];
        Xtest.label = [Xtest.label, Labels(:,ind_test)];
        
        X.data = [X.data, data(:, ind_train)];
        X.label = [X.label, Labels(:,ind_train)];                
      
        Xcv.data = [Xcv.data, data(:, ind_cv)];
        Xcv.label = [Xcv.label, Labels(:,ind_cv)];        
    end 
    T = 0; 
        
elseif optdata.ind_dataset == 3 % This dataset contains outlier itself no outliter adding is needed.
    allx = zeros(165*120, 50*26);
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
            allx(:,(i-1)*26+ii) = a(:);    
        end
    end
    Labels = [sum(kron(eye(50), 1:26)); 1:(50*26)];
    indx = sum(kron(eye(50), [1:7, 14:20])+kron(diag(0:49), ones(1,14)*26));
    ind_te = sum(kron(eye(50), [8:13])+kron(diag(0:49), ones(1,6)*26));
    ind_cv = sum(kron(eye(50), [21:26])+kron(diag(0:49), ones(1,6)*26));
    X.data = allx(:, indx);
    X.label = Labels(:, indx);
    Xcv.data = allx(:, ind_cv);
    Xcv.label = Labels(:,ind_cv);
    Xtest.data = allx(:, ind_te);
    Xtest.label = Labels(:, ind_te);
    T = 0;
else
    fprintf('\nno dataset found\n \n')
end

try 
    X.data = gpu(X.data);
    X.label = gpu(X.label); 
    Xcv.data = gpu(Xcv.data);
    Xcv.label = gpu(Xcv.label); 
    Xtest.data = gpu(Xtest.data);
    Xtest.label = gpu(Xtest.label); 
    fprintf('GPU is used \n')
catch
    fprintf('GPU is not available, calculating on cpu \n')
    
end %end of the file
