function [Xtr, Xcv, Xtest, Truth] = toy_datagen(optdata)

% setup
% X = randn(100, 1000);
% Xcv = randn(100, 200);
% Xtest = randn(100, 200);

if nargin <1
    optdata.o_per = 0.1;
    optdata.outlier_type = 'l1';
end
    

d = 10; % ambient dimension
N = 1500; % total number of samples
Ntrain = 900; % number of traning samples
Ncv = 450; % number of crossvalidation samples
p_outlier = optdata.o_per;    % outlier fraction
Noutlier = p_outlier*N; % number of outliers
SNR = inf;    % SNR for dense noise
snr = 10;   % "SNR" for coefficients
oir = 10;   % outier-inlier power ratio
Ntest = N - Ntrain - Ncv; % number of test samples
d_comm = 2; % common subspace dimension
d_dist = 1; % distinct subspace dimension (per class)
C = 3; % number classes
rng(optdata.rng);%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%--------rng--------%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% generate data
d_all = d_comm+d_dist*C; % all the clean data living in d_all dimension
bases = orth(rand(d,d_all)); % generate all bases
s_amp = 1;  % signal amplitude
n_std = sqrt(s_amp^2*10^(-snr/10)); % noise standard dev

% common subspace
spc = bases(:,1:d_comm)*(s_amp + n_std*randn(d_comm,N))/sqrt(d_comm);

% distinct subspace
nc = N / C; % samples per class
for c = 1:C
    spd(:,(c-1)*nc+1:c*nc) = bases(:,d_comm+(c-1)*d_dist+1:d_comm+c*d_dist) ...
        * (s_amp+n_std*randn(d_dist,nc))/sqrt(d_dist);
end

% overall data
sp = spc + spd;
X0 = sp;

%% add outliers (moved outside of the function)
% E = zeros(d,N);
% X = sp + E;
X = X0;

%% split data for training, CV, and testing
Xtr = zeros(d,Ntrain);
Xtr0 = zeros(d,Ntrain);
Xcv = zeros(d, Ncv);
Xtest = zeros(d, Ntest);
for i = 1:C
    Xtr(:,1+Ntrain/C*(i-1):Ntrain/C*i) = X(:,1+nc*(i-1): nc*(i-1)+Ntrain/C);
    
    ind1= 1+ nc*(i-1) + Ntrain/C;
    Xcv(:, 1+ Ncv/C*(i-1): Ncv/C*i) = X(:,ind1: nc*(i-1) + (Ntrain+Ncv)/C);
    
    ind2 = 1+ nc*(i-1) +(Ntrain+Ncv)/C; 
    Xtest(:, 1+ Ntest/C*(i-1): Ntest/C*i) = X(:, ind2 : nc*(i-1)+N/C); 
end



%% adding Gaussian noise
% X = awgn(X, SNR);

%% normalize data
%X = X./sum(X.^2,1);
%Xcv = Xcv./sum(Xcv.^2,1);
%Xtest = Xtest./sum(Xtest.^2,1);

%%% return Truth
Truth.X0 = X0;
Truth.b = bases;
Truth.P = bases*bases';
Truth.Ptildebase = bases(:, d_comm+1:d_all);


%% show plot of the data 
showplot = 0;
if showplot   
    figure(10); clf
    for i = 1:C
        plot3(spc(1,1+nc*(i-1): nc*i),spc(2,1+nc*(i-1): nc*i),spc(3,1+nc*(i-1): nc*i),'x')
        hold on
        grid on
    end
    title('common subspace')

    figure(11); clf
    for i = 1:C
        plot3(spd(1,1+nc*(i-1): nc*i),spd(2,1+nc*(i-1): nc*i),spd(3,1+nc*(i-1): nc*i),'x')
        hold on
        plot3(spc(1,1+nc*(i-1): nc*i),spc(2,1+nc*(i-1): nc*i),spc(3,1+nc*(i-1): nc*i),'x')
        grid on
    end
    title('common and distinctive subspace')

    figure(12); clf
    for i = 1:C
        plot3(sp(1,1+nc*(i-1): nc*i),sp(2,1+nc*(i-1): nc*i),sp(3,1+nc*(i-1): nc*i),'x')
        hold on
        grid on
    end
    title('synthetic data without outlier')
    
    ncX = Ntrain/C;
    figure(13); clf
    for i = 1:C
        plot3(X(1,1+ncX*(i-1): ncX*i),X(2,1+ncX*(i-1): ncX*i),X(3,1+ncX*(i-1): ncX*i),'x')
        hold on
        grid on
    end
    title('synthetic data with outlier')
end



end % end of the file