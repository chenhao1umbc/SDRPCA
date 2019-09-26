function [Var, opt] = initdata2(X, optdata, X0)
% X is the training data
% Var is initialed variables, opt is options

% initialize data
if isfield(X, 'label')
    [d, N] = size(X.data);
    opt.C = max(X.label(1,:)); % how many classes
%     X = removemean(X.data,  X.label(1,:));
    X = X.data;
else
    [d, N] = size(X);
    opt.C = 3; % how many classes
    X = X;
end
opt.gpu = optdata.gpu;
opt.maxiter = 500; % for main loop
opt.lam = 1e-4 ; % for fisher term
opt.nu = 0.1 ;% for norm(E, 21), norm(E,1)
opt.mumax = 1e8;
opt.eta = 1.1;
opt.rho = 1.1;
opt.tol = 5e-4;
% opt.rank_Ptilde = opt.C;
opt.calcost = true*0;
opt.plotcost = true*0;
opt.saveresult = true;
opt.outlier_type = optdata.outlier_type; % string type 'l1' or 'l21'
% algorithm 2
opt.alpha = 0.1; % for ||L-PL||


Var.mu = 0.1; if optdata.gpu ==1; Var.mu = gpu(Var.mu); end
Var.P = eye(d);if optdata.gpu ==1; Var.P = gpu(Var.P); end
Var.Ptilde = eye(d);if optdata.gpu ==1; Var.Ptilde = gpu(Var.Ptilde); end
Var.E = zeros(d, N);if optdata.gpu ==1; Var.E = gpu(Var.E); end
Var.Y1 = eye(d,N);if optdata.gpu ==1; Var.Y1 = gpu(Var.Y1); end
Var.Y2 = eye(d,N);if optdata.gpu ==1; Var.Y2 = gpu(Var.Y2); end
% algorithm 2
Var.L = X; if optdata.gpu ==1; Var.L = gpu(Var.L); end
Var.J = eye(d, N);if optdata.gpu ==1; Var.J = gpu(Var.J); end

% precalculation
C = opt.C;if optdata.gpu ==1; C = gpu(C); end
Nc = N/C;if optdata.gpu ==1; Nc = gpu(Nc); end % may samples per class is not equal
H1 = kron(eye(C),ones(Nc)/Nc);if optdata.gpu ==1; H1 = gpu(H1); end
H2 = ones(N)/N;if optdata.gpu ==1; H2 = gpu(H2); end

I_N = eye(N); if optdata.gpu ==1; I_N = gpu(eye(N)); end
Within = (I_N - H1)^2;
Between = (H1 - H2)^2;
Conv = I_N*opt.eta;
opt.prec.Xt = X';
opt.prec.X = X;
opt.prec.within = Within;
opt.prec.between = Between;
opt.prec.conv = Conv; % it is not full rank! if opt.eta<=1
opt.prec.A = Within - Between + Conv;
opt.prec.Ainv = inv(opt.prec.A);

% % calculate true Ptilde 
% if nargin > 2 
%     J = X0;
%     A = J*((eye(N) - H1)^2 - (H1 - H2)^2 + opt.eta*eye(N))*J';
%     B = (A+A')/2;
%     [V, D] = eig(B);
%     [v, index] = sort(diag(D),'ascend');
%     ind = index(find(v>1e-4));
%     opt.Pt0 = V(:,ind);    
% end

end % end of the file