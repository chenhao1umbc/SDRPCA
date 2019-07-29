function [Var, opt] = initdata2(X,optdata, X0)
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
opt.maxiter = 200; % for main loop
opt.lam = 0.001 ; % for fisher term
opt.nu = 0.01 ;% for norm(E, 21)
opt.mumax = 1e16;
opt.eta = 1.1;
opt.rho = 1.3;
opt.tol = 1e-10;
% opt.rank_Ptilde = opt.C;
opt.calcost = true*0;
opt.plotcost = opt.calcost*1;
opt.saveresult = true;
opt.outlier_type = optdata.outlier_type; % string type 'l1' or 'l21'
% algorithm 2
opt.alpha = 0.1; % for ||L-PL||

Var.mu = 0.1;
Var.P = eye(d);
Var.Ptilde = eye(d);
Var.E = zeros(d, N);
Var.Y1 = zeros(d,N);
% algorithm 2
Var.L = X;
Var.J = X;
Var.Y2 = zeros(d,N);

% precalculation
C = opt.C;

Nc = N/C; % may samples per class is not equal
H1 = kron(eye(C),ones(Nc)/Nc);
H2 = ones(N)/N;
Within = (eye(N) - H1)^2;
Between = (H1 - H2)^2;
Conv = eye(N)*opt.eta;
opt.prec.Xt = X';
opt.prec.X = X;
opt.prec.within = Within;
opt.prec.between = Between;
opt.prec.conv = Conv; % it is not full rank! if opt.eta<=1
opt.prec.A = Within - Between + Conv;
opt.prec.Ainv = inv(opt.prec.A);

% calculate true Ptilde 
if nargin > 2 
    J = X0;
    A = J*((eye(N) - H1)^2 - (H1 - H2)^2 + opt.eta*eye(N))*J';
    B = (A+A')/2;
    [V, D] = eig(B);
    [v, index] = sort(diag(D),'ascend');
    ind = index(find(v>1e-4));
    opt.Pt0 = V(:,ind);    
end


end % end of the file