function [Var, opt] = init_srrs1(X, optdata, X0)
% X is the training data
% Var is initialed variables, opt is options

% initialize data
if isfield(X, 'label')
    [d, N] = size(X.data);
    opt.C = max(X.label(1,:)); % how many classes
    X = X.data;
else
    [d, N] = size(X);
    opt.C = 3; % how many classes
    X = X;
end
opt.maxiter = 200; % for main loop
opt.lam = 1e-4 ; % for fisher term
opt.nu = 0.1 ;% for norm(E, 21), norm(E,1)
opt.mumax = 1e16;
opt.eta = 1.1;
opt.rho = 1.3;
opt.tol = 1e-10;
% opt.rank_Ptilde = opt.C;
opt.calcost = true*0;
opt.saveresult = true;
opt.outlier_type = optdata.outlier_type; % string type 'l1' or 'l21'

Var.mu = 0.1;
Var.Z = eye(N);
Var.J = eye(N);
Var.Ptilde = eye(d);
Var.E = zeros(d, N);
Var.Y1 = eye(d,N);
Var.Y2 = eye(N);


% precalculation
C = opt.C;
Nc = N/C; % may samples per class is not equal
H1 = kron(eye(C),ones(Nc)/Nc);
H2 = ones(N)/N;

XtX = X'*X;
within = (eye(N) - H1)^2;
between = (H1 - H2)^2;
conv = opt.eta*eye(N);
A = within - between + conv;
invXtXI = inv(XtX+eye(N));
XtXI = XtX+eye(N);

opt.prec.invXtXI = invXtXI;
opt.prec.XtXI = XtXI;
opt.prec.XtX = XtX;
opt.prec.Xt = X';
opt.prec.X = X;
opt.prec.A = A;
opt.prec.within = within;
opt.prec.between = between;
opt.prec.conv = conv;

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