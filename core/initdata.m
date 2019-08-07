function [Var, opt] = initdata(X, optdata, X0)
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
opt.maxiter = 500; % for main loop
opt.lam = 1e-4 ; % for fisher term
opt.nu = 0.1 ;% for norm(E, 21), norm(E,1)
opt.mumax = 1e8;
opt.eta = 1.1;
opt.rho = 1.1;
opt.tol = 5e-6;
% opt.rank_Ptilde = opt.C;
opt.calcost = true*0;
opt.saveresult = true;
opt.outlier_type = optdata.outlier_type; % string type 'l1' or 'l21'

Var.mu = 0.1;
Var.P = eye(d);
Var.J = eye(d);
Var.Ptilde = eye(d);
Var.E = zeros(d, N);
Var.Y1 = eye(d,N);
Var.Y2 = eye(d,d);


% precalculation
C = opt.C;
Nc = N/C; % may samples per class is not equal
H1 = kron(eye(C),ones(Nc)/Nc);
H2 = ones(N)/N;

XXt = X*X';
within = X*(eye(N) - H1)^2*X';
between = X*(H1 - H2)^2*X';
conv = opt.eta*XXt;
A = within - between + conv;
invA = inv(A + 1e-8*eye(d));% make non-singular
XXtIinvA = (XXt+eye(d))*invA;
invXXtI = inv(XXt+eye(d));

opt.prec.invXXtI = invXXtI;
opt.prec.invA = invA;
opt.prec.XXtIinvA = XXtIinvA;
opt.prec.XXt = XXt;
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
    ind = index(v>1e-4);
    opt.Pt0 = V(:,ind);    
end

end % end of the file
