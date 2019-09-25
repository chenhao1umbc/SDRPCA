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

Var.mu = 0.1; if optdata.gpu ==1; Var.mu = gpu(Var.mu); end
Var.P = eye(d);if optdata.gpu ==1; Var.P = gpu(Var.P); end
Var.J = eye(d);if optdata.gpu ==1; Var.J = gpu(Var.J); end
Var.Ptilde = eye(d);if optdata.gpu ==1; Var.Ptilde = gpu(Var.Ptilde); end
Var.E = zeros(d, N);if optdata.gpu ==1; Var.E = gpu(Var.E); end
Var.Y1 = eye(d,N);if optdata.gpu ==1; Var.Y1 = gpu(Var.Y1); end
Var.Y2 = eye(d,d);if optdata.gpu ==1; Var.Y2 = gpu(Var.Y2); end


% precalculation
C = opt.C;if optdata.gpu ==1; C = gpu(C); end
Nc = N/C;if optdata.gpu ==1; Nc = gpu(Nc); end % may samples per class is not equal
H1 = kron(eye(C),ones(Nc)/Nc);if optdata.gpu ==1; H1 = gpu(H1); end
H2 = ones(N)/N;if optdata.gpu ==1; H2 = gpu(H2); end

XXt = X*X';if optdata.gpu ==1; XXt = gpu(XXt); end
within = X*(eye(N) - H1)^2*X';if optdata.gpu ==1; within = gpu(within); end
between = X*(H1 - H2)^2*X';if optdata.gpu ==1; between = gpu(between); end
conv = opt.eta*XXt;if optdata.gpu ==1; conv = gpu(conv); end
A = within - between + conv;if optdata.gpu ==1; A  = gpu(A ); end
invA = inv(A + 1e-8*eye(d));if optdata.gpu ==1; invA = gpu(invA); end% make non-singular
XXtIinvA = (XXt+eye(d))*invA;if optdata.gpu ==1; XXtIinvA = gpu(XXtIinvA); end
invXXtI = inv(XXt+eye(d));if optdata.gpu ==1; invXXtI = gpu(invXXtI); end

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
