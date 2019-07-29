function [Z, L] = updateZL_srrs(Var, opt)
% Var is initialed variables
%opt is options

tau = 2*opt.alpha;
if opt.nu == 0
    D = opt.prec.X; % closed form solution
    [U, Lam, ~] = svd(D,'econ');
    L = D;
else
    if opt.lam == 0
        alpha = Var.mu; 
        D = opt.prec.X -Var.E + Var.Y1/Var.mu; % no J
    else
        alpha = 2*Var.mu; 
        D = (opt.prec.X -Var.E + Var.Y1/Var.mu + Var.J - Var.Y2/Var.mu)/2;
    end  
    t = (alpha +tau)/alpha/tau;
    [U, Sig, V] = svd(D,'econ');
    Lam = pt_alpha_tau(diag(Sig), t, alpha , tau);
    L = U*Lam*V';
end

l = diag(Lam);
lmax = length(l(l>1/sqrt(tau)));
V1 = V(:, 1:lmax);
Z = V1*pt(l, tau)*V1';

% norm_nuc(Var.P) +tau/2*norm(Var.L-Var.P*Var.L,'fro')^2 +alpha/2*norm(D-Var.L,'fro')
% norm_nuc(P) +tau/2*norm(L-P*L,'fro')^2 +alpha/2*norm(D-L,'fro')

% norm_nuc(Var.P) +tau/2*norm(Var.L-Var.P*Var.L,'fro')^2 
% norm_nuc(P) +tau/2*norm(L-P*L,'fro')^2

end % end of the file