function ptLam = pt(Lam, tau)
% Lam is not maxtrix, it is a vector
%opt is options
one_sqrt_tau = 1/sqrt(tau);
l = 1- 1/tau*(Lam(Lam> one_sqrt_tau).^(-2));
ptLam = diag(l);

end % end of the file