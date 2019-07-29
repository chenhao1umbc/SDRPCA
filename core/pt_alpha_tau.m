function Lam = pt_alpha_tau(Sig, t, alpha , tau)
% Sig is not maxtrix, it is a vector
% opt is options

sigma_s = sqrt(t + sqrt(t/alpha));
Sig(Sig< sigma_s) = Sig(Sig< sigma_s)/(t*tau);
Lam = diag(Sig);

end % end of the file