function [Ptilde] = updatePtilde(Var, opt)
% Var is initialed variables
%opt is options

if opt.lam  == 0
    Ptilde = Var.Ptilde;
else
    P = Var.P;
    A = opt.lam*P*opt.prec.A*P';
    B = (A+A')/2;
    [V, D] = eig(B);
    [v, index] = sort(diag(D),'descend'); 
    
    threshold = 1e4;
    min_s = v(1)/threshold;
    P_len = length(v(v> min_s)); % trim small eigen values    
    P_kept_len = floor(opt.percentage*P_len);
    ind_trim = index(P_len:-1:P_kept_len);
    Ptilde = V(:, ind_trim);
end

end % end of the file