function [Ptilde] = updatePtilde2(Var, opt)
% Var is initialed variables
%opt is options

if opt.lam  == 0
    Ptilde = Var.Ptilde;
else
    J = Var.J;
    A = opt.lam*J*opt.prec.A*J';
    B = (A+A')/2;
    [V, D] = eig(B);
    [v, index] = sort(diag(D),'ascend');    
    if isfield(opt, 'rank_Ptilde')
        Ptilde = V(:,index(end - round(opt.rank_Ptilde)+1 :end));
    else
        ind = index(find(v>1e-6));
        Ptilde = V(:,ind);
    end
end

end % end of the file