function [Ptilde] = updatePtilde(Var, opt)
% Var is initialed variables
%opt is options

offset = 0;

if opt.lam  == 0
    Ptilde = Var.Ptilde;
else
    P = Var.P;
    A = opt.lam*P*opt.prec.A*P';
    B = (A+A')/2;
    [V, D] = eig(B);
    [v, index] = sort(diag(D),'descend');    
    if isfield(opt, 'rank_Ptilde')
        ind = index(find(v>1e-6));
        Ptilde = V(:,index(ind(end - offset- round(opt.rank_Ptilde)+1 :...
            end -offset)));
    else
        ind = index(find(v>1e-6));
%         Ptilde = V(:, index(ind(ceil(0.1*end):end)));
        Ptilde = V(:, ind);
    end
end

end % end of the file