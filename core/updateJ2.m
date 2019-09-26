function [J] = updateJ2(Var, opt)
% Var is initialed variables
%opt is options

% this is the cost function for J


if opt.lam == 0
    J = Var.J;
else
    Ptil = Var.Ptilde;
    PtPtt = Ptil*Ptil';
    [d, ~] = size(PtPtt);
    mu_lam_2 = Var.mu/opt.lam/2;
    
    sylA = cpu(PtPtt + eye(d)*1e-8); % make it non-singular
    sylB = cpu(mu_lam_2*opt.prec.Ainv);
    sylC = cpu(mu_lam_2*(Var.L + Var.Y2/Var.mu)*opt.prec.Ainv);
%     J = sylvester(sylA, sylB, sylC);
    J = lyap(sylA, sylB, -sylC);
    if opt.gpu ==1; J = gpu(J); end
end

%     CostJ(J)
%     function Costfunctonv = CostJ(Jin)
%         Costfunctonv = opt.lam*fisher(Var.Ptilde, Jin, 3)+trace(Var.Y2'*(Var.L-Jin))+...
%             Var.mu/2*norm(Var.L - Jin, 'fro')^2;
%     end

end % end of the file
