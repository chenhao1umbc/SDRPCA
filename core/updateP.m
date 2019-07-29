function [P] = updateP(Var, opt)
% Var is initialed variables
%opt is options
if opt.lam ==0
    P = (-Var.Y2/Var.mu + Var.J + opt.prec.XXt - (Var.E - Var.Y1/Var.mu)*opt.prec.Xt)*opt.prec.invXXtI;
else
    PtPtt = Var.Ptilde*Var.Ptilde';
    [d, ~] = size(PtPtt);
    c = Var.mu/2/opt.lam;
    sylA = PtPtt + eye(d)*1e-8;
    sylB = c*opt.prec.XXtIinvA; % A is low rank, not invertable
    sylC = c*(-Var.Y2/Var.mu + Var.J + opt.prec.XXt - (Var.E - Var.Y1/Var.mu)*opt.prec.Xt)*opt.prec.invA;
%     P = sylvester(sylA, sylB, sylC); % too slow
    P = lyap(sylA, sylB, -sylC);
end

end % end of the file