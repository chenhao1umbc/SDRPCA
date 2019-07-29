function [Z] = updateZ(Var, opt)
% Var is initialed variables
%opt is options

temp = -Var.Y2/Var.mu + Var.J + opt.prec.XtX - opt.prec.Xt*(Var.E - Var.Y1/Var.mu);
if opt.lam ==0
    Z = opt.prec.invXtXI*temp;
else
    c = Var.mu/2/opt.lam;
    XtPtPttX = opt.prec.Xt*Var.Ptilde*Var.Ptilde'*opt.prec.X;
    XtPtPttXinv = inv(XtPtPttX+ 1e-6*eye(size(Var.E,2)));
    
    sylA = c*XtPtPttXinv*opt.prec.XtXI;
    sylB = opt.prec.A; % A is low rank, not invertable
    sylC = c*XtPtPttXinv* temp;
    Z = lyap(sylA, sylB, -sylC);
end

% opt.lam*fisher(Var.Ptilde, opt.prec.X*Var.Z,3)
% Var.mu/2*(norm(opt.prec.X - opt.prec.X*Var.Z-Var.E+Var.Y1/Var.mu,'fro')^2 +...
%     norm(Var.Z- Var.J + Var.Y2/Var.mu,'fro')^2)

end % end of the file