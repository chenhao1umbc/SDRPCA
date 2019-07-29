function [orig, alm, obj1, obj2, obj3, obj4] = costfunc2(Var, opt, c1, c2)
% t1 = X - Var.P*X - Var.E;
% t2 = Var.P - Var.J;
% original one is
% norm(P, 'nuc') + lam*fisher(ptilde, L) + nu*norm(E,21) + alpha*norm(L-PL,'fro')
% s.t. X = L + E, ptilde'ptilde = I, P' = P
%
% ALM one is
% norm(P, 'nuc') + lam*fisher(ptilde, J) + nu*norm(E,21) + tr(Y1'(X-J-E))
% + tr(Y2'(L-J)) + mu/2(norm(X-J-E, 'fro')^2 + norm(L-J, 'fro')^2) 
% s.t. ptilde'ptilde = I, P' = P
L_PL = Var.L-Var.P*Var.L;

PtiltL = Var.Ptilde'*Var.L;
LtPtil = PtiltL';
t1 = trace(PtiltL*opt.prec.within*LtPtil);
t2 = trace(PtiltL*opt.prec.between*LtPtil);
t3 = trace(PtiltL*opt.prec.conv*LtPtil);

obj1 = norm_nuc(Var.P);
obj2 = opt.lam*(t1 - t2 + t3);
if strcmp('l21', opt.outlier_type)
    obj3 = opt.nu*norm21(Var.E);
end
if strcmp('l1', opt.outlier_type)
    obj3 = opt.nu*norm1(Var.E);
end
obj4 = opt.alpha*norm(L_PL,'fro');

Part1 = obj1 + obj3+ obj4;
orig = Part1 + obj2;

PtiltJ = Var.Ptilde'*Var.J;
JtPtil = PtiltJ';
tt1 = trace(PtiltJ*opt.prec.within*JtPtil);
tt2 = trace(PtiltJ*opt.prec.between*JtPtil);
tt3 = trace(PtiltJ*opt.prec.conv*JtPtil);

alm = Part1 + opt.lam*(tt1 - tt2 + tt3)+...
    trace(Var.Y1'*c1) + trace(Var.Y2'*c2) + Var.mu/2*(norm(c1, 'fro')^2 + norm(c2, 'fro')^2);

end % end of the file