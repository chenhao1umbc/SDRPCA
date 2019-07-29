function [orig, alm, obj1, obj2, obj3] = costfunc(Var, opt, c1, c2)
% t1 = X - Var.P*X - Var.E;
% t2 = Var.P - Var.J;
% original one is
% norm(P, 'nuc') + lam*fisher(ptilde, PX) + nu*norm(E,21)
% s.t. X = PX + E, ptilde'ptilde = I
%
% ALM one is
% norm(P, 'nuc') + lam*fisher(ptilde, PX) + nu*norm(E,21) + tr(Y1'(X-PX-E))
% + tr(Y2'(P-J)) + mu/2(norm(X-PX-E, 'fro')^2, norm(P-J, 'fro')^2) 
% s.t. ptilde'ptilde = I
ptp = Var.Ptilde'*Var.P;

fisher = trace(ptp*(opt.prec.within - opt.prec.between + opt.prec.conv)*ptp');

obj1 = norm_nuc(Var.P);
obj2 = opt.lam*fisher;
if strcmp('l21', opt.outlier_type)
    obj3 = opt.nu*norm21(Var.E);
end
if strcmp('l1', opt.outlier_type)
    obj3 = opt.nu*norm1(Var.E);
end
orig = obj1 + obj2 + obj3;
alm = orig + trace(Var.Y1'*c1) + trace(Var.Y2'*c2) + ...
      Var.mu/2*(norm(c1,'fro')^2 + norm(c2,'fro')^2);

end % end of the file