function [orig, alm, obj1, obj2, obj3] = costfunc_srrs(Var, opt, c1, c2)
% t1 = X - Var.P*X - Var.E;
% t2 = Var.P - Var.J;
% original one is
% norm(P, 'nuc') + lam*fisher(ptilde, XZ) + nu*norm(E,21)
% s.t. X = XZ + E, ptilde'ptilde = I
%
% ALM one is
% norm(P, 'nuc') + lam*fisher(ptilde, XZ) + nu*norm(E,21) + tr(Y1'(X-XZ-E))
% + tr(Y2'(Z-J)) + mu/2(norm(X-XZ-E, 'fro')^2, norm(Z-J, 'fro')^2) 
% s.t. ptilde'ptilde = I
ptxz = Var.Ptilde'*opt.prec.X*Var.Z;

fisher = trace(ptxz*(opt.prec.within - opt.prec.between + opt.prec.conv)*ptxz');

obj1 = norm_nuc(Var.Z);
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