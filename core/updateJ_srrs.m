function [J] = updateJ_srrs(Var)

% Var is initialed variables
%opt is options
L = Var.Z + Var.Y2/Var.mu;
tau = 1/Var.mu;

J = svt(L, tau);


end % end of the file