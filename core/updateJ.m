function [J] = updateJ(Var)

% Var is initialed variables
%opt is options
L = Var.P + Var.Y2/Var.mu;
tau = 1/Var.mu;

J = svt(L, tau);


end % end of the file