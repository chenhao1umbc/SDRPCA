function [P] = svt(L, tau)

% Var is initialed variables
%opt is options
% tau*norm(P, 'nuc')+ 0.5*norm(P-L, 'fro')
[u, s, v] = svd(L);
s = s -tau;
s(s<0) = 0;
P = u*s*v';

end % end of the file