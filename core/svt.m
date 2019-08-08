function [P] = svt(L, tau)

% Var is initialed variables
%opt is options
% tau*norm(P, 'nuc')+ 0.5*norm(P-L, 'fro')
try
    [u, s, v] = svd(L);
catch
    [u, s, v] = svd(L+ 1e-5*rand(size(L)));
end
s = s -tau;
s(s<0) = 0;
P = u*s*v';

end % end of the file