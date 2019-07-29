function [a] = norm_nuc(A)
% to calculate the nuclear norm of A

a =trace(sqrt(A'*A));

end % end of the file