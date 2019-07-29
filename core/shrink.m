function W = shrink(alpha, Q)

% Var is initialed variables
%opt is options
%alpha*norm(W,21)+0.5*norm(W-Q, 'fro')^2
[r, c] = size(Q);
W = Q;
for i = 1:c
    Qinorm = norm(Q(:,i));
    if Qinorm > alpha
        W(:, i)=(Qinorm - alpha)/Qinorm*Q(:,i);
    else
        W(:, i)=zeros(r,1);
    end
end


end % end of the file