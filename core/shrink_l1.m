function E = shrink_l1(eps, W)

% eps*norm(W,1)+0.5*norm(W-Q, 'fro')^2
E = W;

N = length(E(:));
for i = 1:N
    if E(i) > eps
        E(i) = E(i) - eps;
    elseif E(i) < -eps
        E(i) = E(i) + eps;
    else
        E(i) = 0;
    end
end

end % end of the file