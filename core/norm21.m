function E = norm21(X)
% calculate the 21 norm

    c = size(X, 2);
    xi_norm = zeros(1,c);
    for i = 1:c
        xi_norm(i) = norm(X(:, i));
    end
    E = sum(xi_norm);
    
end% end of the file