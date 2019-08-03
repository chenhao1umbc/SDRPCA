function P = get_proj(x)
% find the projection using PCA
x0 = x-mean(x,2);
[u,s,~] = svd(x0*x0');
diag_s = diag(s);
threshold = 1000;
min_s = max(diag_s)/threshold;
P_len = length(diag_s(diag_s> min_s));
P = u(:, 1:P_len)';

end % end of this file