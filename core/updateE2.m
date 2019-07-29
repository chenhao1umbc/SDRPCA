function [E] = updateE2(Var, opt)
% Var is initialed variables
%opt is options
if opt.nu == 0 
    E = zeros(size(Var.E));
else
    %% l_21 norm
    if strcmp('l21', opt.outlier_type)
        Q = opt.prec.X - Var.L + Var.Y1/Var.mu;
        alpha = opt.nu/Var.mu;
        E = shrink(alpha, Q);
    %% l_1 norm
    elseif strcmp('l1', opt.outlier_type)
        W = opt.prec.X - Var.L + Var.Y1/Var.mu;
        eps = opt.nu/Var.mu;
        E = shrink_l1(eps, W);    
    end
    
end

end % end of the file