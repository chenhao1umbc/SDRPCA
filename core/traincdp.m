function [Var] = traincdp(X, Var, opt)
% X is the training data
% Var is initialed variables
% opt is options
global optdata
c1 = zeros(1,opt.maxiter);if optdata.gpu ==1; c1 = gpu(c1); end
c2 = zeros(1,opt.maxiter);if optdata.gpu ==1; c2 = gpu(c2); end
if opt.calcost
    orig = zeros(1, opt.maxiter);if optdata.gpu ==1; orig = gpu(orig); end
    alm = zeros(1, opt.maxiter);if optdata.gpu ==1; alm = gpu(alm); end
    obj1 = zeros(1, opt.maxiter);if optdata.gpu ==1; obj1 = gpu(obj1); end
    obj2 = zeros(1, opt.maxiter);if optdata.gpu ==1; obj2 = gpu(obj2); end
    obj3 = zeros(1, opt.maxiter);if optdata.gpu ==1; obj3 = gpu(obj3); end
    sparsity = zeros(1, opt.maxiter);if optdata.gpu ==1; sparsity = gpu(sparsity); end
end

for iter = 1:opt.maxiter
    
    Var.Ptilde = updatePtilde(Var, opt);
    Var.P = updateP(Var, opt);
    Var.J = updateJ(Var);
    Var.E = updateE(Var, opt);

    t1 = X - Var.P*X - Var.E;
    t2 = Var.P - Var.J;
    c1(iter) = norm(t1,'fro');
    c2(iter) = norm(t2,'fro');

    if opt.calcost
        [orig(iter), alm(iter), obj1(iter), obj2(iter), obj3(iter)] = ...
           costfunc(Var, opt, t1, t2);
       sparsity(iter) = sum(abs(Var.E(:)) > 1e-6)/numel(Var.E);  % fraction of nonzeros in E
    end
    
    if iter> 10
    if(abs(c1(iter) - c1(iter-10))/abs(c1(iter)) <1e-3) &&...
            (abs(c2(iter) - c2(iter-10))/abs(c2(iter)) <1e-3)
        iter
        break;
    end
    end
    
    if (c1(iter) < opt.tol && c2(iter) < opt.tol)
        break;
    else
        Var.Y1 = Var.Y1 + Var.mu*t1;
        Var.Y2 = Var.Y2 + Var.mu*t2;
        Var.mu = min(opt.mumax, Var.mu*opt.rho);
    end
    
end

if opt.calcost
    figure(100); clf
    plot(orig(1:iter), '-x'); hold on
    plot(obj1(1:iter), 'linewidth',2);
    plot(obj2(1:iter), 'linewidth',2);
    plot(obj3(1:iter), ':x','linewidth',2);
    title('Cost function')
    legend('Cost','||P||_*','\lambda*Fisher','\nu*||E||_{l}');
    
    figure(101)
    plot(alm(1:iter), '-x')
    title('Augmented Lagrangian')
    
    figure(110)
    semilogy(c1(1:iter), '-x')
    title('||X - PX - E||_F')
    
    figure(111)
    semilogy(c2(1:iter), '-x')
    title('||P - J||_F')
    
    figure(112)
    plot(sparsity(1:iter), '-x')
    title('fraction of nonzeros in E')
end



end% end of the file