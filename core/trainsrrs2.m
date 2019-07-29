function [Var] = trainsrrs2(X, Var, opt)
% X is the training data
% Var is initialed variables
%opt is options

c1 = zeros(1, opt.maxiter);
c2 = zeros(1, opt.maxiter);
if opt.calcost
    orig = zeros(1, opt.maxiter);
    alm = zeros(1, opt.maxiter);
    obj1 = zeros(1, opt.maxiter);
    obj2 = zeros(1, opt.maxiter);
    obj3 = zeros(1, opt.maxiter);
    obj4 = zeros(1, opt.maxiter);
    sparsity = zeros(1, opt.maxiter);
end

% calc initial value for ploting 
if opt.plotcost
    t1 = X - Var.L -Var.E;
    if opt.lam == 0        
        t2 = Var.L;
    else
        t2 = Var.L - Var.J;
    end
    if opt.calcost
        [or, al, ob1, ob2, ob3, ob4]= costfunc_srrs2(Var, opt, t1, t2);
    end 
end

for iter= 1: opt.maxiter    
    
    [Var.Z, Var.L] = updateZL_srrs(Var, opt);   
    Var.J = updateJ2(Var, opt); 
    Var.Ptilde = updatePtilde2(Var, opt);
    Var.E = updateE2(Var, opt);
    
    t1 = X - Var.L -Var.E;
    if opt.lam == 0        
        t2 = Var.L;
    else
        t2 = Var.L - Var.J;
    end
    if opt.calcost
        [orig(iter), alm(iter), obj1(iter), obj2(iter), obj3(iter), obj4(iter)]=...
            costfunc_srrs2(Var, opt, t1, t2);
        sparsity(iter) = sum(abs(Var.E(:)) > 1e-6)/numel(Var.E);  % fraction of nonzeros in E
    end 
    c1(iter) = norm(t1,'fro');
    c2(iter) = norm(t2,'fro');
    if c2(iter) < opt.tol && c1(iter) < opt.tol && iter>1
        break        
    else
        Var.Y1 = Var.Y1 + Var.mu*t1;
        Var.Y2 = Var.Y2 + Var.mu*t2;
        Var.mu = min(opt.mumax, Var.mu*opt.rho);
    end
    
end

if opt.calcost
    if opt.plotcost
    figure(100); clf
    plot([or,orig(1:iter)], '-x'); hold on
    plot([ob1,obj1(1:iter)], 'linewidth',2);
    plot([ob2,obj2(1:iter)], 'linewidth',2);
    plot([ob3,obj3(1:iter)], ':x','linewidth',2);
    plot([ob4,obj4(1:iter)], ':*','linewidth',2);
    title('Cost function')
    legend('Cost','||Z||_*','\lambda*Fisher','\nu*||E||_{2,1}', '\alpha||L-PL||_F^2');
    
    figure(101)
    plot([al,alm(1:iter)], '-x')
    title('Augmented Lagrangian')
    
    figure(110)
    semilogy(c1(1:iter), '-x')
    title('||X - XZ - E||_F')
    
    figure(111)
    semilogy(c2(1:iter), '-x')
    title('||L - J||_F')
    
    figure(112)
    plot(sparsity(1:iter), '-x')
    title('fraction of nonzeros in E')
	
    end
end  

end% end of the file