function prel = mypcapredict(P, Xcv, optdata)
    C = max(Xcv.label(end-1, :));
    n = length(Xcv.label(end-1, :));
    rece = zeros(C, n);
    if optdata.gpu
        rece = gpu(rece);
    end
    
    for i = 1:n
        dat = Xcv.data(:,i);
        for ii = 1:C
            datpro =  P{ii}'*dat;
            rece(ii, i) = norm(datpro)^2;
        end
    end
    [~,prel] = max(rece, [], 1);        

end % end of function