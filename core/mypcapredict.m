function prel = mypcapredict(P, Xcv)
    C = max(Xcv.label(end-1, :));
    n = length(Xcv.label(end-1, :));
    rece = zeros(C, n);
    for i = 1:n
        dat = Xcv.data(:,i);
        for ii = 1:C
            datpro =  P{ii}'*dat;
            rece(ii, i) = cpu(norm(datpro)^2);
        end
    end
    [~,prel] = max(rece, [], 1);        

end % end of function