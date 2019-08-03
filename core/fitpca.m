function acc = fitpca(P, Xcv, optdata)

    prelabel = mypcapredict(P, Xcv, optdata);
    a = prelabel - Xcv.label(1,:);
    acc = length(a(a ==0))/length(a);

end % end of function