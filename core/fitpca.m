
function acc = fitpca(P, Xcv)

    prelabel = mypcapredict(P, Xcv);
    a = prelabel - Xcv.label(1,:);
    acc = length(a(a ==0))/length(a);

end % end of function