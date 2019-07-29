function acc = mysvm(Xtr, labels, Xcv, Prj, k)
% perform svm to get the accuracy

    Mdl = fitcecoc( Xtr',labels');
    prelabel = predict(Mdl,(Prj*Xcv.data)');
    a = prelabel - Xcv.label(1,:)';
    acc = length(a(a ==0))/length(a);
    
end
