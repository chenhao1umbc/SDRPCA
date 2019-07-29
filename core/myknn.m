function acc = myknn(Xtr, labels, Xcv, Prj, k)
% perform k-neareast neighbors

    if nargin < 5
        k =5;
    end
    Mdl = fitcknn( Xtr',labels','NumNeighbors',k,'Standardize',1);
    prelabel = predict(Mdl,(Prj*Xcv.data)');
    a = prelabel - Xcv.label(1,:)';
    acc = length(a(a ==0))/length(a);
    
end
