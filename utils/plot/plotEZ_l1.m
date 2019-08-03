[u, s, v] = svd(Var.Z);
figure;plot(diag(s),'x')
title 'the singular values of Z'

[d, n] = size(Var.E);
figure;imagesc(E(1:min(100,d),1:min(100,n)))
title 'the original E'

figure;imagesc(Var.E(1:min(100,d),1:min(100,n)))
title 'the stimated E'