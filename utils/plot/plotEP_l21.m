[u, s, v] = svd(Var.P);
figure;plot(diag(s),'x')
title 'the singular values of P'


figure;
subplot(3,1,1)
stem(sum(E.^2,1))
title 'E\_true vs E\_est'
hold on;stem(sum(Var.E.^2,1))
legend('E\_true', 'E\_est')
xlabel 'index'
ylabel 'magnitude'
subplot(3,1,2)
stem(sum(E.^2,1))
legend('E\_true')
title 'E\_true'
xlabel 'index'
ylabel 'magnitude'
subplot(3,1,3)
stem(sum(Var.E.^2,1))
title 'E\_est'
legend( 'E\_est')
xlabel 'index'
ylabel 'magnitude'