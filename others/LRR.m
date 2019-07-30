clear
clc

addpath('lrr')
m = 4;
n = 5;
i=[1  2  m];j=[1  3  n];s = [6  7  8]; 
A = sparse(i,j,s);
B=full(A);
c1 = randn(m,1);
c2 = randn(1,n);
C = c1*c2;
% C=ones(m, n);

X=1.0*(B+C)

lambda = 1/sqrt(n); % try 0.05 0.1 0.5

[Z1,E1] = solve_lrr(X,X,lambda,1,1);
L1 = X*Z1;
norm(L1- C, 'fro')
E1


[Z2,E2] = solve_lrr(X,eye(m,n),lambda,1,1);
L2 = eye(m,n)*Z2;
norm(L2- C, 'fro')
E2

[Z3,E3] = solve_lrr(X,C,lambda,1,1);
L3 = C*Z3;
norm(L3- C, 'fro')
E3