function v = fisher(P, X, C)
% Var is initialed variables
% opt is options
% C is classes

[~, N] = size(X);
Nc = N/C;
H1 = kron(eye(C),ones(Nc)/Nc);
H2 = ones(N)/N;
Within = (eye(N) - H1)^2;
Between = (H1 - H2)^2;
Conv = 1.1*eye(N);
PX = P'*X;

v = trace(PX*Within*PX') - trace(PX*Between*PX') + trace(PX*Conv*PX');


end % end of the file