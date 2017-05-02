function[Elbo]=elbo2(Gam,Alpha,Lambda,Phi,eta,W,K,D,V)% Evidence lower bound

ExpB = zeros(K,V);
for i  = 1:K
            ExpB(i,:) = exp((psi(Lambda(i,:))))/exp(psi(sum(Lambda(i,:),2)) );
end

e1    = zeros(D,1);    % elbo1 gives elbo for a particular document d
for d = 1:D
e1(d) = elbo1(Gam,Alpha,ExpB, Phi{d},W, K,d,V);
end
    Sum6 =  sum (- gammaln(sum(Lambda,2)) + sum(gammaln(Lambda),2));

    Sum7 = sum(gammaln(sum(eta,2)*V) - V*sum(gammaln(eta),2));
    
    Sum8 = sum(sum((eta - Lambda).*ExpB,2) );
Elbo = sum(e1)+ Sum6 + Sum7 + Sum8; 



