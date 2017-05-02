function[f]=elbo1(Gam,Alpha,B,Phi,W,K,d,V)% Evidence lower bound

Sum1=sum((Alpha -Gam(d,:)).*(psi(Gam(d,:))-psi(sum(Gam(d,:)))));

Sum2=(psi(Gam(d,:)))*(Phi')*sum(W{d},2) - (psi(sum(Gam(d,:)))*ones(1,K))*(Phi')*sum(W{d},2) + sum(sum(Phi.*(W{d}*log(B)')));


Sum3=-sum(sum(Phi.*log(Phi)));

Sum4=gammaln(sum(Alpha))-gammaln(sum(Gam(d,:)));  % Using gammaln function for high values to evaluate log(gamma) function

Sum5=sum(gammaln(Gam(d,:))-gammaln(Alpha));

f=Sum1+Sum2+Sum3+Sum4+Sum5;
end



