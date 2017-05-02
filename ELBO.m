function f=ELBO(Mu,Sig,Sig1,X,Phi,K,N)               % Evidence Lower bound function
Sum1=0;
Sum2=0;
for k=1:K
    Sum1=Sum1-0.5*(size(Sig1{1},2)*log(det(Sig{k})) + trace( (Sig1{k}^-1)*(Sig{k}+Mu{k}*Mu{k}')));
end

for i=1:N
    for k=1:K
        
        Sum2=Sum2+Phi(i,k)*(X{i}'*Mu{k}-0.5*trace(Sig{k}+Mu{k}*Mu{k}') -log(Phi(i,k)));
    end
end

f=Sum1+Sum2;

end