
clc
clear all

%% Simulating Data 
N=100;
D=85;
Iter=0;
epsilon=0.0001;
X=normrnd(0,1,N,D);
Beta=zeros(D,1);
Beta(1:5)=3.1;

Y=X*Beta + normrnd(0,2.5,N,1);

%% VI Algorithm

%Intialization
A=0.01;
B=0.01;
Sig_B=10;
Tau=1000;
rho=exp(-0.5*(N^0.5))/(1+exp(-0.5*(N^0.5)));
w=ones(D,1);
Z=ones(N,1);
Y=(eye(N)-Z*((Z'*Z)\eye(1))*Z')*Y;      % Frisch Waugh Lowell property


X=(eye(N)-Z*((Z'*Z)\eye(1))*Z')*X;       % Frisch Waugh Lowell property

s=(A+(N/2))/Tau;

lb=0;

ub=10;

Ohm=w*w'+ diag(w).*(eye(D)-diag(w));
while norm(ub-lb)>epsilon
    
    Sigma=(Tau*((X'*X).*Ohm) + (Sig_B^-1)*eye(D))\eye(D);
    Mu=Tau*Sigma*diag(w)*(X'*Y);
    
    for j=1:D
        Z=X;
        Z(:,j)=[];
        w1=w;
        w1(j)=[];
        W1=diag(w1);
        Mu1=Mu;
        Mu1(j)=[];
        Sig1=Sigma(:,j);
        Sig1(j)=[];
        %r1=Mu(j)*(X(:,j)'*Y) -X(:,j)'*Z*W1*(Mu1*Mu(j) + Sig1);
        n=(log(rho/(1-rho)) -0.5*Tau*(Mu(j)^2+Sigma(j,j))*(norm(X(:,j))^2) + Tau*(Mu(j)*(X(:,j)'*Y) -(X(:,j)')*Z*W1*(Mu1*Mu(j) + Sig1)));
        w(j)=1/(1+exp(-n));
        
    end
    
    Ohm=w*w'+ diag(w).*(eye(D)-diag(w));
    s=B + 0.5*( norm(Y)^2 -2*Y'*X*diag(w)*Mu + trace( (X'*X .* Ohm)*(Mu*Mu' + Sigma)));
    Tau=(A+0.5*N)/s;
    
    lb=ub;
    ub=elbo(D,N,Sig_B,A,B,rho,w,Mu,Sigma,s);
    Iter=Iter+1;
end

y=zeros(D,1);

for j=1:D
    if w(j)>0.5
        y(j)=1;
    end
end
y
Iter












