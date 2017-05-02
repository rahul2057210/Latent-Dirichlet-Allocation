% Coordinate Ascent Variational Inference(CAVI) Algorithm
clc
clear all

%% Generating data
Mu0={}; % Prior Values of mean vector
X={};   % Data points
K=5; % Number of Clusters
N=1000; % Number of Data points
D=2; % Number of Dimensions
epsilon=0.00001; % Convergence limit
Iter=0; % Number of Iterations
for k=1:K
    Mu0{k}=mvnrnd(zeros(D,1),20*eye(D),1)';
end

P=(1/K)*ones(1,K);  % Probability vector for categorical distribution

for i=1:N
    C=mnrnd(1,P,1);
    sum1=zeros(D,1);
    for r=1:K
        sum1=sum1+C(r)*Mu0{r};
    end
    
    
    X{i}=mvnrnd(sum1,eye(D),1)';
end

A=[];
for i=1:N
    A=[A,X{i}];
end



%% CAVI Algorithm

Mu={};  % Variational Parameters (unknown)
Sig={}; % Variational Parameters (unknown)
Sig1={}; % Variance parameters (Prior)

for k=1:K
    Sig1{k}=20*eye(D);
    Mu{k}=mvnrnd(zeros(D,1),20*eye(D),1)';
    Sig{k}=20*eye(D);
end



Phi=(1/K)*ones(N,K);
A1=ELBO(Mu,Sig,Sig1,X,Phi,K,N);  % ELBO function calculates Evidence lower bound
B1=0;
elbo=[];

while norm((A1-B1))> epsilon
    
    
    for i=1:N
        R=zeros(1,K);
        for k=1:K
            R(k)=(X{i}'*Mu{k}-(trace(Sig{k}+Mu{k}*Mu{k}'))/2);
        end
        t=max(R);
        R=R-t;
        Phi(i,:)=exp(R)/sum(exp(R));
    end
    
    Mu={};
    Sig={};
    for k=1:K
        Sig{k}=(Sig1{k}\eye(D) + sum(Phi(:,k))*eye(D))\eye(D);
        sum1=zeros(D,1);
        for r=1:N
            sum1=sum1+Phi(r,k)*X{r};
        end
        
        Mu{k}=Sig{k}*(sum1);
    end
    
    B1=A1;
    A1=ELBO(Mu,Sig,Sig1,X,Phi,K,N);
    Iter=Iter+1;
    elbo=[elbo;A1];
    
    
end

%plot(1:Iter,elbo)  % Plot of elbo values vs Number of iterations


    


scatter(A(1,:),A(2,:))   % Scatter plot used for D(dimension)=2
hold on

for k=1:K
    
x = Mu{k}(1)-2*(Sig{k}(1,1))^0.5:0.1:Mu{k}(1)+2*(Sig{k}(1,1))^0.5 ; %// x axis
y = Mu{k}(2)-2*(Sig{k}(2,2))^0.5:0.1:Mu{k}(2)+2*(Sig{k}(2,2))^0.5 ; %// y axis
[X Y] = meshgrid(x,y); %// all combinations of x, y
Z = mvnpdf([X(:) Y(:)],Mu{k}',Sig{k}); %// compute Gaussian pdf
Z = reshape(Z,size(X)); %// put into same size as X, Y
contour(X,Y,Z,'red')
hold on
end










