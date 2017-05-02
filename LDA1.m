clc
clear all
%% LDA (Latent Dirichlet Allocation)
% Initialization
tic
load ('20ng.mat')
D=3000; % Number of Documents in the corpus
N=[]; % Number of words in each document
K=50;  % Number of topics
V=2000; % Number of words in the vocabulary
epsilon=0.001; % Convergence
Iter=0; % Iteration
V1=vocabulary;

W={}; % Document
Corpus=full(wordsTrain);  % Collection of documents

A=eye(V);   
for j=1:D                   % Reading document
W{j}=sparse(repelem(A,Corpus(:,j)',1));
N=[N;size(W{j},1)];     % Storing the length of each document in a vector
j;
end


%B=(1/V)*ones(K,V);
a=3*ones(1,V);
B=drchrnd(a,K);
Alpha=(2.2)*ones(1,K);
Gam=(1/K)*ones(D,K);
A1=zeros(1,K);
B1=zeros(K,V);
Phi1={};

%% Variational EM Algorithm

while norm(Alpha-A1)>epsilon && norm(B-B1)>epsilon && Iter<35

% Updating Variational Parameters    
for d=1:D
    a=0;
    b=10;
    
    while norm(b-a)>epsilon
    P=zeros(N(d),K);
    for n=1:N(d)
       r=B*(W{d}(n,:)').*( exp(psi(Gam(d,:))-psi(sum(Gam(d,:)))))';
       P(n,:)=r/sum(r);
    end
    Gam(d,:)=Alpha+sum(P);
    a=b;
    b=elbo1(Gam,Alpha,B,P,W,K,d,V);
    
    end
    d;
end
    display('Updation of variational parameter completed')

A1=Alpha;
B1=B;

% Updating Beta

Sum33=zeros(K,V);
for d=1:D
Sum33=Sum33+Phi1{d}'*W{d};
end
B=Sum33.*repmat(sum(Sum33,2).^-1,1,V);

% Updating Alpha



Alpha_old=Alpha;
g=D*psi(sum(Alpha_old))*ones(1,K)-D*psi(Alpha_old)+sum(psi(Gam)) -sum(psi(sum(Gam,2)))*ones(1,K);
h=-D*psi(1,Alpha_old);
z=D*psi(1,sum(Alpha_old));
c=sum(g.*(h.^-1))/(z^-1 + sum(h.^-1));
a=(g-c).*(h.^-1);
Alpha_new=Alpha_old-a;
while norm(Alpha_old-Alpha_new)>epsilon
    g=D*psi(sum(Alpha_old))*ones(1,K)-D*psi(Alpha_old)+sum(psi(Gam)) -sum(psi(sum(Gam,2)))*ones(1,K);
    h=-D*psi(1,Alpha_old);
    z=D*psi(1,sum(Alpha_old));
   
    c=sum(g.*(h.^-1))/(z^-1 + sum(h.^-1));
    a=(g-c).*(h.^-1);
    r=Alpha_new;
    
    Alpha_new=Alpha_old-a;
    Alpha_old=r;
end
Alpha=Alpha_new;
Iter=Iter+1;
end

% Gam
% Iter
% b    
B?
for k=1:K
    z=B(k,:);
    [a1,a2]=sort(z); % a2 stores the permutation of sorted matrix
    for j=1:10
        disp(V1{a2(length(z)-j)})        % Printing top 10 words from each topic
    end
    fprintf('\n')
end
toc










