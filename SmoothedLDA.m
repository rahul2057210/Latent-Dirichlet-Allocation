%% Smoothed Latent dirichlet allocation
clc
clear all
% % Initialization
% tic
load ('20ng.mat')                % Data file
 D=3000;                          % Number of Documents in the corpus
 N=[];                            % Number of words in each document
 K=50;                            % Number of topics
 V=2000;                          % Number of words in the vocabulary
 epsilon=0.01;                   % Convergence criteria
 Iter=0;                          % Iteration
 V1=vocabulary;
 
 W={};                            % Document
 Corpus=full(wordsTrain);         % Collection of documents
 
 A=eye(V);   
 
 for j=1:D                        % Reading document
 W{j}=sparse(repelem(A,Corpus(:,j)',1));
 N=[N;size(W{j},1)];% Storing number of words in each document in a vector
 j;
 end
 

% 
% toc
%% Inference of variational parameters by VB and global parameters by EM Algorithm
 load('Data.mat')

a=3*ones(1,V);
B=drchrnd(a,K);
Alpha=(2.2)*ones(1,K);
Gam=(1/K)*ones(D,K);
A1=zeros(1,K);
B1=zeros(K,V);




eta    = (0.005);
Lambda = zeros(K,V);
Lambda = Lambda + eta*ones(K,V) ;
Gamma  = (1/K)*ones(D,K);
% Phi    = {};
% A1     = zeros(1,K);
% Alpha  = (2.2)*ones(1,K);
% a=3*ones(1,V);
% B=drchrnd(a,K);
ExpB   = B;
S=20; % Size of minibatch
b=3*ones(1,K);
Phi={};
for d=1:D
    Phi{d}=drchrnd(b,N(d));   % Initializing Phi1
end






% Updating Variational Parameters Phi(z), Gamma(theta), and Lambda(beta) 
tic
while norm(Alpha-A1)>epsilon && Iter<35         % Until Convergence
   t=0;
   Elbo2prev = 0;
   Elbo2curr = 1000102; 
   while norm(Elbo2curr-Elbo2prev)>500000
    Perm=randi([1,D],1,S);
    for d=1:D
        Elbo1prev=0;
        Elbo1curr=1;
        while norm(Elbo1curr-Elbo1prev)>epsilon*100
            phi        = zeros(N(d),K);
                for n  = 1:N(d)
                    r  = ExpB*(W{d}(n,:)').*( exp(psi(Gamma(d,:)))/exp(psi(sum(Gamma(d,:)))))';
%                      r = exp( psi(Gamma(d,:)) - psi(sum(Gamma(d,:))) + psi(Lambda*W{d}(n,:)') - psi(sum(Lambda)) )';
                    phi(n,:) = r/sum(r);
                end
            Gamma(d,:) = Alpha + sum(phi);
            Elbo1prev  = Elbo1curr;
            Elbo1curr  = elbo1(Gamma,Alpha,ExpB,phi,W,K,d,V);
        end
      
        Phi{d} = phi;
    end
    
    Lambda_hat     = zeros(K,V);
    for d=Perm
        Lambda_hat = Lambda_hat + Phi{d}'*W{d};  
    end

    Lambda_hat=eta*ones(K,V)+(D/S)*Lambda_hat;           % Minibatch of S elements
    Rho=(t+1)^(-0.7);
    Lambda=(1-Rho)*Lambda + Rho*Lambda_hat;
    
        Elbo2prev     = Elbo2curr;
        Elbo2curr     = elbo2(Gamma,Alpha,Lambda,Phi,eta*ones(K,V),W,K,D,V);
        for i  = 1:K
            ExpB(i,:) = exp((psi(Lambda(i,:)))- psi(sum(Lambda(i,:),2))) ;
        end
        t=t+1;
   end 
          display('Completion of VParam')

       
%----------------------------------------------------------------------------
A1=Alpha;

Alpha_old=Alpha;
g=D*psi(sum(Alpha_old))*ones(1,K)-D*psi(Alpha_old)+sum(psi(Gamma)) -sum(psi(sum(Gamma,2)))*ones(1,K);
h=-D*psi(1,Alpha_old);
z=D*psi(1,sum(Alpha_old));
c=sum(g.*(h.^-1))/(z^-1 + sum(h.^-1));
a=(g-c).*(h.^-1);
Alpha_new=Alpha_old- 0.5*a;
while norm(Alpha_old-Alpha_new)>epsilon
    g=D*psi(sum(Alpha_old))*ones(1,K)-D*psi(Alpha_old)+sum(psi(Gamma)) -sum(psi(sum(Gamma,2)))*ones(1,K);
    h=-D*psi(1,Alpha_old);
    z=D*psi(1,sum(Alpha_old));
   
    c=sum(g.*(h.^-1))/(z^-1 + sum(h.^-1));
    Elboprev=(g-c).*(h.^-1);
    r=Alpha_new;
    
    Alpha_new=Alpha_old-0.5*Elboprev;
    Alpha_old=r;
end

eta_old=eta;
g=K*V*psi(eta_old*V)-K*V*psi(eta_old)+sum(sum(psi(Lambda)))-V*sum(psi(sum(Lambda,2)));
H=K*psi(1,eta_old*V)*(V^2)-K*V*psi(1,eta_old);
eta_new=eta_old-g/H;
while norm(eta_old-eta_new)>epsilon
eta_old=eta_new;    
g=K*V*psi(eta_old*V)-K*V*psi(eta_old)+sum(sum(psi(Lambda)))-V*sum(psi(sum(Lambda,2)));
H=K*psi(1,eta_old*V)*(V^2)-K*V*psi(1,eta_old);
eta_new=eta_old-g/H;
end

eta = eta_new;

    display('Completion of EM algorithm for alpha and eta')
Alpha=Alpha_new;
Iter = Iter+1;

end

% Gam
% Iter
% Elbocurr    
% B
for k=1:K
    z=ExpB(k,:);
    [a1,a2]=sort(z); % a2 stores the permutation of sorted matrix
    for j=1:10
        disp(V1{a2(length(z)-j)})        % Printing top 10 words from each topic
    end
    fprintf('\n')
end
toc









