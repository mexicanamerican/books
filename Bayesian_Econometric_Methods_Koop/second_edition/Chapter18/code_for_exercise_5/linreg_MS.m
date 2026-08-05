% Markov switching model 
% y_t = mu0 + mu1 s_t + u_t, u_t~N(0,sig2_{s_t})

clear; clc;
nsim = 20000; burnin = 1000;
   
%     % generate data
% mu = [.7 .1]';
% sig2 = [1.5, .2]'/10;
% a = .03;
% b = .05;
% T = 500;
% S = zeros(T,1); S(1) = (rand < .5);
% for t=2:T
%     if S(t-1) == 0
%         S(t) = (rand < a);
%     else
%         S(t) = 1 - (rand < b);
%     end
% end
% y = mu(1) + mu(2)*S + sqrt(sig2(S+1)).*randn(T,1);
% truetheta =  [mu' sig2' a b]';
% trueS = S;

    % load data
load usgdp.txt;
y = (log(usgdp(2:end)) - log(usgdp(1:end-1)))*100;
T = size(y,1);

    % prior
mu0 = [.6,.1]'; iVmu = eye(2);
nu0 = 3; S0 = 1*(nu0 - 1);
a0 = 1; a1 = 9;
b0 = 1; b1 = 9;
 
    % constuct a few things
store_theta = zeros(nsim,6); %[mu,sig2,a,b]
store_S = zeros(T,2);
F = zeros(T,2);

    % initialize the Markov chain    
a = .05; b = .05;
S = zeros(T,1);
mu = [.6 .1]';
sig2 = ones(2,1);

for isim = 1:nsim + burnin
    
    % sample S
        % filtering
    F(1,:) = normpdf([y(1) y(1)],[mu(1) sum(mu)],sqrt(sig2)');
    F(1,:) = F(1,:)/sum(F(1,:));
    P = [1-a,a;b,1-b];
    for t = 2:T
        tmp = (F(t-1,:)*P).*normpdf([y(t) y(t)],[mu(1) sum(mu)],sqrt(sig2)');
        F(t,:) = tmp/sum(tmp);
    end      
        % smoothing
    S(T) = (F(end,2) > rand);
    for t = T-1:-1:1
        prob = F(t,:).*P(:,S(t+1)+1)';
        prob = prob/sum(prob);
        S(t) = (prob(2) > rand);
    end
    
        % sample mu
    X = [ones(T,1) S];
    XiSig = X'*sparse(1:T,1:T,1./sig2(S+1));
    Dmu = (iVmu + XiSig*X)\speye(2);
    mu_hat = Dmu*(iVmu*mu0 + XiSig*y);
    flag = true;
    while flag
        mu = mu_hat + chol(Dmu,'lower')*randn(2,1);
        if mu(2) > 0
            flag = false;
        end
    end
     
        % sample sig2
    e = y - X*mu;
    for i = 0:1
        idx = (S == i);
        Ti = sum(idx);        
        yi = y(idx,:);
        ei = e(idx);
        sig2(i+1) = 1/gamrnd(nu0+Ti/2,1/(S0 + ei'*ei/2));
    end
    
   
    % sample a
    idx_0 = find(S(1:T-1)==0);
    n00 = length(find(S(idx_0+1)==0));
    n01 = length(find(S(idx_0+1)==1));
    a = betarnd(a0+n01,a1+n00);        
    
    %sample b
    idx_1 = find(S(1:T-1)==1);
    n10 = length(find(S(idx_1+1)==0));
    n11 = length(find(S(idx_1+1)==1));
    b = betarnd(b0+n10,b1+n11);    
     
        % store the parameters
    if isim > burnin
        isave = isim - burnin;
        store_theta(isave,:) = [mu' sig2' a b];
        for j=0:1
            store_S(:,j+1) = store_S(:,j+1) + (S == j);
        end        
    end    
    if (mod(isim, 10000) == 0)
        disp([num2str(isim) ' loops... '])
    end   
    
end
theta_hat = mean(store_theta)';
S_hat = store_S/nsim;

% [truetheta theta_hat]
% plot([trueS S_hat(:,2)]); 
% ylim([-.1 1.1]); box off;

tid = linspace(1947,2009.5,T)';
figure;
plot(tid,S_hat(:,1),'linewidth',1); xlim([1940 2010]); box off;


% set(gcf,'Position',[100 100 800 400]);
