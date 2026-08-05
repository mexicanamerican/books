% UC_noncen_sim.m
clear; clc; 
nsim = 20000; burnin = 1000;

% generate data
T = 250;    
randn('seed',123456); rand('seed',789012);
alp0 = .1; tau0 = 3; sigy2 = .5;
sigalp = .1; sigtau = 0;
H = speye(T) - sparse(2:T,1:(T-1),ones(1,T-1),T,T);
HH = H'*H;
alp_tilde = chol(HH,'lower')'\randn(T,1);
tau_tilde = chol(HH,'lower')'\randn(T,1);
alp = alp0 + sigalp*alp_tilde;
tau = tau0 + sigtau*tau_tilde + (1:T)'*alp0 + sigalp*(H\alp_tilde);
y = tau + sqrt(sigy2)*randn(T,1);
truealp = alp; truetau = tau; 

    % initialize for storage
store_tau = zeros(nsim,T);
store_alp = zeros(nsim,T);
store_theta = zeros(nsim,5);  
store_psigtau = zeros(nsim,1);

    % prior
beta0 = zeros(4,1); iVbeta = diag([1 1 .1 .1]);
nu_sig0 = 3; S_sig0 = 1*(nu_sig0-1); 
     
% compute a few things outside the loop
HH = H'*H;
H2 = H*H;
H2H2 = H2'*H2;
Pgam = [HH sparse(T,T); sparse(T,T) H2H2];

    % initialize the Markov chain
sigy2 = var(y); 
alp0 = 0; tau0 = 0;
sigtau = .1; sigalp = .1;

tic;
for isim = 1:nsim+burnin    
        % sample alp and tau
    Xgam = [sigtau*speye(T) sigalp*speye(T)];   
    Kgam = Pgam + Xgam'*Xgam/sigy2;    
    gam_hat = Kgam\(1/sigy2*Xgam'*(y-tau0-(1:T)'*alp0));    
    gam = gam_hat + chol(Kgam,'lower')'\randn(2*T,1); 
        
        % sample beta
    X = [ones(T,1) (1:T)' gam(1:T) gam(T+1:end)];
    beta = sample_beta(y,X,sigy2,beta0,iVbeta);
    tau0 = beta(1); alp0 = beta(2); sigtau = beta(3); sigalp = beta(4);
    
        % permutate the signs of gam and (sigtau, sigalp)        
    U = -1 + 2*(rand>0.5);
    gam = U*gam;
    sigalp = U*sigalp;
    sigtau = U*sigtau;
    
        % compute tau and alp
    tau_tilde = gam(1:T);
    A_tilde = gam(T+1:end);
    alp_tilde = H*A_tilde;
    alp = alp0 + sigalp*alp_tilde;
    tau = tau0 + sigtau*tau_tilde + (1:T)'*alp0 + sigalp*(H\alp_tilde);
       
        % sample sigy2    
    sigy2 = 1/gamrnd(nu_sig0 + T/2,1/(S_sig0 + (y-tau)'*(y-tau)/2));    
    
    if (mod(isim, 1000) ==0)
        disp([num2str(isim) ' loops... ']);
    end
    
    if isim>burnin        
        isave = isim-burnin;
        store_tau(isave,:) = tau';
        store_alp(isave,:) = alp';
        store_theta(isave,:) = [beta' sigy2];
        
            % compute p(sigtau = 0 | y,beta,sigy2)
        [~, beta_hat, Kbeta] = sample_beta(y,X,sigy2,beta0,iVbeta);
        Dbeta = Kbeta\speye(4);
        psigtau = normpdf(0,beta_hat(3),sqrt(Dbeta(3,3))); 
        store_psigtau(isave) = psigtau;
    end
end
toc;
theta_hat = mean(store_theta)
theta_CI = quantile(store_theta,[.025 .975])
tau_hat = mean(store_tau)';
alp_hat = mean(store_alp)';

% figure;
% subplot(2,1,1);
% plot([truetau tau_hat]); box off;
% subplot(2,1,2);
% plot([truealp alp_hat]); box off;
% set(gcf,'Position',[100 100 800 300]);

figure;
subplot(1,2,1);
hist(store_theta(:,3),50); box off;
title('Posterior draws of \sigma_{\tau}');
subplot(1,2,2);
hist(store_theta(:,4),50); box off;
title('Posterior draws of \sigma_{\alpha}');
set(gcf,'Position',[100 100 800 300]);

% SSDR
pri_sigtau = normpdf(0,beta0(3),1/sqrt(iVbeta(3,3)));
BF = pri_sigtau/mean(store_psigtau)