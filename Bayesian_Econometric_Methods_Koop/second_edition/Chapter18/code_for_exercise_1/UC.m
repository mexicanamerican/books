% UC.m
clear; clc; 
nsim = 10000;
burnin = 1000;
data_raw = load('USPCE_2015Q4.csv');
data = 400*log(data_raw(2:end)./data_raw(1:end-1));
y = data;
T = length(y);

    % initialize for storage
store_tau = zeros(nsim,T);
store_theta = zeros(nsim,3);  % [sig2,omega2,tau0]

    % prior
a0 = 5; b0 = 100;
nu_sig0 = 3; S_sig0 = 1*(nu_sig0-1);
nu_omega0 = 3; S_omega0 = .25^2*(nu_omega0-1);

    % initialize the Markov chain
sig2 = 1; omega2 = .1; tau0 = 5;
    
    % compute a few things outside the loop
H = speye(T) - sparse(2:T,1:(T-1),ones(1,T-1),T,T);
HH = H'*H;
HHiota = HH*ones(T,1);

for isim = 1:nsim+burnin

        % sample tau
    Ktau = HH/omega2 + speye(T)/sig2;    
    tau_hat = Ktau\(tau0/omega2*HHiota + y/sig2);
    Ctau = chol(Ktau,'lower');
    tau = tau_hat + Ctau'\randn(T,1); 
   
        % sample sig2    
    sig2 = 1/gamrnd(nu_sig0 + T/2,1/(S_sig0 + (y-tau)'*(y-tau)/2));
    
        % sample omega2        
    omega2 = 1/gamrnd(nu_omega0 + T/2, ...
        1/(S_omega0 + (tau-tau0)'*HH*(tau-tau0)/2));    
    
        % sample tau0
    Ktau0 = 1/b0 + 1/omega2;
    tau0_hat = Ktau0\(a0/b0 + tau(1)/omega2);
    tau0 = tau0_hat + sqrt(Ktau0)'\randn;
    
    if isim>burnin
        i = isim-burnin;
        store_tau(i,:) = tau';      
        store_theta(i,:) = [sig2 omega2 tau0];
    end    
end

theta_hat = mean(store_theta)
theta_CI = quantile(store_theta,[.025 .975])
tau_hat = mean(store_tau)';

tt = (1959.25:.25:2015.75)';
figure;
hold on
    plot(tt,tau_hat); xlim([1959 2016]); 
    plot(tt,y,'--r'); xlim([1959 2016]); 
hold off
box off;
legend('Posterior mean of \tau_t','PCE data');
set(gcf,'Position',[100 100 800 300]);

