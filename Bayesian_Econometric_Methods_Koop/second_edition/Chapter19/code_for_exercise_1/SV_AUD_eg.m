clear; clc;

nsim = 20000;
burnin = 1000;
    % initialize for storage
store_theta = zeros(nsim,3); % [mu h0 sigh2]clear; clc;
load 'AUDUSD.csv';
y = AUDUSD; 
T = length(y);

    % prior
mu0 = 0; Vmu = 100;
a0 = 0; b0 = 100;
nu_h = 3; S_h = .2*(nu_h-1);

    % initialize the Markov chain
sigh2 = .05;
mu = mean(y);
h0 = log(var(y));
h = h0*ones(T,1);
H = speye(T) - sparse(2:T,1:(T-1),ones(1,T-1),T,T);
HH = H'*H;

store_h = zeros(nsim,T);

for isim = 1:nsim + burnin
        % sample mu    
    Kmu = 1/Vmu + sum(1./exp(h));
    mu_hat = Kmu\(mu0/Vmu + sum(y./exp(h)));
    mu = mu_hat + sqrt(Kmu)\randn;        
    
        % sample h
    ystar = log((y-mu).^2 + .0001);
    h = SVRW(ystar,h,h0,sigh2);
    
        % sample sigh2        
    sigh2 = 1/gamrnd(nu_h + T/2, 1/(S_h + (h-h0)'*HH*(h-h0)/2));    
    
        % sample h0
    Kh0 = 1/b0 + 1/sigh2;
    h0_hat = Kh0\(a0/b0 + h(1)/sigh2);
    h0 = h0_hat + sqrt(Kh0)'\randn;

    if (mod(isim, 5000) == 0)
        disp([num2str(isim) ' loops... ']);
    end    
    if isim > burnin
        isave = isim - burnin;
        store_h(isave,:) = h'; 
        store_theta(isave,:) = [mu h0 sigh2];
    end    
end

theta_hat = mean(store_theta);
theta_CI = quantile(store_theta,[.025 .975]);
h_hat = mean(exp(store_h/2))'; 

tt = linspace(2005,2013,T)';
figure;
plot(tt,h_hat); box off; xlim([2005 2013]);
set(gcf,'Position',[100 100 800 300]);