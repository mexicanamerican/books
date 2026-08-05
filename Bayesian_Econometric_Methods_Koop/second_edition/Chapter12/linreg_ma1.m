% MA(1) errors
clear; clc;
nsim = 25000; burnin = 5000;

% load data
data_raw = load('YANKEES.txt');
k = 4;
y = data_raw(:,2);
T = length(y);
X = [ones(T,1) data_raw(:,[5 6 8])];

% prior
beta0 = zeros(k,1); iVbeta0 = speye(k)/100; % mean and precision matrix
nu0 = 0; S0 = .001;
psi0 = 0; Vpsi0 = .09;

% initialize the Markov chain
psi = 0;
beta = (X'*X)\(X'*y);
h = T/sum((y-X*beta).^2);
Hpsi = speye(T) + sparse(2:T,1:(T-1),psi*ones(1,T-1),T,T); 

% define a few things
n_grid = 299; % number of grid points for psi
store_theta = zeros(nsim,k+2);  % [beta' h psi]
store_p_psi0 = zeros(nsim,1); % posterior of psi at 0

for isim = 1:nsim + burnin
          % sample beta    
    X_star = Hpsi\X; y_star = Hpsi\y;    
    Dbeta = (iVbeta0 + X_star'*X_star*h)\speye(k); 
    beta_hat = Dbeta*(iVbeta0*beta0 + X_star'*y_star*h);    
    beta = beta_hat + chol(Dbeta,'lower')*randn(k,1);
    
         % sample h
    u = y - X*beta;
    e = Hpsi\u;
    h = gamrnd((nu0+T)/2,2/(S0*nu0 + e'*e)); 
    
        % sample psi        
    psi_grid = linspace(-1+.01*rand,1-.01*rand,n_grid)'; % uniform grid with random end pts
    psi_grid = sort([psi_grid;0]);  % insert 0    
    idx_0 = find(psi_grid==0);      % index for 0
    lp_psi = zeros(n_grid+1,1);     % log posterior density    
    for igrid = 1:n_grid+1 
        psi_i = psi_grid(igrid);
        lp_psi(igrid) = llike_MA1(psi_i,y-X*beta,h) ...
            -.5*(psi_i-psi0)^2/Vpsi0;
    end    
    p_psi = exp(lp_psi-max(lp_psi));  % exponentiate the log-density
    p_psi = p_psi/(sum(p_psi)*(psi_grid(2)-psi_grid(1))); % normalize
    cdf_psi = cumsum(p_psi); % cdf of psi
    cdf_psi = cdf_psi/cdf_psi(end); % normalize
    psi = psi_grid(find(cdf_psi>rand,1));
    Hpsi = speye(T) + sparse(2:T,1:(T-1),psi*ones(1,T-1),T,T);    
    
    if (mod(isim, 5000) == 0)
        disp([num2str(isim) ' loops... ']);
    end
    
        % store the parameters
    if isim > burnin
        isave = isim - burnin;
        store_theta(isave,:) = [beta' h psi];    
        store_p_psi0(isave,:) = p_psi(idx_0);        
    end
end
theta_hat = mean(store_theta)
theta_CI = quantile(store_theta,[.025 .975])

% plot histogram of psi
hist(store_theta(:,end),50); box off;

%% compute the SDDR
post_psi0 = mean(store_p_psi0);
pri_psi0 = normpdf(0,psi0,sqrt(Vpsi0))...
    /(normcdf(1,psi0,sqrt(Vpsi0))-normcdf(-1,psi0,sqrt(Vpsi0)));
BF_RU = post_psi0/pri_psi0;


