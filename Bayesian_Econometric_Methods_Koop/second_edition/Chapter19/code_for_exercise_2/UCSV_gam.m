% UCSV_gam.m
clear; clc; 
nsim = 20000;
burnin = 1000;
data_raw = load('USCPI_2015Q4.csv');
data = 400*log(data_raw(2:end)./data_raw(1:end-1));
y = data;
T = length(y);

valh = 0;
valg = 0;

%% prior
a0_h = 0; b0_h = 10;
a0_g = 0; b0_g = 10;
a0_tau = 0; b0_tau = 10;
Vomegah = .2;
Vomegag = .2;
    
% initialize the Markov chain
h0 = log(var(y))/5; g0 = log(var(y))/10; tau0 = mean(y);
omegah = sqrt(.2);
omegag = sqrt(.2);
h_tilde = zeros(T,1);
g_tilde = zeros(T,1);
h = h0 + omegah*h_tilde;
g = g0 + omegah*g_tilde; 

% define a few things
n_grid = 500; % number of grid points
omh_grid = linspace(-1,1,n_grid)';
omg_grid = linspace(-1,1,n_grid)';
H = speye(T) - sparse(2:T,1:(T-1),ones(1,T-1),T,T);

% initialize for storeage
store_theta = zeros(nsim,5); % [omegah omegag h0 g0 tau0]
store_tau = zeros(nsim,T); 
store_h = zeros(nsim,T);
store_g = zeros(nsim,T);
store_pomh = zeros(n_grid,1);
store_pomg = zeros(n_grid,1);
store_lpden = zeros(nsim,3); % log posterior densities of 
                             % omegah = valh; omegag = valg;
                             % omegah = omegag 
                            
rand('state', sum(100*clock) ); randn('state', sum(200*clock) );    
for isim = 1:nsim+burnin
    
        % sample tau    
    iOh = sparse(1:T,1:T,1./exp(h));
    HiOgH = H'*sparse(1:T,1:T,1./exp(g))*H;
    Ktau =  HiOgH + iOh;    
    tau_hat = Ktau\(tau0*HiOgH*ones(T,1) + iOh*y);
    tau = tau_hat + chol(Ktau,'lower')'\randn(T,1);
    
        % sample h_tilde 
    ystar = log((y-tau).^2 + .0001);
    [h_tilde,h0,omegah,omegah_hat,Domegah] = ...
        SVRW_gam(ystar,h_tilde,h0,omegah,a0_h,b0_h,Vomegah); 
    h = h0 + omegah*h_tilde;    
    
        % sample g_tilde
    ystar = log((tau-[tau0;tau(1:end-1)]).^2 + .0001);
    [g_tilde,g0,omegag,omegag_hat,Domegag] = ...
        SVRW_gam(ystar,g_tilde,g0,omegag,a0_g,b0_g,Vomegag); 
    g = g0 + omegag*g_tilde;
    
        % sample tau0
    Ktau0 = 1/b0_tau + 1/exp(g(1));
    tau0_hat = Ktau0\(a0_tau/b0_tau + tau(1)/exp(g(1)));
    tau0 = tau0_hat + sqrt(Ktau0)'\randn;
            
    if (mod(isim, 5000) == 0)
        disp([num2str(isim) ' loops... ']);
    end     
    
    if isim > burnin
        isave = isim - burnin;
        store_tau(isave,:) = tau';
        store_h(isave,:) = h'; 
        store_g(isave,:) = g'; 
        store_theta(isave,:) = [omegah omegag h0 g0 tau0]; 

        lh0 = -.5*log(2*pi*Domegah) - .5*(omegah_hat-valh)^2/Domegah;
        lg0 = -.5*log(2*pi*Domegag) - .5*(omegag_hat-valg)^2/Domegag;
        lhg0 = lh0 + lg0;
        store_lpden(isave,:) = [lh0 lg0 lhg0];
        
        store_pomh = store_pomh + normpdf(omh_grid,omegah_hat,sqrt(Domegah));
        store_pomg = store_pomg + normpdf(omg_grid,omegag_hat,sqrt(Domegag));        
    end    
end

theta_hat = mean(store_theta)';
tau_hat = mean(store_tau)';
h_hat = mean(exp(store_h/2))'; 
g_hat = mean(exp(store_g/2))'; 

pomhhat = store_pomh/nsim;
pomghat = store_pomg/nsim;
priden_omh = normpdf(omh_grid,0,sqrt(Vomegah));
priden_omg = normpdf(omg_grid,0,sqrt(Vomegag));
maxlpden = max(store_lpden);
lpostden = log(mean(exp(store_lpden-repmat(maxlpden,nsim,1)))) + maxlpden;
lpriden = [log(normpdf(valh,0,sqrt(Vomegah))) log(normpdf(valg,0,sqrt(Vomegag)))];
lpriden(3) = sum(lpriden(1:2));
lBF = lpriden-lpostden;

tt = (1947.25:.25:2015.75)';
figure;
plot(tt,[tau_hat y]); xlim([1947 2016]); box off;
set(gcf,'Position',[100 100 800 300]);

figure;
plot(tt,[h_hat g_hat]); xlim([1947 2016]); box off;
set(gcf,'Position',[100 100 800 300]);

figure;
subplot(1,2,1); hist(store_theta(:,1),50); box off;
subplot(1,2,2); hist(store_theta(:,2),50); box off;
set(gcf,'Position',[100 100 800 300]);