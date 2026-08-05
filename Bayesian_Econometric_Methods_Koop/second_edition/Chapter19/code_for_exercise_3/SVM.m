clear; clc;
nsim = 20000;
burnin = 1000;

    % prepare the data
SP500_index = xlsread('SP500.csv');
DFF = xlsread('DFF.csv');
id = find(SP500_index(:,2)~=0);
SP500 = [SP500_index(id(2:end),1) ...
    100*log(SP500_index(id(2:end),2)./SP500_index(id(1:end-1),2))];
T = size(SP500,1);
y = zeros(T,1);
for t = 1:T
    id = find(SP500(t,1) == DFF(:,1));    
    y(t) = SP500(t,2) - DFF(id,2);    
end

    % prior
gam0 = zeros(2,1); iVgam = speye(2)/100;
nu_h = 3; S_h = .2^2*(nu_h-1);
a0 = 0; b0 = 100;

    % initialize the Markov chain
alp = 0;
mu = mean(y);
h0 = log(var(y));
sigh2 = .2;
h = sample_SVM_h(y,alp,mu,h0*ones(T,1),h0,sigh2,1);

    % initialize for storeage
store_theta = zeros(nsim,3); % [mu alp sigh2]
store_h = zeros(nsim,T);

    % compute a few things outside the loop
H = speye(T) - sparse(2:T,1:(T-1),ones(1,T-1),T,T);
HH = H'*H;
count_h = 0;
start_time = clock;

for isim = 1:nsim+burnin
        % sample mu and alp
    X = [ones(T,1) exp(h)];
    iSy = sparse(1:T,1:T,1./exp(h));
    XiSy = X'*iSy;
    Kgam = iVgam + XiSy*X;
    gam_hat = Kgam\(iVgam*gam0 + XiSy*y);
    gam = gam_hat + chol(Kgam,'lower')'\randn(2,1);
    mu = gam(1);  alp = gam(2);
    
        % sample h
    [h,flag] = sample_SVM_h(y,alp,mu,h,h0,sigh2);
    count_h = count_h + flag;
    
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
        store_h(isave,:)  = h';         
        store_theta(isave,:) = [mu alp sigh2];
    end
    
end

disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
disp(' ' );

h_hat = mean(exp(store_h/2))';
theta_hat = mean(store_theta)
theta_CI = quantile(store_theta,[.025 .975])

tt = linspace(2013,2016,T)'; 
figure;
plot(tt,h_hat); box off; xlim([2013 2016]);
set(gcf,'Position',[100 100 800 300]);

% figure;
% plot(tt,y); box off; xlim([2013 2016]);
% set(gcf,'Position',[100 100 800 300]);
