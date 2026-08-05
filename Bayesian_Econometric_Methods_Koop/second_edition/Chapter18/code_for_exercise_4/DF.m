% The dynamic factor model
clear; clc;
nsim = 5000;
burnin = 1000;
data = load('freddata_Q.csv');
data_mean = mean(data)';
data_std = std(data)';
idx = ~isnan(data_mean); % extract variables that don't have missing data
n = sum(idx);
T = size(data,1);
Y = (data(:,idx) - repmat(data_mean(idx)',T,1))./repmat(data_std(idx)',T,1);
y = reshape(Y',T*n,1);
k = 1; % # of factors
 
%% initialize for storage
store_F = zeros(nsim,T,k);
store_Lambda = zeros(nsim,n,k);
store_Sig_y = zeros(nsim,n);
store_Sig_f = zeros(nsim,k);
store_phi = zeros(nsim,k^2);
 
%% prior
nLambda = (2*n-k-1)*k/2; % # of free elements in Lambda
VLambda = 100 * ones(nLambda,1);
phi0 = zeros(k^2,1); Vphi = 100 * ones(k^2,1);
nu_y = 5; S_y = (nu_y-1)*ones(n,1);
nu_f = 5; S_f = (nu_f-1)*ones(k,1);

%% initialize the Markov chain
Sig_y = var(Y)';
Sig_f = ones(k,1);
Phi = .6*eye(k);
Lambda = zeros(n,k);
Lambda(1:n+1:end) = 1; % set diagonal elements to 1

hzeros = sparse(k,(T-1)*k); 
vzeros = sparse(T*k,k);
HPhi = speye(T*k) - cat(2,cat(1,hzeros,kron(speye(T-1),Phi)),vzeros);

rand('state', sum(100*clock) ); randn('state', sum(200*clock) );
 
disp('Starting MCMC for the dynamic factor model.... ');
start_time = clock;

for isim = 1:nsim + burnin
        % sample f
    Kf = HPhi'*sparse(1:T*k,1:T*k,repmat(1./Sig_f,T,1))*HPhi ...
        + kron(speye(T),Lambda'*sparse(1:n,1:n,1./Sig_y)*Lambda);
    f_hat = Kf\(kron(speye(T),Lambda'*sparse(1:n,1:n,1./Sig_y))*y);    
    f = f_hat + chol(Kf,'lower')'\randn(T*k,1);
    F = reshape(f,k,T)';
    
        % sample Lambda -- equation by equation
    count_Lam = 0;
    for isave = 2:n         
        if isave <= k
            k_i = isave-1;
            Xf = F(:,1:k_i); 
            K_Lami = sparse(1:k_i,1:k_i,1./VLambda(count_Lam+1:count_Lam+k_i)) ...
                + Xf'*Xf/Sig_y(isave);
            Lami_hat = K_Lami\(Xf'*(Y(:,isave)-F(:,isave))/Sig_y(isave));            
        else
            k_i = k;
            Xf = F;
            K_Lami = sparse(1:k_i,1:k_i,1./VLambda(count_Lam+1:count_Lam+k_i)) ...
                + Xf'*Xf/Sig_y(isave);
            Lami_hat = K_Lami\(Xf'*Y(:,isave)/Sig_y(isave));            
        end
        Lambda(isave,1:k_i) = Lami_hat + chol(K_Lami,'lower')'\randn(k_i,1);
        count_Lam = count_Lam + k_i;
    end    
    
        % sample Sig_y
    E_y = Y - F * Lambda';
    newS_y = S_y +  sum(E_y.^2)'/2;
    Sig_y = 1./gamrnd(nu_y + T/2, 1./newS_y);    
    
        % sample Sig_f
    E_f = [F(1,:); F(2:end,:) - F(1:end-1,:)*Phi];
    newS_f = S_f +  sum(E_f.^2)'/2;
    Sig_f = 1./gamrnd(nu_f + T/2, 1./newS_f);    
    
        % sample Phi 
    Zf = SURform2([zeros(1,k); F(1:end-1,:)],k);
    Kphi = sparse(1:k^2,1:k^2,1./Vphi) ...
        + Zf'*sparse(1:T*k,1:T*k,repmat(1./Sig_f,T,1))*Zf;
    phi_hat = Kphi\(phi0./Vphi ...
        + Zf'*sparse(1:T*k,1:T*k,repmat(1./Sig_f,T,1))*reshape(F',k*T,1));
    phi = phi_hat + chol(Kphi,'lower')'\randn(k^2,1);
    Phi = reshape(phi,k,k)';
    HPhi = speye(T*k) - cat(2,cat(1,hzeros,kron(speye(T-1),Phi)),vzeros);  
    
    if isim > burnin
        isave = isim - burnin;
        store_F(isave,:,:) = F;
        store_Lambda(isave,:,:) = Lambda;        
        store_Sig_y(isave,:) = Sig_y';
        store_Sig_f(isave,:) = Sig_f';
        store_phi(isave,:) = phi';
    end
    
     if (mod(isim,2000) == 0)
        disp([num2str(isim) ' loops... '])
    end
    
end

disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
disp(' ' );

F_hat = squeeze(mean(store_F));
Lambda_hat = squeeze(mean(store_Lambda));

Sig_y_hat = mean(store_Sig_y)';
Sig_f_hat = mean(store_Sig_f)';
Phi_hat = reshape(mean(store_phi),k,k)';

dates = linspace(1959.75,2015.75,T)';
figure;
hold on
    plot(dates,f_hat); 
    hline = refline(0,0);
    hline.Color = 'k';    
hold off
box off; 
xlim([dates(1) dates(end)]); 
ylim([-1 1]); 
set(gcf,'Position',[100 100 800 300]);
