%This program provides some MCMC convergence / code accuracy diagnostics in
%the context of a simple regression model. 

clear;
clc;

%-----------------
%Generate the data
%-----------------
nobs = 1000;
beta0_true = 1; beta1_true = -2; sig_true =.25;
eps = sqrt(sig_true)*randn(nobs,1);
x = rand(nobs,1);
X = [ones(nobs,1) x];
y = beta0_true + beta1_true*x + eps;

%---------------
%Choose priors
%---------------
mu_beta = zeros(2,1);
V_beta = 100*eye(2);
iV_beta = (1/100)*eye(2);
a = 3;
b = (1/2);
ib = 1/b;

%---------------------------------
%Choose sets of initial conditions
%---------------------------------
sig_start = [10 .01 23];
beta0_start = [-15 12 0];
beta1_start = [20 8 -13];

iter =1000;
sig_stored = zeros(iter,3);
beta0_stored = sig_stored;
beta1_stored = sig_stored;

for j = 1:3;
    sig = sig_start(j);
    betas = [beta0_start(j) beta1_start(j)];
    sig_stored(1,j) = sig;
    beta0_stored(1,j) = betas(1);
    beta1_stored(1,j) = betas(2);
    for i = 2:iter;
        %-----------
        %sample beta
        %-----------
        Dbeta = inv(X'*X/sig + iV_beta);
        dbeta = X'*y/sig + iV_beta*mu_beta;
        Hbeta = chol(Dbeta);
        betas = Dbeta*dbeta + Hbeta'*randn(2,1);
        
        %-----------
        %sample sig
        %-----------
        resids = y - X*betas;
        sig = invgamrnd((nobs/2)+a, inv( ib + .5*resids'*resids),1,1);
        
        sig_stored(i,j) = sig;
        beta0_stored(i,j) = betas(1);
        beta1_stored(i,j) = betas(2);
    end;
end;

n1 = 5;
ll = 950;
n2 = 1000;
xc = [950:1:1000];


subplot(321)
plot(sig_stored(1:n1,1)); 
hold on;
plot(sig_stored(1:n1,2),':');
plot(sig_stored(1:n1,3),'--o');
set(gca,'fontsize',16);
set(gca,'Xtick',[]);
xlabel('Iteration');
ylabel('\sigma^2');
hold off;

subplot(322)
plot(xc,sig_stored(ll:n2,1)); 
hold on;
plot(xc,sig_stored(ll:n2,2),':');
plot(xc,sig_stored(ll:n2,3),'--o');
set(gca,'fontsize',16);
xlabel('Iteration');
ylabel('\sigma^2');
hold off;

subplot(323)
plot(beta0_stored(1:n1,1)); 
hold on;
plot(beta0_stored(1:n1,2),':');
plot(beta0_stored(1:n1,3),'--o');
set(gca,'fontsize',16);
set(gca,'Xtick',[]);
xlabel('Iteration');
ylabel('\beta_0');
hold off;

subplot(324)
plot(xc,beta0_stored(ll:n2,1)); 
hold on;
plot(xc,beta0_stored(ll:n2,2),':');
plot(xc,beta0_stored(ll:n2,3),'--o');
xlabel('Iteration');
set(gca,'fontsize',16);
ylabel('\beta_0');
hold off;

subplot(325)
plot(beta1_stored(1:n1,1)); 
hold on;
plot(beta1_stored(1:n1,2),':');
plot(beta1_stored(1:n1,3),'--o');
xlabel('Iteration');
set(gca,'fontsize',16);
set(gca,'Xtick',[]);
ylabel('\beta_1');
hold off;

subplot(326)
plot(xc,beta1_stored(ll:n2,1)); 
hold on;
plot(xc,beta1_stored(ll:n2,2),':');
plot(xc,beta1_stored(ll:n2,3),'--o');
xlabel('Iteration');
set(gca,'fontsize',16);
ylabel('\beta_1');
hold off;

print('Converge_graph','-deps');
