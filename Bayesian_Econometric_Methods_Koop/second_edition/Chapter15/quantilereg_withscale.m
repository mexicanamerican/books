%This program performs quantile regression on the acs wage / education data
clear;
clc;
randn('seed',sum(100*clock));
rand('seed',sum(100*clock));

load acsdata.raw;
y = acsdata(:,1);
ed  =acsdata(:,2);
nobs = length(ed);
X = [ones(nobs,1) ed];
k = size(X,2);

randn('seed',sum(100*clock));
rand('seed',sum(100*clock));

iter = 25000;
burn = 1000;
betas_keep = zeros(iter-burn,k);
z = zeros(nobs,1);


p = .25;
tau = 2/(p*(1-p)); %tau is tau^2 in the text
theta = (1-(2*p))/(p*(1-p));
gigparm = 2+( (theta^2)/tau);

a = 2+((theta^2)/(tau));
a=a*ones(nobs,1);

betas = 2*ones(k,1);
sig=1;

Vbeta = 100000*eye(k);
invVbeta = inv(Vbeta);
mubeta = zeros(k,1);
asig = 3;
bsig =(1/2);
ibsig = 1/bsig;

for i = 1:iter;
    
    %-----------------------
    %sample the latent data (called z; this is nu in the text)
    %-----------------------
        ts = tau*sig;
        resids2 = ((y - X*betas).^2)/(ts);
        a=((2/sig) + ((theta^2)/(ts)))*ones(nobs,1);
    
        tempdraws = invgaussrnd(resids2,a);
        z = (1./tempdraws')';
    
    %--------------------------- 
    %sample the scale parameter
    %--------------------------
        resids3 = ((y-X*betas-theta*z).^2)./(2*tau*z);
        sig = invgamrnd(asig+(1.5*nobs),inv(ibsig + sum(z) + sum(resids3)),1,1);
    
    %-----------
    %sample beta
    %------------
        ts = tau*sig;
        Xtilde = bsxfun(@rdivide,X,z);
        ytilde= (y-theta*z)./z;
        XX = Xtilde'*X;
        XY = X'*ytilde;
        %Dbeta = inv( (1/ts)*XX + invVbeta);
        %dbeta = (1/ts)*XY  + invVbeta*mubeta;
        Dbeta = ts*(eye(k)/XX);
        dbeta = (1/ts)*XY;
        Hbeta = chol(Dbeta);
        betas = Dbeta*dbeta + Hbeta'*randn(k,1);
       
        if i > burn;
            betas_keep(i-burn,:) = betas';
        end;
end;

disp('Done with sampling')
bb = mean(betas_keep);
stdevs = std(betas_keep);

disp('Posterior Mean')
bb


        
            
        