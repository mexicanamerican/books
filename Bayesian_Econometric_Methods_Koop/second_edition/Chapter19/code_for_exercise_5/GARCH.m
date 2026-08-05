clear; clc;
load nyse.txt;
y = 100*nyse(:,2);
T = size(y,1);
tid = linspace(1952,1995+11/12,T)';
nsim = 20000;
burnin = 1000;

    %% prior
lpri_gam = @(x) 0;
lpri_mu = @(x) 0;

    %% initialize for storeage
store_theta = zeros(nsim,4); % mu, gam
store_sig2 = zeros(nsim,T); 
store_llike = zeros(nsim,1);
    %% initialize the Markov chain
mu = mean(y);
e = y-mu;
e2 = e.^2;
Z = [ones(T-1,1) e2(1:end-1)];
tempb = log((Z'*Z)\(Z'*e2(2:end)));
sig20 = var(y);
gamt = fminsearch(@(x)-loglike_garch(e,x,sig20)-lpri_gam(x),[tempb; tempb(2)]);
expgamt = exp(gamt);
sig2 = zeros(T,1);
dsig2 = zeros(T,3);
dsig2(1,1) = expgamt(1);
dsig2(1,3) = expgamt(3)*sig20;
sig2(1) = expgamt(1) + expgamt(3)*sig20;
for t=2:T
    sig2(t) = expgamt(1) + expgamt(2)*e2(t-1) + expgamt(3)*sig2(t-1);
    dsig2(t,1) = expgamt(1) + expgamt(3)*dsig2(t-1,1);
    dsig2(t,2) = expgamt(2)*e2(t-1) + expgamt(3)*dsig2(t-1,2);
    dsig2(t,3) = expgamt(3)*(sig2(t-1) + dsig2(t-1,3));
end
S = repmat(.5./sig2.*(e2./sig2-1),1,3) .* dsig2;
I = S'*S;    
lprop = @(x) -.5*(x-gamt)'*I*(x-gamt);
Cgam = chol(I,'lower');
gam = gamt + Cgam'\randn(3,1);    
while sum(exp(gam(2:3))) > .999 
    gam = gamt + Cgam'\randn(3,1);    
end
ybar = mean(y);
s2 = var(y)/T;
countgam = 0;
countmu = 0;

disp('Starting GARCH.... ');
disp(' ' );
randn('seed',sum(clock*97)); rand('seed',sum(clock*37));

start_time = clock;
for loop = 1:nsim + burnin
        %% sample mu
    muc = ybar + sqrt(s2)*randn;
    [llikec,sig2c] = loglike_garch(y-muc,gam,sig20);
    [llike,sig2] = loglike_garch(y-mu,gam,sig20);
    alpMH = llikec + lpri_mu(muc) + .5*(muc-ybar)^2/s2 ...
        - (llike + lpri_mu(mu) + .5*(mu-ybar)^2/s2);
    if alpMH > log(rand)
        mu = muc;
        countmu = countmu + 1;        
    end
    
        %% sample gam      
    e = y-mu;
    [llike,sig2] = loglike_garch(e,gam,sig20);    
    if loop == 1 || rand>.5  %% maximize with prob 0.5
        e2 = e.^2;        
        gamt = fminsearch(@(x)-loglike_garch(e,x,sig20)-lpri_gam(x),gamt);        
        expgamt = exp(gamt);
        sig2 = zeros(T,1);
        dsig2 = zeros(T,3);
        dsig2(1,1) = expgamt(1);
        dsig2(1,3) = expgamt(3)*sig20;
        sig2(1) = expgamt(1) + expgamt(3)*sig20;
        for t=2:T
            sig2(t) = expgamt(1) + expgamt(2)*e2(t-1) + expgamt(3)*sig2(t-1);
            dsig2(t,1) = expgamt(1) + expgamt(3)*dsig2(t-1,1);
            dsig2(t,2) = expgamt(2)*e2(t-1) + expgamt(3)*dsig2(t-1,2);
            dsig2(t,3) = expgamt(3)*(sig2(t-1) + dsig2(t-1,3));
        end
        S = repmat(.5./sig2.*(e2./sig2-1),1,3) .* dsig2;
        I = S'*S;        
        df = 10;
        lprop = @(x) -(df+3)/2*log(1+(x-gamt)'*I*(x-gamt)/df);
        Cgam = chol(I,'lower');
    end        
    gamc = gamt + (Cgam'\randn(3,1))/sqrt(gamrnd(df/2,2/df));
    if sum(exp(gamc(2:3))) < .99 %% impose stationarity
        [llikec,sig2c] = loglike_garch(e,gamc,sig20);
        alpMH = llikec + lpri_gam(gamc) - lprop(gamc)...
            - (llike + lpri_gam(gam) - lprop(gam));
        if alpMH > log(rand)
            gam = gamc;
            sig2 = sig2c;
            llike = llikec;
            countgam = countgam + 1;        
        end
    end
    if loop>burnin
        isave = loop-burnin;        
        store_theta(isave,:) = [mu exp(gam)'];
        store_sig2(isave,:) = sig2';
        store_llike(isave) = llike;        
    end    
    if ( mod( loop, 5000 ) ==0 )
        disp(  [ num2str( loop ) ' loops... ' ] )
    end    
end
disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
disp(' ' );

thetahat = mean(store_theta)';
sighat = mean(sqrt(store_sig2))';  % plot std. dev.
thetastd = std(store_theta)';
accept = [countmu/(nsim+burnin) countgam/(nsim+burnin)]

figure;    
plot(tid, sighat, 'LineWidth',1,'Color','blue'); box off;
title('\sigma_t');

fprintf('\n'); 
fprintf('Parameter   | Posterior mean (Posterior std. dev.):\n'); 
fprintf('mu          | %.2f (%.2f)\n', thetahat(1), thetastd(1)); 
fprintf('alpha_0     | %.2f (%.2f)\n', thetahat(2), thetastd(2)); 
fprintf('alpha_1     | %.2f (%.2f)\n', thetahat(3), thetastd(3)); 
fprintf('beta_1      | %.2f (%.2f)\n', thetahat(4), thetastd(4)); 

