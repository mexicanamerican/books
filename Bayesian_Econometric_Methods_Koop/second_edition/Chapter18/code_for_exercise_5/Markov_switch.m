%Code for Exercise 19.5: The Markov switching model
%two regimes,different variance in each, no covariates.
clear;
clc;
randn('seed',sum(100*clock));
rand('seed',sum(100*clock));

load usgdp.txt;
%realgdp = usgdp(:,1)*1000 + usgdp(:,2);
nobs_A = length(usgdp);
y_lag = usgdp(1:nobs_A-1);
yt = usgdp(2:nobs_A);
%The data is for GDP, make it into GDP growth
y = (log(yt) - log(y_lag))*100;
nobs = length(y);

%prior hyperparameters
a0 = 1/2;
a1 = 1/2;
b0 = 1/2;
b1 = 1/2;
v11 = 3;
v01 = 3;
v12 = 1;
v02 = 1;
m0 = 0;
V0 = 10;
invV0 = inv(V0);
m1 = 1;
V1 = 10;
invV1 = inv(V1);

%initialize stuff
iter = 50000;
burn = 1000;
    %starting values of algorithm
a = 1/2;
b = 1/2;
mu0 = 0;
mu1 = 2;
sig = .5;


a_final = zeros(iter-burn,1);
b_final = a_final;
mu1_final = a_final;
mu0_final = a_final;
sig_final = a_final;
sig0_final = a_final;
S_tY_t1 = zeros(nobs,2);
S_tY_t = zeros(nobs,2);
states_counter = zeros(nobs,1);

for i = 1:iter;
    
    
    %---------------------
    %sample latent states
    %---------------------
    t_t = zeros(nobs,1);
    t_tminus1 = zeros(nobs,1);
    muR1 = mu0 + mu1;
    
    state1_ords = normpdf(y,muR1,sqrt(sig));   %n \times 1 ordinates under Regime 1
    state0_ords = normpdf(y,mu0,sqrt(sig));    %n \times 1 ordinates under Regime 2
    
    P1 = a/(a+b);
    P2 = b/(a+b);
    
    %forward filter to get s_t | Y_t, theta mass functions
    for j = 1:nobs;
        if j==1
            t_tminus1(1) = P1;                                                      %notation to denote Pr(s_t=1 | Y_t-1,theta)
            t_t(1) =  P1*state1_ords(1)/(P1*state1_ords(1) + P2*state0_ords(1) );          
                                                                                    %notation to denote Pr(s_t=1 | Y_t, theta)
        else
            t_tminus1(j) = a*(1-t_t(j-1)) + (1-b)*t_t(j-1);
            t_t(j) = (t_tminus1(j)*state1_ords(j)) / (t_tminus1(j)*state1_ords(j) + (1 - t_tminus1(j))*state0_ords(j));
        end;
    end;
    St_t = [t_t (1-t_t)];
    St_tminus1 = [t_tminus1 (1 - t_tminus1)];
    
    %backward sample to sample latent states. 
    U = rand(nobs,1);
    states = zeros(nobs,1);
    sT = .5 + .5*sign(St_t(nobs,1) - U(1));     %draw s(T) from s_T | Y_t, theta
    
    states(nobs,1) = sT;
    
    for j=1:nobs-1;
        svec = states(nobs - j +1)*[(1-b) a] + (1-states(nobs-j+1))*[b (1-a)];
        prob1 = St_t(nobs-j,1)*svec(1) / (St_t(nobs-j,1)*svec(1) + St_t(nobs-j,2)*svec(2));
        states(nobs-j) = .5 + .5*sign(prob1 - U(j+1)); 
        
    end;
   
    %-------------------
    %Sample u_0, u_1
    %-------------------
    variances = ones(nobs,1)*sig;
    %transform the data 
    ytilde = y./(sqrt(variances));
    const_tilde = ones(nobs,1)./sqrt(variances);
    stilde = states./sqrt(variances);
    y1 = ytilde - mu1*stilde;
    
    %sample mu0
    Dm0 = inv(const_tilde'*const_tilde + invV0);
    dm0 = const_tilde'*y1 + invV0*m0;
    mu0 = Dm0*dm0 + sqrt(Dm0)*randn(1,1);
    
    
    %sample mu1
    y2 = ytilde - const_tilde*mu0;
    Dm1 = inv(stilde'*stilde + invV1);
    dm1 = stilde'*y2 + invV1*m1;
    mu_trunc = Dm1*dm1;
    var_trunc = Dm1;
    mu1 = truncnorm2(mu_trunc,var_trunc,0,999);
    
    %---------------------------
    %sample sigma_0^2, sigma_1^2
    %---------------------------
    resids = y - mu0 - mu1*states;
        sig = invgamrnd(v11 + (nobs/2),inv(inv(v12) + .5*sum((resids.^2) )),1,1);
   
   
   
    
    %-----------------
    %sample a,b 
    %-----------------
    
    %sample a
        %first, find all the instances where the states are zero, from
        %1-T-1:
        points0 = find(states(1:nobs-1)==0);
        points_0eval = points0+1;
        states0_tab = states(points_0eval);
        points_00 = find(states0_tab==0);
        points_01 = find(states0_tab==1);
        n00 = length(points_00);
        n01 = length(points_01);
        
        a = betarnd(a0+n01,a1+n00);
    
    %sample b
        points1 = find(states(1:nobs-1)==1);
        points_1eval = points1+1;
        states1_tab = states(points_1eval);
        points_11 = find(states1_tab==1);
        points_10 = find(states1_tab==0);
        n11 = length(points_11);
        n10 = length(points_10);
        
        b = betarnd(b0+n10,b1+n11);
    
        if i > burn;
            mu0_final(i-burn,1) = mu0;
            mu1_final(i-burn,1) = mu1;
            a_final(i-burn,1) = a;
            b_final(i-burn,1) = b;
            sig_final(i-burn,1) = sig;
            states_counter = states_counter + states;
        end;
    
end;

disp('Mean of recession persistence parameter,');
[mean(1-a_final) std(1-a_final)]
disp('Mean of expansion persistence parameter,  Std. Dev');
[mean(1-b_final) std(1-b_final)]

disp('Means of Variance parameters (left column=estimate,  right = stdev), regime0, regime1');
[mean(sig_final)  std(sig_final)]
disp('Means of Intercept parameters (left column=estimate, right = std dev), regime0, regime1');
[mean(mu0_final)  std(mu0_final); mean(mu1_final)  std(mu1_final)]

c3 = [.25 .5 .75]';
c4 = [0 .25 .5 .75]';
%-------------------
states_prob = states_counter/(iter-burn);
xgrid = [];
for j = 1947:2009;
    if j ==1947;
        addon = 1947*ones(3,1) + c3;
    else
        addon = j*ones(4,1) + c4;
    end
    xgrid = [xgrid;addon];
end;

plot(xgrid,1-states_prob);
xlabel('Time (Quarterly)');
ylabel('Probability')

    
