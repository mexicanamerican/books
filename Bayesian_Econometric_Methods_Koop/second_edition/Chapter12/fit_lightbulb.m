%----------------------------------------
%this code fits the light bulb data model
%----------------------------------------
clear;
clc;
randn('seed',sum(100*clock));


%-------------------
%Load and Clean Data
%--------------------
load lightbulbdata;
y = lightbulbdata(:,1);
lny = log(y);
type = lightbulbdata(:,2);
nobs = length(y);
INCAND = (ones(nobs,1)-type);
n1 = sum(type);
n0 = nobs-n1;

%-------------------------
%Set Prior Hyperparameters
%-------------------------
alpha1 = .2;
alpha0 = .2;
a1=3;
a0=3;
b1=(1/(2));
b1inv = inv(b1);
b0 = (1/(2));
b0inv = inv(b0);

parm1 = (n1/2)+a1;
parm0 = (n0/2)+a0;

%--------------------------------------------
%Set Number of iterations, initial conditions
%--------------------------------------------
Iter = 51000;
Burn = 1000;

theta0 = 1;
theta1 = 1;

%-----------------------
%Define Storeage Objects
%-----------------------
thetas_keep = zeros(Iter-Burn,2);
sigmas_keep =zeros(Iter-Burn,2);
cost_keep = zeros(Iter-Burn,2);
%--------------------
%Begin Loop, i=1:Iter
%--------------------

for i = 1:Iter;
    %--------------------------------
    %Sample Conditional 1 (sigma_1^2)
    %--------------------------------
    resids_1 = type.*(lny -theta0-theta1);
    sig1 = invgamrnd(parm1,inv(b1inv +.5*sum(resids_1.^2)),1,1);
    
    
    %--------------------------------
    %Sample Conditional 2 (sigma_0^2)
    %--------------------------------
    resids_0 = INCAND.*(lny -theta0);
    sig0 = invgamrnd(parm0,inv(b0inv +.5*sum(resids_0.^2)),1,1);
    
    %--------------------------------
    %Sample Conditional 3 (theta_1)
    %--------------------------------
    S1 = sum(type.*(lny-theta0));
    mean1 = (S1-alpha1*sig1)/n1;
    var1 = sig1/n1;
    theta1 = mean1 + sqrt(var1)*randn(1,1);
    
    %--------------------------------
    %Sample Conditional 4 (theta_0)
    %--------------------------------
    variances = type*sig1 + INCAND*sig0;
    S0 = sum( (lny-type*theta1)./variances );
    A0 = (n1*sig0+n0*sig1)/(sig1*sig0);
    invA0 = inv(A0);
    var0 = invA0;
    mean0 = invA0*(S0-alpha0);
    theta0 = mean0 + sqrt(var0)*randn(1,1);
  
    %------------------------------
    %If i > burn, store simulations
    %------------------------------
    if i > Burn;
        thetas_keep(i-Burn,:) = [theta0 theta1];
        sigmas_keep(i-Burn,:) = [sig0 sig1];
       
        
        %-----------------------------------------------------
        %Insert Extra Steps to calculate number of bulbs, cost
        %-----------------------------------------------------
        LED_hours = 0;
        nLED=0;
        INCAND_hours = 0;
        nINCAND=0;
        while LED_hours<25;
            hours_temp = exp( (theta0+theta1)+sqrt(sig1)*randn(1,1));
            LED_hours = LED_hours + hours_temp;
            nLED = nLED+1;
        end;
        while INCAND_hours<25;
            hours_temp = exp( (theta0)+sqrt(sig0)*randn(1,1));
            INCAND_hours = INCAND_hours + hours_temp;
            nINCAND = nINCAND+1;
        end;
        cost_LED = 5*nLED +1.05*25;
        cost_INCAND= nINCAND + 9*25;
    cost_keep(i-Burn,:) = [cost_INCAND cost_LED];
    end;

    
end;

%--------------------------
%Calculate Posterior Means
%---------------------------
mu_theta = mean(thetas_keep);
mu_sigma = mean(sigmas_keep);


%----------------------------------------------------------
%Calculate mean and variance of each lognormal distribution
%----------------------------------------------------------
theta_LED = sum(mu_theta);
theta_INCAND = mu_theta(1);
sigma_LED = mu_sigma(2);
sigma_INCAND = mu_sigma(1);

disp('Means of INCAND and LED lognormals');
[exp(theta_INCAND + sigma_INCAND/2) exp(theta_LED + sigma_LED/2)]

disp('Variances of INCAND and LED Lognormals');
[(exp(sigma_INCAND)-1)*(exp(2*theta_INCAND + sigma_INCAND)) (exp(sigma_LED)-1)*(exp(2*theta_LED + sigma_LED))]
