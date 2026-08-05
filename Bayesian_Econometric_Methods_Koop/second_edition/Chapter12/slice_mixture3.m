clear; clc;
%This program implements slice sampling on a three component mixture model.

%---
%SETUP THE PARAMETERS OF THE THREE COMPONENT MIXTURE
%---

rand('seed',sum(100*clock));

mu1 = -1;
mu2 = 0;
mu3 = 2;

sig1 = .1;
sig2 = .1;
sig3 = .2;

p1 = .4;
p2 = .2;
p3 = 1-p1-p2;

xgrid = linspace(-3,4,500);

dens1 = (1/sqrt(2*pi*sig1))*exp(-(1/(2*sig1))*(xgrid-mu1).^2);
dens2 = (1/sqrt(2*pi*sig2))*exp(-(1/(2*sig2))*(xgrid-mu2).^2);
dens3 = (1/sqrt(2*pi*sig3))*exp(-(1/(2*sig3))*(xgrid-mu3).^2);
density = p1*dens1 + p2*dens2+p3*dens3;

x = 2; %initial condition for slice sampler
fx = mix3ord(mu1,mu2,mu3,sig1,sig2,sig3,p1,p2,p3,x);
w = .5;

iter = 101000;
burn = 1000;
xdraws = zeros(iter-burn,1);

for i = 1:iter;
    
    %----------
    %Sample y|x
    %----------
    y = fx*rand(1,1);
    %-----------
    %Sample x|y 
    %----------
    
    LL = x-w*rand(1,1);
    UL = LL+w;
    
    lower_ordinate = mix3ord(mu1,mu2,mu3,sig1,sig2,sig3,p1,p2,p3,LL);
    upper_ordinate = mix3ord(mu1,mu2,mu3,sig1,sig2,sig3,p1,p2,p3,UL);
    
    %----------
    %Implement steps to widen interval until lower (LL) and upper (UL)
    %bounds are outide of the slice
    %----------
        while lower_ordinate>y;
            LL = LL-w;
            lower_ordinate = mix3ord(mu1,mu2,mu3,sig1,sig2,sig3,p1,p2,p3,LL);
       end;
    
        while upper_ordinate>y;
            UL = UL+w;
            upper_ordinate = mix3ord(mu1,mu2,mu3,sig1,sig2,sig3,p1,p2,p3,UL);
        end;
    
    %-----------------
    %Draw a candidate from the resulting interval
    %-----------------
    xcand = LL + (UL-LL)*rand(1,1);
    fxcand = mix3ord(mu1,mu2,mu3,sig1,sig2,sig3,p1,p2,p3,xcand);
    
    %--------------------
    %Reject candidate if outside of the slice and adaptively shrink
    %interval
    %--------------------
    while fxcand<y;
        if xcand < x
            LL = xcand;
        else
            UL = xcand;
        end;
        
         xcand = LL + (UL-LL)*rand(1,1);
         fxcand = mix3ord(mu1,mu2,mu3,sig1,sig2,sig3,p1,p2,p3,xcand);
    end;
    x=xcand;
    fx = fxcand;

    if i > burn;
        xdraws(i-burn,1) = x;
    end;

end;


[dom ran] = epanech2(xdraws);
plot(dom,ran);
hold on;
plot(xgrid,density,'r--');
legend({'Slice Sampling Density (Kernel Estimate)','True Mixture Density'},'fontsize',18);
hold off;

