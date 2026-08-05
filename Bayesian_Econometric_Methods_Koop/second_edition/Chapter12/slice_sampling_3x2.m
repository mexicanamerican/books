%This function file illustrates slice sampling to generate draws from the
%density f(x) = 3x^2, 0 < x < 1. 

x = .3;
fx = 3*(x^2);
iter = 11000;
burn = 1000;

draws = zeros(iter-burn,1);
x_inversion = draws;

for i = 1:iter;
    u = fx*rand(1,1);
    lowbd = sqrt(u/3);
    x = lowbd + (1-lowbd)*rand(1,1);
    fx = 3*(x^2);
    inversion_draw = rand(1,1)^(1/3);
    
    if i > burn;
        draws(i-burn,1) = x;
        x_inversion(i-burn,1)= inversion_draw;
    end;
end;

subplot(211);
hist(draws,30);
xlabel('Histogram of Slice Sampling Draws','Fontsize',14);
subplot(212);
hist(x_inversion,30);
xlabel('Histogram of Draws via Inversion','Fontsize',14);



    