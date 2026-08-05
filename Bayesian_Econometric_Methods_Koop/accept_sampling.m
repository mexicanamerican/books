%This m-file performs acceptance sampling 
%for the triangular density, using a uniform 
%source density;
clear;
clc;
rand('seed',sum(100*clock));
randn('seed',sum(100*clock));
%----------------------------
%sample from triangular using U(-1,1) source
%-----------------------------
M = 2;
nkeep = 25000;
total_counter = 0;
counter = 0;
triang_keep = zeros(nkeep,1);
    while counter < nkeep;
        total_counter = total_counter +1;
        U = rand(1,1);
        candidate = -1 + 2*rand(1,1);   %draw from the source density
        true_kernel = 1 - abs(candidate);
        source_kernel = (1/2);
        if U <= true_kernel/(M*source_kernel)
            counter = counter+1;
            triang_keep(counter,1) = candidate;
        end;
    end;
        
disp('Observed fraction accepted, uniform source');
counter/total_counter

%------------------------------------------
%sample from triangular using N(0,1) source
%------------------------------------------
%sig = 1;        %change this line of code to examine how performance changes 
                %as sigma^2 changes. 
%M=1;            %
sig = 1/6;
max_point = (1 + sqrt(1/3))/2;
triang_ord = 1 - abs(max_point);
norm_ord  = exp( - (1/(2*sig))*max_point^2);
M = triang_ord/norm_ord;
theoretical_accept = 1/(M*sqrt(2*pi*sig))

nkeep = 25000;
counter = 0;
triang_keep2 = zeros(nkeep,1);
total_counter = 0;
    while counter < nkeep;
        total_counter = total_counter + 1;
        U = rand(1,1);
        candidate = sqrt(sig)*randn(1,1);   %draw from the source density
        true_kernel = max([(1 - abs(candidate)) 0]);
        source_kernel = exp(- (1/(2*sig))*candidate^2);
        if U <= true_kernel/(M*source_kernel)
            counter = counter+1;
            triang_keep2(counter,1) = candidate;
        end;
    end;
disp('Observed Fraction of draws accepted wth Normal Source Density');
counter/total_counter 


%You can "uncomment" the following 5 lines to plot the estimated 
%densities from the uniform and normal source densities
%[dom ran] = epanech2(triang_keep);    
%[dom2 ran2] = epanech2(triang_keep2);

%plot(dom,ran);
%hold on;
%plot(dom2,ran2,'r:');

%Plot the ``blanketing function'' for the Normal Source with sigma^2 = 1/6.
sig = 1/6;
xgrid = linspace(-1,1,25)';
topdens = 1 - abs(xgrid);
bottomdens = M*exp(- (1/(2*sig))*xgrid.^2);
plot(xgrid,topdens);
hold on;
plot(xgrid,bottomdens,'r:');
bottomdens2 = exp(- (1/2)*xgrid.^2);
bottomdens3 = ones(25,1);
plot(xgrid,bottomdens2,'k-o');
plot(xgrid,bottomdens3,'g-d');

xlabel('X');
ylabel('Density');
gtext('N(0,1/6)');
gtext('N(0,1)');
gtext('U(-1,1)');
gtext('Triangular Density');
hold off;