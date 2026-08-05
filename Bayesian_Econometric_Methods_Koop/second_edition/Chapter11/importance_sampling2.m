%MATLAB code for Importance sampling #2 question

clear;
clc;


true_theta = 2;  %note that built-in matlab mfile ``exprnd'' that we will use hereparameterizes the density such that the mean is theta^{-1}, and thus the parameterization in the question is reproduced by using the inverse. 
itrue_theta = inv(true_theta);
iter = 1000;
%----------
%PART(b)
%-----------
IS_grid = [.5 2 4]'; %again, these are inverse of what is stated in the problem set
iIS_grid = [2 .5 .25]';
%IS_forquestion =[.5 2 4];
means_IS = zeros(iter,length(IS_grid));

for k = 1:length(IS_grid);
for j = 1:iter;
    importance_samples = exprnd(IS_grid(k),1000,1);
    true_dens = itrue_theta*exp(-itrue_theta*importance_samples);
    IS_dens = iIS_grid(k,1)*exp(-iIS_grid(k,1)*importance_samples);
    means_IS(j,k) = mean(importance_samples.*true_dens./IS_dens);
end;
Titles = ['Displaying results for alpha=', num2str(IS_grid(k))];
disp(Titles);
means = mean(means_IS(:,k));
stdevs = std(means_IS(:,k));
fprintf('Mean %s, Std Dev %d \n',means,stdevs)
disp(' ');
disp(' ');
end;
