function draws = invgaussrnd(a,b);
%This program generates a sample from a particular GIG distribution,
%specifically, one of the form
%GIG(-1/2,a,b) which has density proportional to
%p(x) propto x^(-1/2) exp(-(1/2)[ax + bx^(-1)])
%
%the syntax is gigdraw(a,b);
%the approach is to first draw from an inverse gaussian distribution with
%kernel
%p(y) propto x^(-3/2) exp( - lambda(x-mu)^2/(2 \mu^2 x) );
%
%We do this using the algorithm provided by Michael, Schucany and Haas
%(1976 Statistical Computing). The reciprocal of this draw gives a draw
%from the desired GIG distribution. 


n=length(a);
%draws = zeros(n,1);

lambda = b;
mu2 = lambda./a;
mu = sqrt(mu2);

thresh =1;
i=0;
while thresh~=0;
    i=i+1;
    if i>1
        disp('something went wrong, sampling again');
    end;
counter =0;

v0 = (randn(n,1).^2);
x1 = mu + (mu2.*v0)./(2*lambda) - (mu./(2*lambda)).*sqrt(4*mu.*lambda.*v0 + mu2.*(v0.^2));
xtemp = (1.0e-3)*ones(1,n);   %replace x1 with very small number when =0.
x1 = max([x1' ; xtemp])';

%disp('problems with x1');
%min(x1)
%max(x1)
%sum(isnan(x1))
%sum(isinf(x1))

x2 = mu2./x1;
%disp('problems with x2');
%min(x2)
%max(x2)
%sum(isnan(x2))
%sum(isinf(x2))

p = mu./(mu+x1);
u = rand(n,1);

D = u<p;

draws = x2 + D.*(x1-x2);

counter = sum(isnan(draws));
counter = counter + sum(isinf(draws));
counter = counter + sum(draws<=0);
thresh = counter*thresh;
end;
%points1 = find(u<p);
%points2 = find(u>=p);


%draws(points1)=x1(points1);
%draws(points2)=x2(points2);



