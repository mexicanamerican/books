% sample_SVM_h.m
function [h flag] = sample_SVM_h(y,alp,mu,h,h0,sigh2,accept)
if nargin == 6
    accept = 0;
end
T = size(h,1);
flag = 0;
H = speye(T) - sparse(2:T,1:(T-1),ones(1,T-1),T,T);
HH = H'*H;
s2 = (y-mu).^2;
e_h = 1; ht = h;
while e_h> 10^(-4);
    exp_ht = exp(ht);
    tmp1 = .5*alp^2*exp_ht;
    tmp2 = .5*s2./exp_ht;
    f = -.5 - tmp1 + tmp2;
    G = sparse(1:T,1:T,tmp1+tmp2);
    S = -1/sigh2*HH*(ht - h0) + f;
    Kh = HH/sigh2 + G;
    new_ht = ht + Kh\S;
    e_h = max(abs(new_ht-ht));
    ht = new_ht;
end 
h_hat = ht;
exp_ht = exp(h_hat);
G = sparse(1:T,1:T,.5*alp^2*exp_ht+.5*s2./exp_ht);
Kh = HH/sigh2 + G; 
lph = @(x) -.5*(x-h0)'*HH*(x-h0)/sigh2 ...
    -.5*sum(x) -.5*exp(-x)'*(y-mu-alp*exp(x)).^2;
lg = @(x) -.5*(x-h_hat)'*Kh*(x-h_hat);
hc = h_hat + chol(Kh,'lower')'\randn(T,1); 
alp_MH = lph(hc) - lph(h) + lg(h) - lg(hc);    
if exp(alp_MH) > rand || accept == 1
    h = hc;
    flag = 1;
end
end
