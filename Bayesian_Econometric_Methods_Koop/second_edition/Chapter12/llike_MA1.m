function lden = llike_MA1(psi,e,h)
T = length(e);
Hpsi = speye(T) + sparse(2:T,1:(T-1),psi*ones(1,T-1),T,T); 
u = Hpsi\e;
lden = -T/2*log(2*pi/h) -.5*(u'*u)*h;
end