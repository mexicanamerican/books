% Based on the code in the URL, 1D nozzle solution, conservative form
% Anderson's CFD_the_basics_with_applications Chapter 7 
% https://github.com/abhiyanpaudel/quasi-1D-nozzle-flows/blob/main/quasi_1D_with_shock_capturing/quasi_1D_with_shock_capturing.m
clc
clear all

% --- Inputs & Setup (Anderson Chapter 7) ---
dx = 0.05;             % Grid size                   
c  = 0.5;              % Courant number                 
gamma = 1.4;           % Ratio of specific heats
x = 0:dx:3.0;          % Grid range
k = length(x);
Cx = 0.2;              % Artificial viscosity parameter

% Parabolic area distribution
A = 1 + 2.2.*(x - 1.5).^2;   

% --- Initial Conditions ---
rho = zeros(1, k);
T   = zeros(1, k);

for i = 1:k
    if x(i) >= 0 && x(i) < 0.5
        rho(i) = 1;                          
        T(i)   = 1;                          
    elseif x(i) >= 0.5 && x(i) < 1.5
        rho(i) = 1 - 0.366*(x(i) - 0.5);         
        T(i)   = 1 - 0.167*(x(i) - 0.5);           
    elseif x(i) >= 1.5 && x(i) < 2.1
        rho(i) = 0.634 - 0.702*(x(i) - 1.5);     
        T(i)   = 0.833 - 0.4908*(x(i) - 1.5);      
    else
        rho(i) = 0.5892 + 0.10228*(x(i) - 2.1);  
        T(i)   = 0.93968 + 0.0622*(x(i) - 2.1);    
    end
end

V = 0.59 ./ (rho.*A);                          
P = rho .* T;                                  

gamma1 = 1/gamma;
gamma2 = gamma - 1;

% Solution vectors
U1 = rho .* A;                                           
U2 = rho .* A .* V;                                        
U3 = rho .* A .* ((T./gamma2) + (0.5*gamma.*V.^2));          

% Flux vectors
F1 = U2;                                                    
F2 = (U2.^2 ./ U1) + (1 - gamma1).*(U3 - 0.5*gamma.*U2.^2 ./ U1);     
F3 = (gamma.*U2.*U3 ./ U1) - 0.5*gamma*gamma2.*U2.^3 ./ U1.^2;    

resmax = 1e-6;      
res = 1;
nstep = 0;          

% --- Main Time-Stepping Loop ---
while res > resmax
    dta = (c*dx) ./ (V + T.^0.5);
    dt = min(dta);
    nstep = nstep + 1;
     
    % Predictor Step
    for i = 2:k-1
        J2_p = gamma1*rho(i)*T(i)*((A(i+1) - A(i))/dx);
        dU1dt_p(i) = -(F1(i+1) - F1(i))/dx;
        dU2dt_p(i) = -(F2(i+1) - F2(i))/dx + J2_p;
        dU3dt_p(i) = -(F3(i+1) - F3(i))/dx;
    end
     
    for i = 2:k-1
        num = abs(P(i+1) - 2*P(i) + P(i-1));
        den = P(i+1) + 2*P(i) + P(i-1);
        S1_p(i) = (Cx*num*(U1(i+1) - 2*U1(i) + U1(i-1)))/den;
        S2_p(i) = (Cx*num*(U2(i+1) - 2*U2(i) + U2(i-1)))/den;
        S3_p(i) = (Cx*num*(U3(i+1) - 2*U3(i) + U3(i-1)))/den;
    end

    for i = 2:k-1
        U1_p(i) = U1(i) + (dU1dt_p(i)*dt) + S1_p(i);      
        U2_p(i) = U2(i) + (dU2dt_p(i)*dt) + S2_p(i);      
        U3_p(i) = U3(i) + (dU3dt_p(i)*dt) + S3_p(i);      
    end
     
    U1_p(1) = U1(1); U2_p(1) = U2(1); U3_p(1) = U3(1);
    U1_p(k) = U1(k); U2_p(k) = U2(k); U3_p(k) = U3(k);

    for i = 1:k
        rho_p(i)  = U1_p(i)/A(i);                               
        V_p(i)    = U2_p(i)/U1_p(i);                              
        T_p(i)    = gamma2*((U3_p(i)/U1_p(i)) - 0.5*gamma*V_p(i)^2);
        P_p(i)    = rho_p(i)*T_p(i);                              
        F1_p(i)   = U2_p(i);                                     
        F2_p(i)   = (U2_p(i)^2/U1_p(i)) + (1-gamma1)*(U3_p(i) - 0.5*gamma*U2_p(i)^2/U1_p(i));       
        F3_p(i)   = (gamma*U2_p(i)*U3_p(i)/U1_p(i)) - (0.5*gamma*gamma2*U2_p(i)^3/U1_p(i)^2);      
    end

    % Corrector Step
    for i = 2:k-1
        J2_c = gamma1*rho_p(i)*T_p(i)*((A(i) - A(i-1))/dx);
        dU1dt_c(i) = -(F1_p(i) - F1_p(i-1))/dx;
        dU2dt_c(i) = -(F2_p(i) - F2_p(i-1))/dx + J2_c;
        dU3dt_c(i) = -(F3_p(i) - F3_p(i-1))/dx;
    end
     
    for i = 2:k-1
        num = abs(P_p(i+1) - 2*P_p(i) + P_p(i-1));
        den = P_p(i+1) + 2*P_p(i) + P_p(i-1);
        S1_c(i) = (Cx*num*(U1_p(i+1) - 2*U1_p(i) + U1_p(i-1)))/den;
        S2_c(i) = (Cx*num*(U2_p(i+1) - 2*U2_p(i) + U2_p(i-1)))/den;
        S3_c(i) = (Cx*num*(U3_p(i+1) - 2*U3_p(i) + U3_p(i-1)))/den;
    end
     
    % Average Derivatives & Update Solution
    for i = 2:k-1
        dU1dt_av(i) = 0.5*(dU1dt_p(i) + dU1dt_c(i));
        dU2dt_av(i) = 0.5*(dU2dt_p(i) + dU2dt_c(i));
        dU3dt_av(i) = 0.5*(dU3dt_p(i) + dU3dt_c(i));
        
        U1(i) = U1(i) + (dU1dt_av(i)*dt) + S1_c(i);  
        U2(i) = U2(i) + (dU2dt_av(i)*dt) + S2_c(i);  
        U3(i) = U3(i) + (dU3dt_av(i)*dt) + S3_c(i);  
    end
      
    % Inlet Boundary Conditions
    U1(1) = A(1);                    
    U2(1) = 2*U2(2) - U2(3);           
    ve = U2(1)/U1(1);
    U3(1) = U1(1)*((1/gamma2) + 0.5*gamma*ve^2); 
     
    % Outlet Boundary Conditions
    U1(k) = 2*U1(k-1) - U1(k-2);           
    U2(k) = 2*U2(k-1) - U2(k-2);           
    U3(k) = (0.6784*A(k)/gamma2) + (0.5*gamma*U2(k)^2/U1(k));
    
    for i = 1:k
        F1(i) = U2(i);
        F2(i) = (U2(i)^2/U1(i)) + (1-gamma1)*(U3(i) - 0.5*gamma*U2(i)^2/U1(i));
        F3(i) = (gamma*U2(i)*U3(i)/U1(i)) - 0.5*gamma*gamma2*U2(i)^3/U1(i)^2;
    end
    
    % Update Primitive Variables
    for i = 1:k
        rho(i) = U1(i)/A(i);                            
        V(i)   = U2(i)/U1(i);                             
        T(i)   = gamma2*((U3(i)/U1(i)) - 0.5*gamma*V(i)^2); 
        M(i)   = V(i)*T(i)^-0.5;
        P(i)   = rho(i)*T(i);                             
    end

    % Convergence criterion
    res1 = abs(dU1dt_av(16));
    res2 = abs(dU2dt_av(16));
    res = max(res1, res2);
    
    if nstep == 2600, break; end
end

% --- Stream Direct Printout (Anderson Table 7.9 Format) ---
mf = rho .* A .* V; % Mass flow rate

fprintf('\n=== Numerical Results (Conservative Form) at Step %d ===\n\n', nstep);
fprintf('%-8s %-8s %-10s %-8s %-8s %-8s %-8s %-10s\n', ...
    'x/L', 'A/A*', 'rho/rho0', 'V/a0', 'T/T0', 'p/p0', 'Mach', 'Mass_Flow');
fprintf('-------------------------------------------------------------------------\n');

for i = 1:k
    fprintf('%-8.2f %-8.4f %-10.4f %-8.4f %-8.4f %-8.4f %-8.4f %-10.4f\n', ...
        x(i), A(i), rho(i), V(i), T(i), P(i), M(i), mf(i));
end
