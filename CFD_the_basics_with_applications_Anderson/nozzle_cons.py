import numpy as np

# --- Inputs & Setup (Anderson Chapter 7) ---
dx = 0.05             # Grid size                   
c = 0.5               # Courant number                 
gamma = 1.4           # Ratio of specific heats
x = np.arange(0, 3.0 + dx/2, dx)  # Grid range (0 to 3.0)
k = len(x)
Cx = 0.2              # Artificial viscosity parameter

# Parabolic area distribution
A = 1 + 2.2 * (x - 1.5)**2   

# --- Initial Conditions ---
rho = np.zeros(k)
T = np.zeros(k)

for i in range(k):
    if 0 <= x[i] < 0.5:
        rho[i] = 1.0                          
        T[i] = 1.0                          
    elif 0.5 <= x[i] < 1.5:
        rho[i] = 1.0 - 0.366 * (x[i] - 0.5)         
        T[i] = 1.0 - 0.167 * (x[i] - 0.5)           
    elif 1.5 <= x[i] < 2.1:
        rho[i] = 0.634 - 0.702 * (x[i] - 1.5)     
        T[i] = 0.833 - 0.4908 * (x[i] - 1.5)      
    else:
        rho[i] = 0.5892 + 0.10228 * (x[i] - 2.1)  
        T[i] = 0.93968 + 0.0622 * (x[i] - 2.1)    

V = 0.59 / (rho * A)                          
P = rho * T                                  

gamma1 = 1.0 / gamma
gamma2 = gamma - 1.0

# Solution vectors
U1 = rho * A                                           
U2 = rho * A * V                                        
U3 = rho * A * ((T / gamma2) + (0.5 * gamma * V**2))          

# Flux vectors
F1 = U2.copy()                                                    
F2 = (U2**2 / U1) + (1.0 - gamma1) * (U3 - 0.5 * gamma * U2**2 / U1)     
F3 = (gamma * U2 * U3 / U1) - 0.5 * gamma * gamma2 * U2**3 / U1**2    

resmax = 1e-6      
res = 1.0
nstep = 0          

# Allocate derivative and artificial viscosity buffers
dU1dt_p = np.zeros(k)
dU2dt_p = np.zeros(k)
dU3dt_p = np.zeros(k)
S1_p = np.zeros(k)
S2_p = np.zeros(k)
S3_p = np.zeros(k)

dU1dt_c = np.zeros(k)
dU2dt_c = np.zeros(k)
dU3dt_c = np.zeros(k)
S1_c = np.zeros(k)
S2_c = np.zeros(k)
S3_c = np.zeros(k)

dU1dt_av = np.zeros(k)
dU2dt_av = np.zeros(k)
dU3dt_av = np.zeros(k)

U1_p = np.zeros(k)
U2_p = np.zeros(k)
U3_p = np.zeros(k)
rho_p = np.zeros(k)
V_p = np.zeros(k)
T_p = np.zeros(k)
P_p = np.zeros(k)
F1_p = np.zeros(k)
F2_p = np.zeros(k)
F3_p = np.zeros(k)

M = np.zeros(k)

# --- Main Time-Stepping Loop ---
while res > resmax:
    dta = (c * dx) / (V + np.sqrt(T))
    dt = np.min(dta)
    nstep += 1
     
    # Predictor Step
    for i in range(1, k - 1):
        J2_p = gamma1 * rho[i] * T[i] * ((A[i+1] - A[i]) / dx)
        dU1dt_p[i] = -(F1[i+1] - F1[i]) / dx
        dU2dt_p[i] = -(F2[i+1] - F2[i]) / dx + J2_p
        dU3dt_p[i] = -(F3[i+1] - F3[i]) / dx
     
    for i in range(1, k - 1):
        num = abs(P[i+1] - 2 * P[i] + P[i-1])
        den = P[i+1] + 2 * P[i] + P[i-1]
        S1_p[i] = (Cx * num * (U1[i+1] - 2 * U1[i] + U1[i-1])) / den
        S2_p[i] = (Cx * num * (U2[i+1] - 2 * U2[i] + U2[i-1])) / den
        S3_p[i] = (Cx * num * (U3[i+1] - 2 * U3[i] + U3[i-1])) / den

    for i in range(1, k - 1):
        U1_p[i] = U1[i] + (dU1dt_p[i] * dt) + S1_p[i]      
        U2_p[i] = U2[i] + (dU2dt_p[i] * dt) + S2_p[i]      
        U3_p[i] = U3[i] + (dU3dt_p[i] * dt) + S3_p[i]      
     
    U1_p[0], U2_p[0], U3_p[0] = U1[0], U2[0], U3[0]
    U1_p[-1], U2_p[-1], U3_p[-1] = U1[-1], U2[-1], U3[-1]

    for i in range(k):
        rho_p[i] = U1_p[i] / A[i]                               
        V_p[i] = U2_p[i] / U1_p[i]                              
        T_p[i] = gamma2 * ((U3_p[i] / U1_p[i]) - 0.5 * gamma * V_p[i]**2)
        P_p[i] = rho_p[i] * T_p[i]                              
        F1_p[i] = U2_p[i]                                     
        F2_p[i] = (U2_p[i]**2 / U1_p[i]) + (1.0 - gamma1) * (U3_p[i] - 0.5 * gamma * U2_p[i]**2 / U1_p[i])       
        F3_p[i] = (gamma * U2_p[i] * U3_p[i] / U1_p[i]) - (0.5 * gamma * gamma2 * U2_p[i]**3 / U1_p[i]**2)      

    # Corrector Step
    for i in range(1, k - 1):
        J2_c = gamma1 * rho_p[i] * T_p[i] * ((A[i] - A[i-1]) / dx)
        dU1dt_c[i] = -(F1_p[i] - F1_p[i-1]) / dx
        dU2dt_c[i] = -(F2_p[i] - F2_p[i-1]) / dx + J2_c
        dU3dt_c[i] = -(F3_p[i] - F3_p[i-1]) / dx
     
    for i in range(1, k - 1):
        num = abs(P_p[i+1] - 2 * P_p[i] + P_p[i-1])
        den = P_p[i+1] + 2 * P_p[i] + P_p[i-1]
        S1_c[i] = (Cx * num * (U1_p[i+1] - 2 * U1_p[i] + U1_p[i-1])) / den
        S2_c[i] = (Cx * num * (U2_p[i+1] - 2 * U2_p[i] + U2_p[i-1])) / den
        S3_c[i] = (Cx * num * (U3_p[i+1] - 2 * U3_p[i] + U3_p[i-1])) / den
     
    # Average Derivatives & Update Solution
    for i in range(1, k - 1):
        dU1dt_av[i] = 0.5 * (dU1dt_p[i] + dU1dt_c[i])
        dU2dt_av[i] = 0.5 * (dU2dt_p[i] + dU2dt_c[i])
        dU3dt_av[i] = 0.5 * (dU3dt_p[i] + dU3dt_c[i])
        
        U1[i] += (dU1dt_av[i] * dt) + S1_c[i]  
        U2[i] += (dU2dt_av[i] * dt) + S2_c[i]  
        U3[i] += (dU3dt_av[i] * dt) + S3_c[i]  
      
    # Inlet Boundary Conditions
    U1[0] = A[0]                    
    U2[0] = 2 * U2[1] - U2[2]           
    ve = U2[0] / U1[0]
    U3[0] = U1[0] * ((1.0 / gamma2) + 0.5 * gamma * ve**2) 
     
    # Outlet Boundary Conditions
    U1[-1] = 2 * U1[-2] - U1[-3]           
    U2[-1] = 2 * U2[-2] - U2[-3]           
    U3[-1] = (0.6784 * A[-1] / gamma2) + (0.5 * gamma * U2[-1]**2 / U1[-1])
    
    for i in range(k):
        F1[i] = U2[i]
        F2[i] = (U2[i]**2 / U1[i]) + (1.0 - gamma1) * (U3[i] - 0.5 * gamma * U2[i]**2 / U1[i])
        F3[i] = (gamma * U2[i] * U3[i] / U1[i]) - 0.5 * gamma * gamma2 * U2[i]**3 / U1[i]**2
    
    # Update Primitive Variables
    for i in range(k):
        rho[i] = U1[i] / A[i]                            
        V[i] = U2[i] / U1[i]                             
        T[i] = gamma2 * ((U3[i] / U1[i]) - 0.5 * gamma * V[i]**2) 
        M[i] = V[i] * (T[i]**-0.5)
        P[i] = rho[i] * T[i]                             

    # Convergence criterion (checking grid index 15, corresponding to node 16 in MATLAB)
    res1 = abs(dU1dt_av[15])
    res2 = abs(dU2dt_av[15])
    res = max(res1, res2)
    
    if nstep == 2600:
        break

# --- Stream Direct Printout (Anderson Table 7.9 Format) ---
mf = rho * A * V

print(f"\n=== Numerical Results (Conservative Form) at Step {nstep} ===\n")
print(f"{'x/L':<8} {'A/A*':<8} {'rho/rho0':<10} {'V/a0':<8} {'T/T0':<8} {'p/p0':<8} {'Mach':<8} {'Mass_Flow':<10}")
print("-" * 73)

for i in range(k):
    print(f"{x[i]:<8.2f} {A[i]:<8.4f} {rho[i]:<10.4f} {V[i]:<8.4f} {T[i]:<8.4f} {P[i]:<8.4f} {M[i]:<8.4f} {mf[i]:<10.4f}")
