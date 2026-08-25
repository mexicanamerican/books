import time
import numpy as np
import matplotlib.pyplot as plt

def quasi_1d_nozzle_flow_non_conservative(x, dx, A, C, nt, gamma):
    nx = len(x)
    rho1 = 1 - 0.314 * x
    T1 = 1 - 0.2314 * x
    V1 = (0.1 + 1.09 * x) * (T1**0.5)

    m1 = np.zeros((nt, nx))
    throat = np.argmin(np.abs(A - 1.0))

    rho_throat1 = np.zeros(nt)
    T_throat1 = np.zeros(nt)
    V_throat1 = np.zeros(nt)
    P_throat1 = np.zeros(nt)
    M_throat1 = np.zeros(nt)
    residual1 = np.zeros((nt, 3))

    drho_dt_p = np.zeros(nx)
    dV_dt_p = np.zeros(nx)
    dT_dt_p = np.zeros(nx)

    drho_dt_c = np.zeros(nx)
    dV_dt_c = np.zeros(nx)
    dT_dt_c = np.zeros(nx)

    for k in range(nt):
        rho_old = rho1.copy()
        T_old = T1.copy()
        V_old = V1.copy()

        dt = np.min(C * dx / (T1**0.5 + V1))

        # Predictor Step
        for i in range(1, nx - 1):
            dVdx = (V1[i+1] - V1[i]) / dx
            dlogAdx = (np.log(A[i+1]) - np.log(A[i])) / dx
            drhodx = (rho1[i+1] - rho1[i]) / dx
            dTdx = (T1[i+1] - T1[i]) / dx

            drho_dt_p[i] = -rho1[i]*dVdx - rho1[i]*V1[i]*dlogAdx - V1[i]*drhodx
            dV_dt_p[i] = -V1[i]*dVdx - (1/gamma)*(dTdx + (T1[i]/rho1[i])*drhodx)
            dT_dt_p[i] = -V1[i]*dTdx - (gamma-1)*T1[i]*(dVdx + V1[i]*dlogAdx)

            rho1[i] = rho1[i] + drho_dt_p[i]*dt
            V1[i] = V1[i] + dV_dt_p[i]*dt
            T1[i] = T1[i] + dT_dt_p[i]*dt

        # Corrector Step
        for i in range(1, nx - 1):
            dVdx = (V1[i] - V1[i-1]) / dx
            dlogAdx = (np.log(A[i]) - np.log(A[i-1])) / dx
            drhodx = (rho1[i] - rho1[i-1]) / dx
            dTdx = (T1[i] - T1[i-1]) / dx

            drho_dt_c[i] = -rho1[i]*dVdx - rho1[i]*V1[i]*dlogAdx - V1[i]*drhodx
            dV_dt_c[i] = -V1[i]*dVdx - (1/gamma)*(dTdx + (T1[i]/rho1[i])*drhodx)
            dT_dt_c[i] = -V1[i]*dTdx - (gamma-1)*T1[i]*(dVdx + V1[i]*dlogAdx)

        drho_dt = 0.5 * (drho_dt_p + drho_dt_c)
        dV_dt = 0.5 * (dV_dt_p + dV_dt_c)
        dT_dt = 0.5 * (dT_dt_p + dT_dt_c)

        residual1[k, 0] = np.max(np.abs(drho_dt))
        residual1[k, 1] = np.max(np.abs(dV_dt))
        residual1[k, 2] = np.max(np.abs(dT_dt))

        # Final Update
        for j in range(1, nx - 1):
            rho1[j] = rho_old[j] + drho_dt[j]*dt
            V1[j] = V_old[j] + dV_dt[j]*dt
            T1[j] = T_old[j] + dT_dt[j]*dt

        # Boundary Conditions
        V1[0] = 2 * V1[1] - V1[2]
        V1[-1] = 2 * V1[-2] - V1[-3]
        rho1[-1] = 2 * rho1[-2] - rho1[-3]
        T1[-1] = 2 * T1[-2] - T1[-3]

        m1[k, :] = rho1 * A * V1
        P1 = rho1 * T1
        M1 = V1 / (T1**0.5)

        rho_throat1[k] = rho1[throat]
        M_throat1[k] = M1[throat]
        P_throat1[k] = P1[throat]
        T_throat1[k] = T1[throat]
        V_throat1[k] = V1[throat]

    return rho1, T1, V1, P1, M1, m1, rho_throat1, V_throat1, T_throat1, P_throat1, M_throat1, residual1


def quasi_1d_nozzle_flow_conservative(x, dx, A, C, nt, gamma):
    nx = len(x)
    rho_at_t_0 = np.zeros(nx)
    T_at_t_0 = np.zeros(nx)
    V_at_t_0 = np.zeros(nx)

    for i in range(nx):
        if 0 <= x[i] < 0.5:
            rho_at_t_0[i] = 1.0
            T_at_t_0[i] = 1.0
        elif 0.5 <= x[i] < 1.5:
            rho_at_t_0[i] = 1.0 - 0.366 * (x[i] - 0.5)
            T_at_t_0[i] = 1.0 - 0.167 * (x[i] - 0.5)
        elif 1.5 <= x[i] <= 3.5:
            rho_at_t_0[i] = 0.634 - 0.3879 * (x[i] - 1.5)
            T_at_t_0[i] = 0.833 - 0.3507 * (x[i] - 1.5)

        V_at_t_0[i] = 0.59 / (rho_at_t_0[i] * A[i])

    rho2 = rho_at_t_0.copy()
    T2 = T_at_t_0.copy()
    V2 = V_at_t_0.copy()

    m2 = np.zeros((nt, nx))
    U1 = rho2 * A
    U2 = rho2 * A * V2
    U3 = rho2 * A * ((T2 / (gamma - 1)) + (gamma / 2) * (V2**2))

    throat = np.argmin(np.abs(A - 1.0))

    rho_throat2 = np.zeros(nt)
    T_throat2 = np.zeros(nt)
    V_throat2 = np.zeros(nt)
    P_throat2 = np.zeros(nt)
    M_throat2 = np.zeros(nt)
    residual2 = np.zeros((nt, 3))

    dU1_dt_p = np.zeros(nx)
    dU2_dt_p = np.zeros(nx)
    dU3_dt_p = np.zeros(nx)

    dU1_dt_c = np.zeros(nx)
    dU2_dt_c = np.zeros(nx)
    dU3_dt_c = np.zeros(nx)

    for k in range(nt):
        U1_old = U1.copy()
        U2_old = U2.copy()
        U3_old = U3.copy()

        dt = np.min(C * dx / (T2**0.5 + V2))

        F1 = U2
        F2 = (U2**2) / U1 + ((gamma - 1) / gamma) * (U3 - 0.5 * gamma * (U2**2) / U1)
        F3 = gamma * U2 * U3 / U1 - 0.5 * gamma * (gamma - 1) * (U2**3) / (U1**2)

        # Predictor Step
        for i in range(1, nx - 1):
            J2 = (1 / gamma) * rho2[i] * T2[i] * ((A[i+1] - A[i]) / dx)
            dU1_dt_p[i] = -(F1[i+1] - F1[i]) / dx
            dU2_dt_p[i] = -(F2[i+1] - F2[i]) / dx + J2
            dU3_dt_p[i] = -(F3[i+1] - F3[i]) / dx

            U1[i] = U1[i] + dU1_dt_p[i] * dt
            U2[i] = U2[i] + dU2_dt_p[i] * dt
            U3[i] = U3[i] + dU3_dt_p[i] * dt

        rho2 = U1 / A
        V2 = U2 / U1
        T2 = (gamma - 1) * ((U3 / U1) - ((gamma / 2) * (V2**2)))

        F1 = U2
        F2 = ((U2**2) / U1) + (((gamma - 1) / gamma) * (U3 - ((gamma / 2) * (U2**2 / U1))))
        F3 = ((gamma * U2 * U3) / U1) - ((gamma * (gamma - 1)) / 2) * (U2**3) / (U1**2)

        # Corrector Step
        for i in range(1, nx - 1):
            J2 = (1 / gamma) * rho2[i] * T2[i] * ((A[i] - A[i-1]) / dx)
            dU1_dt_c[i] = -(F1[i] - F1[i-1]) / dx
            dU2_dt_c[i] = -(F2[i] - F2[i-1]) / dx + J2
            dU3_dt_c[i] = -(F3[i] - F3[i-1]) / dx

        dU1_dt = 0.5 * (dU1_dt_p + dU1_dt_c)
        dU2_dt = 0.5 * (dU2_dt_p + dU2_dt_c)
        dU3_dt = 0.5 * (dU3_dt_p + dU3_dt_c)

        residual2[k, 0] = np.max(np.abs(dU1_dt))
        residual2[k, 1] = np.max(np.abs(dU2_dt))
        residual2[k, 2] = np.max(np.abs(dU3_dt))

        # Final Update
        for j in range(1, nx - 1):
            U1[j] = U1_old[j] + dU1_dt[j] * dt
            U2[j] = U2_old[j] + dU2_dt[j] * dt
            U3[j] = U3_old[j] + dU3_dt[j] * dt

        # Boundary Conditions
        U1[0] = rho2[0] * A[0]
        U2[0] = 2 * U2[1] - U2[2]
        U3[0] = U1[0] * ((T2[0] / (gamma - 1)) + 0.5 * gamma * V2[0]**2)

        U1[-1] = 2 * U1[-2] - U1[-3]
        U2[-1] = 2 * U2[-2] - U2[-3]
        U3[-1] = 2 * U3[-2] - U3[-3]

        rho2 = U1 / A
        V2 = U2 / U1
        T2 = (gamma - 1) * ((U3 / U1) - 0.5 * gamma * V2**2)
        M2 = V2 / (T2**0.5)
        P2 = rho2 * T2
        m2[k, :] = rho2 * A * V2

        rho_throat2[k] = rho2[throat]
        V_throat2[k] = V2[throat]
        T_throat2[k] = T2[throat]
        P_throat2[k] = P2[throat]
        M_throat2[k] = M2[throat]

    return rho2, T2, V2, P2, M2, m2, rho_throat2, V_throat2, T_throat2, P_throat2, M_throat2, residual2


# --- Main Execution Script ---
if __name__ == "__main__":
    nx = 31
    L = 3.0
    x = np.linspace(0, L, nx)
    dx = x[1] - x[0]
    A = 1 + 2.2 * (x - 1.5)**2
    gamma = 1.4
    nt = 1400
    C = 0.5

    # Non-conservative simulation
    t0 = time.time()
    rho1, T1, V1, P1, M1, m1, rho_tr1, V_tr1, T_tr1, P_tr1, M_tr1, res1 = \
        quasi_1d_nozzle_flow_non_conservative(x, dx, A, C, nt, gamma)
    print(f"Time taken by non-conservative form: {time.time() - t0:.4f} s")

    # Conservative simulation
    t0 = time.time()
    rho2, T2, V2, P2, M2, m2, rho_tr2, V_tr2, T_tr2, P_tr2, M_tr2, res2 = \
        quasi_1d_nozzle_flow_conservative(x, dx, A, C, nt, gamma)
    print(f"Time taken by conservative form: {time.time() - t0:.4f} s")

    # Figure 1: Non-conservative Profile
    fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    axs[0].plot(x, M1, 'g-', lw=2)
    axs[0].set_ylabel('Mach Number')
    axs[0].set_title('Non-conservative Form Profile', color='g', fontweight='bold')
    axs[0].grid(True, which='both')

    axs[1].plot(x, P1, 'b-', lw=2)
    axs[1].set_ylabel('Pressure Ratio')
    axs[1].grid(True, which='both')

    axs[2].plot(x, rho1, 'r-', lw=2)
    axs[2].set_ylabel('Density Ratio')
    axs[2].grid(True, which='both')

    axs[3].plot(x, T1, 'm-', lw=2)
    axs[3].set_ylabel('Temperature Ratio')
    axs[3].set_xlabel('x/L')
    axs[3].grid(True, which='both')
    plt.tight_layout()

    # Figure 2: Conservative Profile
    fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    axs[0].plot(x, M2, 'g-', lw=2)
    axs[0].set_ylabel('Mach Number')
    axs[0].set_title('Conservative Form Profile', color='g', fontweight='bold')
    axs[0].grid(True, which='both')

    axs[1].plot(x, P2, 'b-', lw=2)
    axs[1].set_ylabel('Pressure Ratio')
    axs[1].grid(True, which='both')

    axs[2].plot(x, rho2, 'r-', lw=2)
    axs[2].set_ylabel('Density Ratio')
    axs[2].grid(True, which='both')

    axs[3].plot(x, T2, 'm-', lw=2)
    axs[3].set_ylabel('Temperature Ratio')
    axs[3].set_xlabel('x/L')
    axs[3].grid(True, which='both')
    plt.tight_layout()

    # Figure 3: Comparison at Steady State (Mass Flow)
    plt.figure(figsize=(8, 5))
    plt.plot(x, 0.579 * np.ones_like(x), 'k--', label='Exact', lw=2)
    plt.plot(x, m1[-1, :], 'r-', label='Non-conservative', lw=2)
    plt.plot(x, m2[-1, :], 'b-', label='Conservative', lw=2)
    plt.xlabel('x/L')
    plt.ylabel('rho*A*V / rho_0*A*a_0')
    plt.title('Mass Flow Rate Comparison (Steady State)', color='g', fontweight='bold')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig('out1.jpg')
