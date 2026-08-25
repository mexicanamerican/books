import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# ==============================================================================
# Global Parameters & Constants
# ==============================================================================
gamma = 1.4
dum1 = 2.0 / (gamma + 1.0)
dum2 = (gamma - 1.0) / (gamma + 1.0)
dum3 = (gamma - 1.0) / 2.0

# Driven section (Left state)
rhoL = 8.0
pL = 10.0 / gamma
aL = np.sqrt(gamma * pL / rhoL)

# Working section (Right state)
rhoR = 1.0
pR = 1.0 / gamma
aR = 1.0


# ==============================================================================
# Helper Functions for Transformations and Time Step
# ==============================================================================
def hyp_trans_usol_w(usol):
    """Computes W = (rho, rho*U, E) from usol = (rho, U, p)"""
    w = np.zeros_like(usol)
    w[0, :] = usol[0, :]
    w[1, :] = usol[0, :] * usol[1, :]
    w[2, :] = usol[2, :] / (gamma - 1.0) + 0.5 * w[1, :] * usol[1, :]
    return w


def hyp_trans_w_usol(w):
    """Computes usol = (rho, U, p) from W = (rho, rho*U, E)"""
    usol = np.zeros_like(w)
    usol[0, :] = w[0, :]
    usol[1, :] = w[1, :] / w[0, :]
    usol[2, :] = (gamma - 1.0) * (w[2, :] - 0.5 * w[1, :] * usol[1, :])
    return usol


def hyp_trans_w_f(w):
    """Computes F = (rho*U, rho*U^2 + p, (E + p)*U) from W = (rho, rho*U, E)"""
    f = np.zeros_like(w)
    f[0, :] = w[1, :]
    uloc = w[1, :] / w[0, :]
    rhou2 = w[1, :] * uloc
    press = (gamma - 1.0) * (w[2, :] - 0.5 * rhou2)
    f[1, :] = rhou2 + press
    f[3 - 1, :] = (w[2, :] + press) * uloc
    return f


def hyp_calc_dt(w, dx, cfl):
    """Computes the time step dt based on the CFL condition"""
    uloc = w[1, :] / w[0, :]
    press = (gamma - 1.0) * (w[2, :] - 0.5 * w[1, :] * uloc)
    a = np.sqrt(gamma * press / w[0, :])
    return cfl * dx / np.max(np.abs(uloc) + a)


# ==============================================================================
# Exact Solution Functions
# ==============================================================================
def hyp_mach_compat(x):
    """Compatibility relation for the shock tube problem"""
    return (x - 1.0 / x) - aL / dum2 * (1.0 - (pR / pL * (dum1 * gamma * x**2 - dum2))**(dum3 / gamma))


def hyp_shock_tube_exact(x, x0, t):
    """Computes exact solution for the shock tube problem"""
    m_size = len(x)
    uex = np.zeros((3, m_size))

    # Find Shock Mach number Ms using root finding
    ms_solution = fsolve(hyp_mach_compat, 2.0)
    Ms = ms_solution[0]
    print(f"Shock Mach number Ms = {Ms:.6f}")

    dumm = Ms * Ms
    p1 = pR * (dum1 * gamma * dumm - dum2)
    rho1 = rhoR / (dum1 / dumm + dum2)
    U1 = dum1 * (Ms - 1.0 / Ms)

    a2 = aL - dum3 * U1
    rho2 = rhoL * (p1 / pL)**(1.0 / gamma)

    x1 = x0 - aL * t
    x2 = x0 + (U1 - a2) * t
    x3 = x0 + U1 * t
    x4 = x0 + Ms * t

    # Region L
    idum = np.where(x <= x1)[0]
    uex[0, idum] = rhoL
    uex[1, idum] = 0.0
    uex[2, idum] = pL

    # Expansion Region E
    idum = np.where((x1 < x) & (x <= x2))[0]
    uex[1, idum] = dum1 * (aL + (x[idum] - x0) / t)
    adet = dum1 * (aL - dum3 * (x[idum] - x0) / t)
    uex[2, idum] = pL * (adet / aL)**(2.0 * gamma / (gamma - 1.0))
    uex[0, idum] = gamma * uex[2, idum] / (adet * adet)

    # Region 2
    idum = np.where((x2 < x) & (x <= x3))[0]
    uex[0, idum] = rho2
    uex[1, idum] = U1
    uex[2, idum] = p1

    # Region 1
    idum = np.where((x3 < x) & (x <= x4))[0]
    uex[0, idum] = rho1
    uex[1, idum] = U1
    uex[2, idum] = p1

    # Region R
    idum = np.where(x4 < x)[0]
    uex[0, idum] = 1.0
    uex[1, idum] = 0.0
    uex[2, idum] = 1.0 / gamma

    return uex


# ==============================================================================
# Plotting Utility
# ==============================================================================
def hyp_plot_graph2(t, xx, u1, u2, txt1, txt2):
    """Plots exact vs numerical solutions side-by-side or in subplots"""
    ylabels = [r'$\rho$', r'$U$', r'$p$']
    titles = ['Density', 'Velocity', 'Pressure']

    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    for k in range(3):
        axes[k].plot(xx, u1[k, :], 'r-', label=txt1, linewidth=2)
        axes[k].plot(xx, u2[k, :], 'bo-', label=txt2, linewidth=1.5, markersize=4)
        axes[k].set_title(f"{titles[k]} at t = {t}", fontsize=12)
        axes[k].set_xlabel('x')
        axes[k].set_ylabel(ylabels[k])
        axes[k].legend()
        axes[k].grid(True)

    plt.tight_layout()
    plt.savefig('sod1.jpg')


# ==============================================================================
# Main Driver Simulation Script
# ==============================================================================
def main():
    M = 81
    dx = 1.0 / (M - 1)
    xx = np.linspace(0, 1.0, M)
    x0 = 0.5

    ip = np.where(xx >= x0)[0]
    in_idx = np.where(xx < x0)[0]

    usol = np.zeros((3, M))
    usol[0, ip] = rhoR
    usol[0, in_idx] = rhoL
    usol[1, :] = 0.0
    usol[2, ip] = pR
    usol[2, in_idx] = pL

    w = hyp_trans_usol_w(usol)

    tfinal = 0.2
    t = 0.0
    cfl = 0.95
    D = 2.0
    Ddx = D * dx

    txt1 = f"MacCormack (D={D})"
    print("Running MacCormack scheme...")

    while t < tfinal:
        dt = hyp_calc_dt(w, dx, cfl)
        if t + dt > tfinal:
            dt = tfinal - t

        F = hyp_trans_w_f(w)
        
        # Add artificial dissipation to F
        diff_w = np.diff(w, n=1, axis=1)
        pad_left = np.zeros((3, 1))
        F = F - Ddx * np.hstack((pad_left, diff_w))

        # Predictor step (wtilde, Ftilde for 0 to M-2)
        wtilde = w[:, 0:M-1] - (dt / dx) * (F[:, 1:M] - F[:, 0:M-1])
        Ftilde = hyp_trans_w_f(wtilde)

        # Add artificial dissipation to Ftilde
        diff_wtilde = np.diff(wtilde, n=1, axis=1)
        pad_right = np.zeros((3, 1))
        Ftilde = Ftilde - Ddx * np.hstack((diff_wtilde, pad_right))

        # Corrector step (update internal points 1 to M-2)
        w[:, 1:M-1] = 0.5 * (w[:, 1:M-1] + wtilde[:, 1:M-1]) - \
                      0.5 * (dt / dx) * (Ftilde[:, 1:M-1] - Ftilde[:, 0:M-2])

        t += dt

    print(f"Final time reached: {t:.4f}")

    usol_num = hyp_trans_w_usol(w)
    uex = hyp_shock_tube_exact(xx, x0, tfinal)

    hyp_plot_graph2(tfinal, xx, uex, usol_num, 'Exact sol.', txt1)


if __name__ == "__main__":
    main()
