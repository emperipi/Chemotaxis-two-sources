# -------------------------------------------------------------
# prw_2_grads_module.py
# -------------------------------------------------------------
# Module for simulating chemotactic cell trajectories and analysis
# in a two-source environment. Includes functions for trajectory
# simulation, histogram generation, gradient measurement, and movie creation.
# -------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from scipy.special import erf, erfc
from matplotlib.colors import LogNorm, PowerNorm 
from matplotlib import animation

# -------------------------------------------------------------
# Trajectory simulation for a single cell
# -------------------------------------------------------------
@njit()
def traj_simulation(x0, y0, phi0, Rcell, v0, tau, Dtheta, paramA, paramB, tau_sensing, T, dt, L='False'):
    """
    Simulate a single cell trajectory in a two-source chemotaxis environment.
    Returns arrays of x, y positions and orientation phi.
    """
    n_steps = int(T/dt)
    n_sensing = int(tau_sensing/dt)
    x, y = np.zeros(n_steps), np.zeros(n_steps)
    x[0], y[0] = x0, y0
    phi = np.zeros(n_steps)
    phi[0] = phi0
    for i in range(n_steps-1):
        if(i % n_sensing == 0):
            phi_bar = measure_gradients_direction(x[i], y[i], Rcell, paramA, paramB)
        dphi = phi[i] - phi_bar        
        phi[i+1] = phi[i] - 1/tau * np.sin(dphi)*dt + np.sqrt(2*Dtheta*dt)*np.random.normal(0,1)
        x[i+1] = x[i] + v0*np.cos(phi[i])*dt
        y[i+1] = y[i] + v0*np.sin(phi[i])*dt
        # Periodic boundary conditions
        if L != 'False':
            if x[i+1] > L:
                x[i+1] += -2*L
            if x[i+1] < -L:
                x[i+1] += 2*L
            if y[i+1] > L:
                y[i+1] += -2*L
            if y[i+1] < -L:
                y[i+1] += 2*L
    return x, y, phi

# -------------------------------------------------------------
# 2D histogram of cell positions over time
# -------------------------------------------------------------
@njit()
def traj_simulation_hist2D(x0, y0, phi0, Rcell, v0, tau, Dtheta, paramA, paramB, tau_sensing, T, dt, L, nbins=500, t_snapshot=100):
    """
    Simulate cell trajectory and accumulate 2D histogram of positions.
    """
    n_steps = int(T/dt)
    n_sensing = int(tau_sensing/dt)
    n_snapshot = int(t_snapshot/dt)
    l = L // 2
    dr_bins = l // nbins
    hist_2d = np.zeros((nbins, nbins))
    x, y = np.zeros(n_steps), np.zeros(n_steps)
    x[0], y[0] = x0, y0
    phi = np.zeros(n_steps)
    phi[0] = phi0
    for i in range(n_steps-1):
        if(i % n_sensing == 0):
            phi_bar = measure_gradients_direction(x[i], y[i], Rcell, paramA, paramB)
        dphi = phi[i] - phi_bar        
        phi[i+1] = phi[i] - 1/tau * np.sin(dphi)*dt + np.sqrt(2*Dtheta*dt)*np.random.normal(0,1)
        x[i+1] = x[i] + v0*np.cos(phi[i])*dt
        y[i+1] = y[i] + v0*np.sin(phi[i])*dt
        # Accumulate histogram at snapshot intervals
        if(i % n_snapshot == 0):
            if (x[i+1] > -l/2) & (x[i+1] < l/2) & (y[i+1] > -l/2) & (y[i+1] < l/2):
                ix = int((x[i+1] + l/2 - dr_bins/2) // dr_bins)
                iy = int((y[i+1] + l/2 - dr_bins/2) // dr_bins)
                hist_2d[ix, iy] += 1
        # Periodic boundary conditions
        if x[i+1] > L:
            x[i+1] += -2*L
        if x[i+1] < -L:
            x[i+1] += 2*L
        if y[i+1] > L:
            y[i+1] += -2*L
        if y[i+1] < -L:
            y[i+1] += 2*L
    return hist_2d

# -------------------------------------------------------------
# Vectorized snapshot update for multiple trajectories
# -------------------------------------------------------------
@njit()
def traj_simulation_snapshots(x, y, phi, v0, tau, Dtheta, phi_bar, dt, L):
    """
    Update positions and orientation for multiple trajectories (vectorized).
    """
    x += v0 * np.cos(phi) * dt
    y += v0 * np.sin(phi) * dt
    Ntraj = len(phi)
    dphi = phi - phi_bar
    phi += -1/tau * np.sin(dphi) * dt + np.sqrt(2*Dtheta*dt) * np.random.normal(0, 1, Ntraj)
    # Periodic boundary conditions
    x[x > L] = x[x > L] - 2*L 
    x[x < -L] = x[x < -L] + 2*L
    y[y > L] = y[y > L] - 2*L 
    y[y < -L] = y[y < -L] + 2*L
    return x, y, phi

# -------------------------------------------------------------
# Snapshot histogram for a set of positions
# -------------------------------------------------------------
@njit()
def snapshot(x, y, l, Ntraj, nbins, dr_bins):
    """
    Compute 2D histogram snapshot for Ntraj positions.
    """
    snapt = np.zeros((nbins**2))
    for i in range(Ntraj):
        if (x[i] > -l/2) & (x[i] < l/2) & (y[i] > -l/2) & (y[i] < l/2):
            ix = int((x[i] + l/2 - dr_bins/2) // dr_bins)
            iy = int((y[i] + l/2 - dr_bins/2) // dr_bins)
            ih = ix * nbins + iy
            snapt[ih] += 1
    snapt = snapt / np.sum(snapt) / dr_bins**2
    return snapt

# -------------------------------------------------------------
# Gradient measurement for a single position
# -------------------------------------------------------------
@njit()
def measure_gradients_direction(x, y, Rcell, paramA, paramB, epsilon=16.0):
    """
    Compute the mean chemotactic direction at position (x, y)
    based on two sources and their parameters.
    """
    nA, KDA, lambdaA, SA, x0A, y0A = paramA
    nB, KDB, lambdaB, SB, x0B, y0B = paramB
    # Source A calculations
    rA = ((x-x0A)**2 + (y-y0A)**2)**0.5
    rsA = rA/(2*epsilon)**0.5
    sA = (epsilon/2/lambdaA**2)**0.5
    Erf1A = erf(rsA-sA)
    Erfc2A = erfc(rsA+sA)
    if rA == 0:
        CrA = SA * ((2/np.pi/epsilon)**0.5 - np.exp(sA**2)*(1-erf(sA))/lambdaA)
        pA = 0
    else:
        frA = 0.5*np.exp(sA**2) * (1+Erf1A - np.exp(2*rA/lambdaA)*Erfc2A)
        fp_over_fA = ((8/np.pi/epsilon)**0.5 * np.exp(-(rsA-sA)**2)
                - 2/lambdaA * Erfc2A*np.exp(2*rA/lambdaA)) / (1 + Erf1A - Erfc2A*np.exp(2*rA/lambdaA))
        CrA = SA * np.exp(-rA/lambdaA) / rA * frA
        pA = 2*Rcell*np.abs(1/rA + 1/lambdaA - fp_over_fA)
    sigma2_pA = 8*(CrA + KDA)**2/(nA*CrA*KDA)
    SNRA = pA**2/sigma2_pA
    # Source B calculations
    rB = ((x-x0B)**2 + (y-y0B)**2)**0.5
    rsB = rB/(2*epsilon)**0.5
    sB = (epsilon/2/lambdaB**2)**0.5
    Erf1B = erf(rsB-sB)
    Erfc2B = erfc(rsB+sB)
    if rB == 0:
        CrB = SB * ((2/np.pi/epsilon)**0.5 - np.exp(sB**2)*(1-erf(sB))/lambdaB)
        pB = 0
    else:
        frB = 0.5*np.exp(sB**2) * (1+Erf1B - np.exp(2*rB/lambdaB)*Erfc2B)
        fp_over_fB = ((8/np.pi/epsilon)**0.5 * np.exp(-(rsB-sB)**2)
                - 2/lambdaB * Erfc2B*np.exp(2*rB/lambdaB)) / (1 + Erf1B - Erfc2B*np.exp(2*rB/lambdaB))
        CrB = SB * np.exp(-rB/lambdaB) / rB * frB
        pB = 2*Rcell*np.abs(1/rB + 1/lambdaB - fp_over_fB)
    sigma2_pB = 8*(CrB + KDB)**2/(nB*CrB*KDB)
    SNRB = pB**2/sigma2_pB
    # Direction calculation (weights can be adjusted)
    wA = 1
    wB = 1
    phi_A = np.arctan2(y0A-y, x0A-x) + np.random.vonmises(0, SNRA)
    phi_B = np.arctan2(y0B-y, x0B-x) + np.random.vonmises(0, SNRB)
    phi_bar = np.arctan2(wA * np.sin(phi_A) + wB * np.sin(phi_B), wA * np.cos(phi_A) + wB * np.cos(phi_B))
    return phi_bar

# -------------------------------------------------------------
# Vectorized gradient measurement for arrays of positions
# -------------------------------------------------------------
def measure_gradients_direction_vec(x, y, Rcell, paramA, paramB, epsilon=16.0):
    """
    Vectorized version of gradient direction measurement for arrays.
    """
    nA, KDA, lbdaA, SA, x0A, y0A = paramA
    nB, KDB, lbdaB, SB, x0B, y0B = paramB
    rA = ((x-x0A)**2 + (y-y0A)**2)**0.5
    rB = ((x-x0B)**2 + (y-y0B)**2)**0.5
    SNRA, CrA = SNR_regularized_delta_source(rA, SA, nA, lbdaA, KDA, Rcell, epsilon)
    SNRB, CrB = SNR_regularized_delta_source(rB, SB, nB, lbdaB, KDB, Rcell, epsilon)
    phi_A = np.arctan2(y0A-y, x0A-x) + np.random.vonmises(0, SNRA)
    phi_B = np.arctan2(y0B-y, x0B-x) + np.random.vonmises(0, SNRB)
    phi_bar = np.arctan2(np.sin(phi_A) + np.sin(phi_B), np.cos(phi_A) + np.cos(phi_B))
    return phi_bar

# -------------------------------------------------------------
# SNR calculation for regularized delta source
# -------------------------------------------------------------
def SNR_regularized_delta_source(r, S0, n, lbda, KD, Rcell, epsilon=16.0):
    """
    Calculate SNR and concentration for a regularized delta source.
    """
    r[r == 0] = -1.0
    r_ep = r / (2*epsilon)**0.5
    s_ep = (epsilon/2/lbda**2)**0.5
    Erf1 = erf(r_ep - s_ep)
    Erfc2 = erfc(r_ep + s_ep)
    fr = 0.5 * np.exp(s_ep**2) * (1 + Erf1 - np.exp(2*r/lbda)*Erfc2)
    Cr = S0 * np.exp(-r/lbda) / r * fr
    fp_over_f = ((8/np.pi/epsilon)**0.5 * np.exp(-(r_ep-s_ep)**2)
                 - 2/lbda * Erfc2 * np.exp(2*r/lbda)) \
                / (1 + Erf1 - Erfc2 * np.exp(2*r/lbda))
    p = 2 * Rcell * np.abs(1/r + 1/lbda - fp_over_f)
    Cr[r < 0] = S0 * ((2/np.pi/epsilon)**0.5 - np.exp(s_ep**2)*(1-erf(s_ep))/lbda)
    p[r < 0] = 0
    sigma2_p = 8 * (Cr + KD)**2 / (n * Cr * KD)
    return p**2 / sigma2_p, Cr

# -------------------------------------------------------------
# Movie frame update and animation creation
# -------------------------------------------------------------
def update_movie_frames(s, im, ax, X, Y, snap_lst, dmeanx, dmeany, nbins, x0A, x0B, cmap, norm=None):
    """
    Update function for movie animation frames.
    """
    snap = snap_lst[s].reshape((nbins, nbins))
    ax.set_aspect('equal')
    ax.clear()
    ax.streamplot(X, Y, dmeanx, dmeany, density=1.5, linewidth=0.4, color='g', arrowstyle='->', arrowsize=0.5) 
    im = ax.pcolormesh(X, Y, snap.T, cmap=cmap, norm=norm)
    ax.scatter(x0A, 0, marker='o', s=5, c='w')
    ax.scatter(x0B, 0, marker='o', s=5, c='w')
    ax.text(x0A+5, 5, 'A', c='w', fontweight='bold')
    ax.text(x0B+5, 5, 'B', c='w', fontweight='bold')
    return im

def make_movie(filename, X, Y, snap_lst, dmeanx, dmeany, nbins, x0A, x0B):   
    """
    Create and save movie animation of cell density and mean direction.
    """
    plt.style.use('dark_background')
    cmap = 'coolwarm'
    fig1, ax1 = plt.subplots(figsize=(5,5))
    Nframes = len(snap_lst)
    snap_max = np.max(snap_lst)
    print(snap_max)
    snap = snap_lst[0].reshape((nbins, nbins))
    ax1.set_aspect('equal')
    ax1.streamplot(X, Y, dmeanx, dmeany, density=1.5, linewidth=0.4, color='g', arrowstyle='->', arrowsize=0.5) 
    im1 = ax1.pcolormesh(X, Y, snap.T, cmap=cmap, norm=LogNorm(vmax=snap_max))
    ax1.scatter(x0A, 0, marker='o', s=5, c='w')
    ax1.scatter(x0B, 0, marker='o', s=5, c='w')
    ax1.text(x0A+5, 5, 'A', c='w', fontweight='bold')
    ax1.text(x0B+5, 5, 'B', c='w', fontweight='bold')
    ani1 = animation.FuncAnimation(fig1, update_movie_frames, frames=range(Nframes), interval=5, repeat=False, 
                                    fargs=(im1, ax1, X, Y, snap_lst, dmeanx, dmeany, nbins, x0A, x0B, cmap, LogNorm(vmax=snap_max)))
    fig1.colorbar(im1, ax=ax1)
    FFwriter = animation.FFMpegWriter()
    ani1.save('Movie_'+filename+'_LogNorm.mp4', writer=FFwriter, dpi=300)
    plt.close()
