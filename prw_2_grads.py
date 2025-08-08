# -------------------------------------------------------------
# Chemotactic Cell Trajectory Simulation and Data Output Script
# -------------------------------------------------------------
# This script simulates cell trajectories in a two-source chemotaxis environment
# and saves the positions to a file for further analysis.
#
# Usage: python prw_2_grads.py <dt_r> <T> <Ntraj> <v0> <Dtheta> <tau> <nA> <KDA> <lambdaA> <SA> <nB> <KDB> <lambdaB> <SB>
#
# Arguments:
#   dt_r    : Relative timestep to tau (dt/tau)
#   T       : Total simulation time
#   Ntraj   : Number of trajectories to simulate
#   v0      : Cell speed
#   Dtheta  : Polarity noise
#   tau     : Relaxation rate
#   nA      : Number of type A receptors
#   KDA     : Dissociation constant for A
#   lambdaA : Chemokine A lengthscale
#   SA      : Source A strength
#   nB      : Number of type B receptors
#   KDB     : Dissociation constant for B
#   lambdaB : Chemokine B lengthscale
#   SB      : Source B strength
#
# Output:
#   Saves positions to 'prw2grads_pos_vs_t_<parameters>.dat'
# -------------------------------------------------------------

import numpy as np
import sys
import prw_2_grads_module as md
from numba import jit

# --- Parse command-line arguments ---
dt_r = float(sys.argv[1])      # Relative timestep to tau: dt/tau
T = float(sys.argv[2])         # Total simulation time
Ntraj = int(sys.argv[3])       # Number of trajectories
v0 = float(sys.argv[4])        # Cell speed
Rcell = 10                     # Cell radius (fixed)
Dtheta = float(sys.argv[5])    # Polarity noise
tau = float(sys.argv[6])       # Relaxation rate

dt = dt_r * tau                # Actual timestep
ntau = int(tau / dt)           # Steps per relaxation time
tau_sensing = 1                # Gradient sensing correlation time [s]

# --- Chemokine A parameters ---
nA = int(sys.argv[7])          # Number of type A receptors
KDA = float(sys.argv[8])       # Dissociation constant A
lambdaA = float(sys.argv[9])   # Chemokine A lengthscale
SA = float(sys.argv[10])       # Source A strength

# --- Chemokine B parameters ---
nB = int(sys.argv[11])         # Number of type B receptors
KDB = float(sys.argv[12])      # Dissociation constant B
lambdaB = float(sys.argv[13])  # Chemokine B lengthscale
SB = float(sys.argv[14])       # Source B strength

# --- Output filename construction ---
filename = (
    'N_{0}_v0_{1}_tau_{2}_Dth_{3}_nA_{4}_KDA_{5}_lbdaA_{6}_SA_{7}_nB_{8}_KDB_{9}_lbdaB_{10}_SB_{11}_T_{12}_dt_{13}'
    .format(Ntraj, v0, tau, Dtheta, nA, KDA, lambdaA, SA, nB, KDB, lambdaB, SB, T, dt_r)
)
pos_t = "prw2grads_pos_vs_t_" + filename + ".dat"
# msd_t = "prw2grads_msd_vs_t_" + filename + ".dat"  # Uncomment to save MSD data

# --- Simulation grid and histogram setup (not used in output) ---
L = 2000                      # Simulation box size
nbins = 500                   # Number of bins for 2D histogram
t_snapshot = T / 50           # Snapshot interval
hist_2d = np.zeros((nbins, nbins))

# --- Source positions ---
x0A, y0A = -200, 0            # Position of source A
x0B, y0B = 200, 0             # Position of source B

paramA = nA, KDA, lambdaA, SA, x0A, y0A
paramB = nB, KDB, lambdaB, SB, x0B, y0B

n_steps = int(T / dt)
x0, y0 = 0, 0                 # Initial cell position
xs, ys = [], []               # Lists to store trajectory positions

# --- Main simulation loop ---
for i in range(Ntraj):
    phi0 = np.random.uniform(-np.pi, np.pi)  # Random initial orientation
    # x0 = np.random.uniform(-L, L)          # Uncomment for random initial x
    # y0 = np.random.uniform(-L, L)          # Uncomment for random initial y

    # Simulate trajectory
    x, y, phi = md.traj_simulation(
        x0, y0, phi0, Rcell, v0, tau, Dtheta,
        paramA, paramB, tau_sensing, T, dt, L
    )
    # Downsample trajectory for output
    xs.extend(x[::2 * ntau])
    ys.extend(y[::2 * ntau])


# --- Save positions to file ---
with open(pos_t, "w") as f:
    a = np.array((xs, ys)).T
    np.savetxt(f, a)
