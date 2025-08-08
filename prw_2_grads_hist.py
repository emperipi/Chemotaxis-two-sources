import numpy as np
import sys
import prw_2_grads_module as md

# Parse simulation parameters from command-line arguments
dt_r = float(sys.argv[1])  # Relative timestep to tau: dt/tau
T = float(sys.argv[2])     # Total simulation time
Ntraj = int(sys.argv[3])   # Number of cell trajectories
v0 = float(sys.argv[4])    # Cell speed
Rcell = 10                 # Cell radius (fixed)
Dtheta = float(sys.argv[5])# Polarity noise
tau = float(sys.argv[6])   # Relaxation rate

dt = dt_r * tau            # Actual timestep
ntau = int(tau / dt)       # Number of steps per tau
tau_sensing = 1            # Gradient sensing correlation time [s]

# Chemokine A parameters
nA = int(sys.argv[7])
KDA = float(sys.argv[8])
lambdaA = float(sys.argv[9])
SA = float(sys.argv[10])

# Chemokine B parameters
nB = int(sys.argv[11])
KDB = float(sys.argv[12])
lambdaB = float(sys.argv[13])
SB = float(sys.argv[14])

# Output filename encoding all parameters
filename = (
    f"N_{Ntraj}_v0_{v0}_tau_{tau}_Dth_{Dtheta}_nA_{nA}_KDA_{KDA}_lbdaA_{lambdaA}_SA_{SA}"
    f"_nB_{nB}_KDB_{KDB}_lbdaB_{lambdaB}_SB_{SB}_T_{T}_dt_{dt_r}"
)
hist_t = f"prw2grads_hist_{filename}.dat"

# Simulation box and histogram settings
L = 2000
nbins = 250
t_snapshot = 5
hist_2d = np.zeros((nbins, nbins))

# Source positions
x0A, y0A = -100, 0
x0B, y0B = 100, 0
paramA = (nA, KDA, lambdaA, SA, x0A, y0A)
paramB = (nB, KDB, lambdaB, SB, x0B, y0B)

n_steps = int(T / dt)

# Simulate Ntraj independent cell trajectories and accumulate histogram
for i in range(Ntraj):
    phi0 = np.random.uniform(-np.pi, np.pi)  # Random initial orientation
    x0 = np.random.uniform(-L, L)            # Random initial x position
    y0 = np.random.uniform(-L, L)            # Random initial y position
    # Simulate trajectory and accumulate histogram
    hist_2di = md.traj_simulation_hist2D(
        x0, y0, phi0, Rcell, v0, tau, Dtheta,
        paramA, paramB, tau_sensing, T, dt, L, nbins, t_snapshot
    )
    hist_2d += hist_2di

# Save final histogram to file
with open(hist_t, "w") as f:
    np.savetxt(f, hist_2d)