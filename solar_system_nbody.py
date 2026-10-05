"""
N-Body Solar System Simulation
================================
Sun + Mercury + Venus + Earth + Mars + Jupiter

Units:
    distance  =  AU
    time      =  years
    mass      =  solar masses
    G         =  4*pi^2  (from Kepler's Third Law)

All 6 bodies interact gravitationally.
Sun is particle 0 — M_sun disappears as a separate constant.
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =============================================================================
# Particle data  <-- ONLY SECTION TO EDIT
# =============================================================================

G = 4 * np.pi**2    # AU^3 / (yr^2 Msun)

pos0 = np.array([
    [0.000, 0.0],   # Sun      row 0
    [0.387, 0.0],   # Mercury  row 1
    [0.723, 0.0],   # Venus    row 2
    [1.000, 0.0],   # Earth    row 3
    [1.524, 0.0],   # Mars     row 4
    [5.203, 0.0],   # Jupiter  row 5
])

vel0 = np.array([
    [0.0, 0.0],                      # Sun (set below)
    [0.0, np.sqrt(G / 0.387)],       # Mercury
    [0.0, np.sqrt(G / 0.723)],       # Venus
    [0.0, np.sqrt(G / 1.000)],       # Earth
    [0.0, np.sqrt(G / 1.524)],       # Mars
    [0.0, np.sqrt(G / 5.203)],       # Jupiter
])

mass   = np.array([1.0, 1.651e-7, 2.447e-6, 3.003e-6, 3.213e-7, 9.548e-4])
names  = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter"]
colors = ["yellow", "gray",  "orange", "blue",  "red",  "brown"]
N      = len(mass)

# =============================================================================
# Zero total momentum  (set Sun velocity)
# =============================================================================

vel0[0] = -np.sum(mass[1:, None] * vel0[1:], axis=0) / mass[0]

# =============================================================================
# Flatten into state vector
# =============================================================================

y0 = np.concatenate([pos0.flatten(), vel0.flatten()])   # shape (4*N,) = (24,)

# =============================================================================
# rhs — 5-line universal force law for all pairs
# =============================================================================

def rhs(y, t):
    pos = y[:2*N].reshape(N, 2)
    vel = y[2*N:].reshape(N, 2)

    diff = pos[None, :, :] - pos[:, None, :]        # (N, N, 2)
    dist = np.linalg.norm(diff, axis=2)              # (N, N)
    np.fill_diagonal(dist, 1.0)                      # avoid 0/0

    acc = G * (mass[None, :, None] * diff
               / dist[:, :, None]**3
               ).sum(axis=1)                         # (N, 2)

    return np.concatenate([vel.flatten(), acc.flatten()])

# =============================================================================
# Integrate  (12 years = 1 Jupiter orbit)
# =============================================================================

t = np.linspace(0, 12, 20000)

print("Integrating {} bodies for {:.0f} years ...".format(N, t[-1]))
sol = odeint(rhs, y0, t)
print("Done.  sol.shape =", sol.shape)

# =============================================================================
# Extract positions and velocities
# =============================================================================

pos_all = sol[:, :2*N].reshape(len(t), N, 2)   # (nsteps, N, 2)
vel_all = sol[:, 2*N:].reshape(len(t), N, 2)   # (nsteps, N, 2)

# =============================================================================
# Energy conservation  (fully vectorized — no loops)
# =============================================================================

# Kinetic energy at every time step  ->  shape (nsteps,)
KE = 0.5 * np.sum(mass[None, :, None] * vel_all**2, axis=(1, 2))

# All pairwise distance vectors  ->  shape (nsteps, N, N, 2)
diff_all = pos_all[:, None, :, :] - pos_all[:, :, None, :]

# All pairwise distances  ->  shape (nsteps, N, N)
dist_all = np.linalg.norm(diff_all, axis=3)
dist_all[:, np.arange(N), np.arange(N)] = 1.0   # diagonal: avoid 0/0

# Outer product of masses  ->  shape (N, N)
mass_ij = mass[:, None] * mass[None, :]

# Potential energy: sum upper triangle (i < j) to avoid double counting
PE = -G * np.sum(
    np.triu(mass_ij, k=1)[None, :, :] / dist_all,
    axis=(1, 2)
)                                                  # shape (nsteps,)

E = KE + PE

print("\nEnergy conservation")
print("  E[0]           = {:.6e}".format(E[0]))
print("  std(E)         = {:.6e}".format(E.std()))
print("  relative error = {:.6e}".format(E.std() / abs(E.mean())))

# =============================================================================
# Plot 1 — Orbits
# =============================================================================

plt.figure(figsize=(7, 7))
for i in range(N):
    plt.plot(pos_all[:, i, 0],
             pos_all[:, i, 1],
             color=colors[i],
             label=names[i],
             lw=0.8)
plt.axis('equal')
plt.xlabel("x (AU)")
plt.ylabel("y (AU)")
plt.title("Solar System Orbits")
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# =============================================================================
# Plot 2 — Energy conservation
# =============================================================================

fig, ax = plt.subplots(2, 1, figsize=(8, 5))

ax[0].plot(t, E, color='steelblue', lw=0.8)
ax[0].set_ylabel("Total Energy")
ax[0].set_title("Energy Conservation")

# Relative deviation from initial value
dE = (E - E[0]) / abs(E[0])
ax[1].plot(t, dE, color='tomato', lw=0.8)
ax[1].set_ylabel("Relative Error  (E - E0) / |E0|")
ax[1].set_xlabel("Time (years)")

plt.tight_layout()
plt.show()

# =============================================================================
# Plot 3 — Animation
# =============================================================================

fig, ax = plt.subplots(figsize=(7, 7))
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

pad = 1.15 * np.max(np.linalg.norm(pos_all.reshape(-1, 2), axis=1))
ax.set_xlim(-pad, pad)
ax.set_ylim(-pad, pad)
ax.set_aspect('equal')
ax.set_title("Solar System — odeint", color='white')
ax.tick_params(colors='white')

pts    = [ax.plot([], [], 'o', ms=6, color=colors[i], label=names[i])[0] for i in range(N)]
trails = [ax.plot([], [], '-', lw=0.8, color=colors[i], alpha=0.5)[0]   for i in range(N)]
ax.legend(loc='upper right', facecolor='black', labelcolor='white', fontsize=9)

def animate(k):
    for i in range(N):
        pts[i].set_data([pos_all[k, i, 0]], [pos_all[k, i, 1]])
        trails[i].set_data(pos_all[:k, i, 0], pos_all[:k, i, 1])
    return pts + trails

ani = FuncAnimation(fig, animate, frames=len(t), interval=10, blit=True)
plt.tight_layout()
plt.show()


