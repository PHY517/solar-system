# Solar System N-Body Simulation

Two Python scripts that simulate the gravitational motion of the inner Solar System
(Sun, Mercury, Venus, Earth, Mars, Jupiter) in 2D, using two different numerical
integrators. Written for **PHY 517 — Computational Physics**.

---

## Files

| File | Integrator | Dependencies |
|------|-----------|--------------|
| `solar_system_nbody.py` | `scipy.integrate.odeint` (RK45) | numpy, scipy, matplotlib |
| `solar_system_vv.py`    | Velocity Verlet (built-in)       | numpy, matplotlib |

Both files share the same physics, the same initial conditions, and produce the
same three output windows: orbit plot, energy conservation plot, and animation.

---

## Physics

### Newton's Law of Gravitation

Every pair of bodies $(i, j)$ interacts via

$$\mathbf{F}_{ij} = G \frac{m_i m_j}{|\mathbf{r}_j - \mathbf{r}_i|^3}(\mathbf{r}_j - \mathbf{r}_i)$$

The acceleration on body $i$ due to all other bodies is

$$\mathbf{a}_i = G \sum_{j \neq i} m_j \frac{\mathbf{r}_j - \mathbf{r}_i}{|\mathbf{r}_j - \mathbf{r}_i|^3}$$

This is the **N-body gravitational problem**. For $N = 6$ bodies there are
$\binom{6}{2} = 15$ unique interacting pairs.

### Units

Astronomical units are used so that Kepler's Third Law gives $G$ exactly:

| Quantity | Unit |
|----------|------|
| Distance | AU (astronomical unit) |
| Time | year |
| Mass | solar mass |
| $G$ | $4\pi^2$ AU³ yr⁻² $M_\odot^{-1}$ |

With these units a planet in a circular orbit at $r = 1$ AU has orbital period
exactly $T = 1$ year, which is easy to verify.

### Initial Conditions

All planets start on the positive $x$-axis with circular orbital speeds:

$$v_c = \sqrt{\frac{G M_\odot}{r}}$$

where $r$ is the semi-major axis.

| Body | Row | $r$ (AU) | Mass ($M_\odot$) |
|------|-----|----------|-----------------|
| Sun | 0 | 0.000 | 1.0 |
| Mercury | 1 | 0.387 | 1.651 × 10⁻⁷ |
| Venus | 2 | 0.723 | 2.447 × 10⁻⁶ |
| Earth | 3 | 1.000 | 3.003 × 10⁻⁶ |
| Mars | 4 | 1.524 | 3.213 × 10⁻⁷ |
| Jupiter | 5 | 5.203 | 9.548 × 10⁻⁴ |

### The Sun as Particle 0

The Sun is **not** fixed at the origin. It is treated as a regular particle
(row 0 in every array). This removes `M_sun` as a separate constant — the
Sun's mass is simply `mass[0] = 1.0`.

To prevent the entire system from drifting, the Sun's initial velocity is set
so that the **total momentum is zero**:

$$\mathbf{v}_\odot = -\frac{\sum_{i=1}^{N-1} m_i \mathbf{v}_i}{m_\odot}$$

In code:

```python
vel0[0] = -np.sum(mass[1:, None] * vel0[1:], axis=0) / mass[0]
```

### Conservation Laws

For a closed gravitational system, two quantities are conserved:

**Total energy**

$$E = \underbrace{\frac{1}{2}\sum_i m_i v_i^2}_{T} \underbrace{- G \sum_{i<j} \frac{m_i m_j}{r_{ij}}}_{V}$$

**Total angular momentum** (z-component in 2D)

$$L_z = \sum_i m_i (x_i \dot{y}_i - y_i \dot{x}_i)$$

Both are monitored as integration quality checks. A relative energy error
below $10^{-6}$ indicates a well-resolved simulation.

---

## Algorithms

### Vectorized Acceleration (shared by both files)

All $N^2$ pairwise interactions are computed simultaneously using NumPy
broadcasting — no Python loops:

```python
def acceleration(pos):
    diff = pos[None, :, :] - pos[:, None, :]   # (N, N, 2)  all rij vectors
    dist = np.linalg.norm(diff, axis=2)         # (N, N)     all |rij|
    np.fill_diagonal(dist, 1.0)                 # avoid 0/0 on diagonal
    acc  = G * (mass[None, :, None] * diff
                / dist[:, :, None]**3
                ).sum(axis=1)                   # (N, 2)
    return acc
```

The self-force vanishes automatically because `diff[i, i] = pos[i] - pos[i] = 0`.

### File 1 — `solar_system_nbody.py` (odeint / RK45)

`scipy.integrate.odeint` solves the ODE system internally using an
adaptive-step Runge-Kutta method (LSODA, order 4–5).

The positions and velocities are packed into a flat state vector:

```
y = [x₀, y₀, x₁, y₁, ..., x₅, y₅,  vx₀, vy₀, ..., vx₅, vy₅]
    |-------- positions (2N) ----------|-------- velocities (2N) --------|
```

The `rhs(y, t)` function unpacks `y`, computes accelerations, and returns
`dy/dt` as a flat array. After integration, `sol` is reshaped:

```python
pos_all = sol[:, :2*N].reshape(len(t), N, 2)   # (nsteps, N, 2)
vel_all = sol[:, 2*N:].reshape(len(t), N, 2)
```

**Pros:** Adaptive step size; high-order accuracy per step.  
**Cons:** Non-symplectic — energy drifts monotonically over long simulations.

### File 2 — `solar_system_vv.py` (Velocity Verlet)

The Velocity Verlet algorithm advances the system by one timestep $\Delta t$
in three explicit steps:

$$\mathbf{r}(t + \Delta t) = \mathbf{r}(t) + \mathbf{v}(t)\,\Delta t + \tfrac{1}{2}\mathbf{a}(t)\,\Delta t^2$$

$$\mathbf{a}(t + \Delta t) = \mathbf{F}\!\left[\mathbf{r}(t+\Delta t)\right] / m$$

$$\mathbf{v}(t + \Delta t) = \mathbf{v}(t) + \tfrac{1}{2}\left[\mathbf{a}(t) + \mathbf{a}(t+\Delta t)\right]\Delta t$$

The velocity uses the **average** of the old and new acceleration. This
time-symmetric update is what makes the algorithm **symplectic** — it exactly
preserves the phase-space volume of Hamiltonian mechanics.

In code, using NumPy arrays of shape `(N, 2)`:

```python
pos   = pos + vel * dt + 0.5 * a * dt**2   # Eq 1
a_new = acceleration(pos)                   # Eq 2
vel   = vel + 0.5 * (a + a_new) * dt       # Eq 3
a     = a_new
```

**Pros:** Symplectic — energy oscillates around a constant with no long-term drift;
no scipy needed; fully transparent.  
**Cons:** Fixed time step; 2nd-order accuracy per step.

### Integrator Comparison

| Property | `odeint` (RK45) | Velocity Verlet |
|----------|----------------|-----------------|
| Algorithm | Runge-Kutta 4/5 | Symplectic 2nd-order |
| Symplectic | No | **Yes** |
| Energy behavior | Monotonic drift | Bounded oscillation |
| Step size | Adaptive | Fixed |
| Order | 4th–5th | 2nd |
| Long simulations | Unreliable | **Preferred** |
| Dependencies | numpy + scipy | numpy only |

---

## Energy Conservation (vectorized)

Both files compute the total energy at every time step without any Python loops,
using NumPy broadcasting over the full `(nsteps, N, N)` distance array:

```python
# Kinetic energy  ->  (nsteps,)
KE = 0.5 * np.sum(mass[None, :, None] * vel_all**2, axis=(1, 2))

# All pairwise distances  ->  (nsteps, N, N)
diff_all = pos_all[:, None, :, :] - pos_all[:, :, None, :]
dist_all = np.linalg.norm(diff_all, axis=3)
dist_all[:, np.arange(N), np.arange(N)] = 1.0

# Potential energy — upper triangle avoids double counting  ->  (nsteps,)
mass_ij = mass[:, None] * mass[None, :]
PE = -G * np.sum(np.triu(mass_ij, k=1)[None, :, :] / dist_all, axis=(1, 2))

E = KE + PE
```

`np.triu(mass_ij, k=1)` keeps only the pairs where $i < j$, ensuring each
pair is counted exactly once.

---

## Data Structures

Both files use the same array layout throughout:

```
pos_all.shape = (nsteps, N, 2)
vel_all.shape = (nsteps, N, 2)

Indexing:  pos_all[time_step, body_index, coordinate]
           coordinate: 0 = x,  1 = y
           body_index: 0 = Sun, 1 = Mercury, 2 = Venus,
                       3 = Earth, 4 = Mars, 5 = Jupiter
```

---

## Output

Each script produces three windows in sequence:

1. **Orbit plot** — trajectories of all 6 bodies over 12 years
2. **Energy conservation** — two panels:
   - Top: $E(t)$ vs time
   - Bottom: relative error $(E - E_0) / |E_0|$ vs time
3. **Animation** — dark-background real-time simulation with orbital trails

---

## Requirements

```
numpy
matplotlib
scipy          # required only for solar_system_nbody.py
```

Install with:

```bash
pip install numpy matplotlib scipy
```

---

## Usage

```bash
# odeint version
python solar_system_nbody.py

# Velocity Verlet version
python solar_system_vv.py
```

---

## Extending the Simulation

To add a planet (e.g. Saturn), add **one row** to each of these arrays and
nothing else changes:

```python
pos0   = np.array([..., [9.537, 0.0]])    # Saturn semi-major axis
vel0   = np.array([..., [0.0, np.sqrt(G / 9.537)]])
mass   = np.array([..., 2.858e-4])        # Saturn mass
names  = [..., "Saturn"]
colors = [..., "goldenrod"]
```

The integrator, energy check, and animation all adapt automatically because
they loop over `N = len(mass)`.

---

## References

- Verlet, L. (1967). *Computer "Experiments" on Classical Fluids.* Physical Review, 159(1), 98.
- Leimkuhler, B. & Reich, S. (2004). *Simulating Hamiltonian Dynamics.* Cambridge University Press.
- Hairer, E., Lubich, C. & Wanner, G. (2006). *Geometric Numerical Integration.* Springer.

---

*PHY 517 — Computational Physics · Central Michigan University*
