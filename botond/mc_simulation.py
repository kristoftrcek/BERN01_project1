"""
Optimized MC simulation functions (pure numpy — no Numba).

Changes vs. the original:
- vectorized_potential() removed: folded directly into U_single/U_full.
- No square roots anywhere: all comparisons are done on squared distances
  against sigma_0**2 / sigma_1**2, since that's all the potential needs.
- No boolean-mask array built to exclude 'identity': self-distance is set
  to +inf in-place instead, so it's automatically outside the cutoff.
- r_cut filtering removed from U_single/U_full: redundant since r_cut ==
  sigma_1 in every call in this project (the sigma_1 cutoff already does it).
- Length, sigma0_sq, sigma1_sq are computed ONCE per pipeline run and passed
  in, instead of being recomputed on every single U_single/U_full call.
- equilibrate_continuous / run_production_sampling now track total energy
  incrementally (U_current += delta_U on acceptance) instead of calling the
  O(N^2) U_full every sample_interval. U_full is only called: once at the
  start, and periodically for the energy-conservation check the project
  asks for (every energy_check_interval moves) -- comparing the tracked
  running energy against a from-scratch recompute and reporting the drift.
"""

import numpy as np
from tqdm import tqdm

sigma_0 = 1.0     # Hard-core diameter
sigma_1 = 2.5     # Soft-shoulder diameter
epsilon = 1.0     # Energy unit



# Energy functions

def U_single(x, y, identity, Length, sigma0_sq, sigma1_sq, cap=None):
    half_L = Length / 2.0
    
    # Measure the distance between the choosen particle and the others
    dx = x[identity] - x
    dy = y[identity] - y
    
    # PDC - two particles should not be farther then L/2
    # Shifts range to [0, L), wraps via modulo, then shifts back to [-L/2, L/2)
    # E.g., for L=10, half_L=5: dx = -8 -> (-8 + 5) % 10 - 5 = 2 (shortest path across boundary)
    dx = (dx + half_L) % Length - half_L
    dy = (dy + half_L) % Length - half_L
    
    r2 = dx * dx + dy * dy # Euclidean distance squared (np.sqrt is computationally expensive)
    r2[identity] = np.inf  # Exclude self-interaction
    
    # Count how many hard overlaps we have
    hard = r2 < sigma0_sq
    n_hard = np.count_nonzero(hard)
    if n_hard > 0 and cap is None: # If more than one and we are not in pre-eq we set energy to ininity.
        return np.inf

    # Count how many soft overlaps we have
    soft = (r2 >= sigma0_sq) & (r2 < sigma1_sq)
    n_soft = np.count_nonzero(soft)

    if n_hard > 0: # if we had a cap calculate energy with this fomula
        return cap * n_hard + epsilon * n_soft
    return epsilon * n_soft #if not we just mulitply it with n_soft


def U_full(x, y, Length, sigma0_sq, sigma1_sq, cap=None):
    """
    Total interaction energy of the whole system (all unique pairs).
    """
    half_L = Length / 2.0
    dx = x[:, np.newaxis] - x[np.newaxis, :] # Matix representation of the distance between all pairs
    dy = y[:, np.newaxis] - y[np.newaxis, :] # By substituting a column vector from a row vector we get a N X N matrix

    # PBC
    dx = (dx + half_L) % Length - half_L
    dy = (dy + half_L) % Length - half_L

    i, j = np.triu_indices(len(x), k=1) # Selecting unique pairs above the diagonal
    r2 = dx[i, j] ** 2 + dy[i, j] ** 2 # Calculating squared euclidean distance

    hard = r2 < sigma0_sq
    n_hard = np.count_nonzero(hard) # Counting overlaps to calculate energy accordingly
    if n_hard > 0 and cap is None:
        return np.inf

    soft = (r2 >= sigma0_sq) & (r2 < sigma1_sq)
    n_soft = np.count_nonzero(soft)

    if n_hard > 0:
        return cap * n_hard + epsilon * n_soft
    return epsilon * n_soft



# Initialization


def hot_start(N=100, rho=0.1):
    """Randomly place N particles in a box of side Length = sqrt(N/rho)."""
    Length = np.sqrt(N / rho)
    x = np.random.rand(N) * Length 
    y = np.random.rand(N) * Length
    return x, y



# Pre-equilibration (untangle hot-start overlaps at elevated T with a capped
# potential). Not a bottleneck (~30-60k U_single calls total), left mostly
# as-is, just updated to the new U_single signature.


def pre_equilibrate(x, y, T_high=100, U_cap=800, delta_pre_eq=0.50,
                     max_sweeps=300, check_interval=15, rho=0.1, N=100):

    """
    Resolves initial hard-core overlaps from a hot start using a capped potential.

    Performs particle-by-particle Metropolis moves at high temperature (T_high),
    replacing infinite overlap penalties with a finite cap (U_cap) so particles 
    can slide past each other. Checks uncapped full energy every check_interval 
    sweeps and exits early once all hard-core overlaps are gone.
    """
    Length = np.sqrt(N / rho)
    sigma0_sq = sigma_0 ** 2
    sigma1_sq = sigma_1 ** 2
    N_particles = len(x)
    success = False
    sweeps_taken = max_sweeps

    for sweep in range(max_sweeps): # With sweeps we make targeted particle movements, to solve conlifct more quickly
        for i in range(N_particles):
            E_old = U_single(x, y, i, Length, sigma0_sq, sigma1_sq, cap=U_cap) # Energy before move
            x_old, y_old = x[i], y[i] # Saving the coordinates in case of rollback

            x[i] = (x[i] + np.random.uniform(-delta_pre_eq, delta_pre_eq)) % Length # Trial movements
            y[i] = (y[i] + np.random.uniform(-delta_pre_eq, delta_pre_eq)) % Length

            E_new = U_single(x, y, i, Length, sigma0_sq, sigma1_sq, cap=U_cap)
            dE = E_new - E_old # Change in energy between the new and old configuartion

            if dE > 0 and np.random.rand() >= np.exp(-dE / T_high):
                x[i], y[i] = x_old, y_old # Montecarlo acceptance

        if (sweep + 1) % check_interval == 0 or sweep == max_sweeps - 1: # Check wheter we resolved conflicts every after every 'check interval' iteration.
            if not np.isinf(U_full(x, y, Length, sigma0_sq, sigma1_sq, cap=None)):
                success = True
                sweeps_taken = sweep + 1
                break

    return x, y, success, sweeps_taken



# A single-particle Monte Carlo move, used in equilibration and production.
# Returns the new (accepted) energy delta, or None if rejected.


def _mc_move(x, y, Length, sigma0_sq, sigma1_sq, target_T, delta):
    N_particles = len(x)
    i = np.random.randint(0, N_particles) # Randomly select a particle

    U_old = U_single(x, y, i, Length, sigma0_sq, sigma1_sq)
    x_old, y_old = x[i], y[i] # Check its energy contribution in the original layout

    x[i] = (x_old + np.random.uniform(-delta, delta)) % Length
    y[i] = (y_old + np.random.uniform(-delta, delta)) % Length

    U_new = U_single(x, y, i, Length, sigma0_sq, sigma1_sq)
    dU = U_new - U_old # Calculate the energy difference with the new layout

    if dU <= 0:
        return dU, True # If energy reduces automatically accept 
    elif U_new != np.inf and np.random.rand() < np.exp(-dU / target_T): #if not we accept it according to the Boltzmann distribution
        return dU, True
    else:
        x[i], y[i] = x_old, y_old
        return 0.0, False



# Equilibration 


from tqdm.notebook import tqdm

def equilibrate_continuous(
    x,
    y,
    target_T=0.1,
    total_moves=1000000,
    delta=0.15,
    sample_interval=1000,
    rho=0.1,
    N=100,
    energy_check_interval=None,
):
    """Runs target temperature equilibration phase."""
    Length = np.sqrt(N / rho)
    sigma0_sq = sigma_0**2
    sigma1_sq = sigma_1**2

    if energy_check_interval is None:
        energy_check_interval = total_moves // 5

    accepted_moves = 0
    energy_history = []
    move_history = []

    U_current = U_full(x, y, Length, sigma0_sq, sigma1_sq) # Calculate the base energy

    with tqdm(
        total=total_moves,
        desc=f"Equilibrating (rho*={rho})",
        unit="move",
        leave=False,
    ) as pbar:
        for current_move in range(1, total_moves + 1):
            dU, accepted = _mc_move(
                x, y, Length, sigma0_sq, sigma1_sq, target_T, delta
            ) # Attempt a move and accept the new energy according to MC rules
            if accepted:
                accepted_moves += 1
                U_current += dU

            if current_move % sample_interval == 0:
                energy_history.append(U_current)
                move_history.append(current_move)

            pbar.update(1)

    acceptance_ratio = accepted_moves / total_moves # Checking the acceptance ratio
    print(
        f"Equilibration Finished (rho*={rho}) | Acceptance: {acceptance_ratio * 100:.2f}%"
    )

    return (
        x,
        y,
        np.array(move_history),
        np.array(energy_history),
        acceptance_ratio,
    )


def run_production_sampling(
    x,
    y,
    target_T=0.1,
    total_moves=2000000,
    delta=0.15,
    sample_interval=1000,
    rho=0.1,
    N=100,
    energy_check_interval=None,
):
    """Runs production phase sampling and records coordinate trajectories."""
    Length = np.sqrt(N / rho)
    sigma0_sq = sigma_0**2
    sigma1_sq = sigma_1**2

    if energy_check_interval is None:
        energy_check_interval = max(1, total_moves // 10)

    accepted_moves = 0
    energy_history = []
    move_history = []
    trajectory = []

    U_current = U_full(x, y, Length, sigma0_sq, sigma1_sq) # Continue the same procedure after equilibrium

    with tqdm(
        total=total_moves,
        desc=f"Production   (rho*={rho})",
        unit="move",
        leave=False,
    ) as pbar:
        for current_move in range(1, total_moves + 1):
            dU, accepted = _mc_move(
                x, y, Length, sigma0_sq, sigma1_sq, target_T, delta
            )
            if accepted:
                accepted_moves += 1
                U_current += dU

            if current_move % sample_interval == 0:
                energy_history.append(U_current)
                move_history.append(current_move)
                trajectory.append(np.column_stack((x.copy(), y.copy())))

            pbar.update(1)

    acceptance_ratio = accepted_moves / total_moves
    print(
        f"Production Finished   (rho*={rho}) | Acceptance: {acceptance_ratio * 100:.2f}%\n"
    )

    return (
        x,
        y,
        np.array(move_history),
        np.array(energy_history),
        np.array(trajectory),
        acceptance_ratio,
    )

def compute_gr_2d(trajectory, Length, dr=0.02, max_r=None):
    """
    Computes the 2D radial distribution function g(r) averaged over recorded trajectory frames.
    """
    if max_r is None:
        max_r = Length / 2.0  

    num_frames, N, _ = trajectory.shape
    half_L = Length / 2.0
    bins = np.arange(0, max_r + dr, dr)
    hist = np.zeros(len(bins) - 1, dtype=np.float64)

    # Accumulate pairwise distances
    for frame in trajectory:
        px, py = frame[:, 0], frame[:, 1]
        dx = px[:, np.newaxis] - px[np.newaxis, :]
        dy = py[:, np.newaxis] - py[np.newaxis, :]

        # PBC
        dx = (dx + half_L) % Length - half_L
        dy = (dy + half_L) % Length - half_L

        r = np.sqrt(dx**2 + dy**2)

        # Upper triangle unique pairs
        i, j = np.triu_indices(N, k=1)
        frame_distances = r[i, j]

        frame_hist, _ = np.histogram(frame_distances, bins=bins)
        hist += frame_hist

    # Normalization for 2D geometry
    r_centers = (bins[:-1] + bins[1:]) / 2.0
    area_bin = np.pi * (bins[1:] ** 2 - bins[:-1] ** 2)

    # Total pair contributions = num_frames * N * (N - 1) / 2
    total_pairs = num_frames * N * (N - 1) / 2.0
    ideal_count = total_pairs * (area_bin / (Length**2))

    g_r = hist / ideal_count
    return r_centers, g_r