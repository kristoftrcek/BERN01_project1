import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

class Two_D_test:
    """
    Class containing the parameters and results of one 2D test.
    """
    def __init__(self, config=None):
        self.config = config
        # copy the parameters that wont change
        self.N = config.N
        self.rho = config.rho
        self.sigma_0 = config.sigma_0
        self.sigma_1 = config.sigma_1
        self.epsilon = config.epsilon
        self.T_target = config.T_target
        self.cap = config.cap
        self.L = config.L


    def make_step(self, i, max_step_size, T_accept):
        # 1. Calculate the particles energy contribution with its N-1 neighbours
        E_old = self.config.U_single(i)
        x_old, y_old = self.config.x[i], self.config.y[i]

        # 2. Propose move with sigma_0/2 size in the positive/negative direction
        self.config.x[i] = (x_old + np.random.uniform(-max_step_size, max_step_size)) % self.L # While ensuring periodic boundary conditions
        self.config.y[i] = (y_old + np.random.uniform(-max_step_size, max_step_size)) % self.L

        # 3. Calculate the particles energy contribution after the move.
        E_new = self.config.U_single(i)
        dE = E_new - E_old

        # 4. We accept the move according to the boltzman distirbution
        if dE > 0 and np.random.rand() >= np.exp(-dE / T_accept): # We use T_high to temporarly accept higher energy states thus we do not get stuck in the untangling.
            self.config.x[i], self.config.y[i] = x_old, y_old  # Reject move


    def pre_equilibrate(self, T_high=5.0, max_sweeps=300):
        """
        Untangles overlapping particles from a hot start using a capped potential.
        Returns:
            success (bool): True if all hard-core overlaps (r < sigma_0) were resolved.
            sweeps_taken (int): Number of actual sweeps required to resolve overlaps.
        """

        success = False
        max_step_size = self.sigma_0
        sweeps_taken = max_sweeps

        self.config.U_samples_preeq = np.zeros(max_step_size)

        for sweep in range(max_sweeps):
            # A single sweep attempts a trial move for every particle in the system.
            # Sequential sweeps guarantee uniform sampling across all particles, preventing
            # localized overlap bottlenecks that occur with purely random particle selection.
            for i in range(self.N):
                self.make_step(i, max_step_size, T_high)

            # Check if all hard-core overlaps are resolved across the box
            # after every sweep we run a U_full to check wheter the conflict caused by core overlaps have resolved.
            U_calc = self.config.U_full()
            self.config.U_samples_preeq[sweep] = U_calc
            if not np.isinf(U_calc):
                success = True
                sweeps_taken = sweep + 1
                break

        return success, sweeps_taken
    

    def run_simulation(self, n_iter=1000, n_samples=100, n_bins=100, draw=False):
        """
        Run the MCMC simulation, take samples and draw the final results
        """

        max_step_size = self.sigma_0
        n_iter_per_sample = n_iter // n_samples

        self.config.U_samples = np.zeros(n_samples)
        self.config.ard_samples = np.zeros((n_samples,n_bins))

        for i in range(n_samples):
            for j in range(n_iter_per_sample):
                # Pick a random particle and make a step.
                idx = np.random.randint(0, self.config.N)
                self.make_step(idx, max_step_size, self.config.T_target)

            # Save the energy and radial density after every sample
            self.config.U_samples[i] = self.config.U_full()
            # self.config.ard_samples[i], _ = self.config.average_radial_density(n_bins)

            if draw:
                self.config.visualize(title=f"Particle Setup after {i+1} samples, U={self.config.U_samples[i]:.2f}")




class Particle_config:
    """
    Class containing particle config info
    """
     
    def __init__(self, N=30, rho=0.227, L=None, sigma_0=1.0, sigma_1=2.5, epsilon=1.0, T_target=1.0, cap=np.inf):
        self.N = N
        self.rho = rho
        self.sigma_0 = sigma_0
        self.sigma_1 = sigma_1
        self.epsilon = epsilon
        self.T_target = T_target
        self.cap = cap
        if L is None:
            self.L = np.sqrt(N / rho)
        else:
            self.L = L
        self.x = None
        self.y = None
        self.U_samples_preeq = None
        self.U_samples = None


    def hot_start_config(self):
        """
        Create a hot start configuration
        """    
        self.x = np.random.rand(self.N) * self.L
        self.y = np.random.rand(self.N) * self.L

    
    def cold_start_ort_grid_config(self):
        """
        Create a cold start configuration of particles in an orthogonal grid
        """
        row_count = np.ceil(np.sqrt(self.N)).astype(int)
        col_count = np.ceil(self.N / row_count).astype(int)
        remainder = self.N - row_count * (col_count-1)

        # Particles are filled in column by column
        x_vals = np.linspace(0, self.L, col_count, endpoint=False) # do not include the last sample
        x_vals += self.L / (2 * col_count) # center the values
        particles_per_column = row_count * np.ones(col_count, dtype=int)
        particles_per_column[-1] = remainder
        self.x = np.repeat(x_vals, particles_per_column) # repeat x value for the number of particles in each column
        
        y_vals = np.linspace(0, self.L, row_count, endpoint=False)
        y_vals += self.L / (2 * row_count) # center the values
        y_vals_mat = np.repeat(y_vals[np.newaxis,:], col_count-1, axis=0)
        self.y = np.concat([np.reshape(y_vals_mat, np.size(y_vals_mat)), y_vals[:remainder]]) # repeat y values array for each column, add remaining particles


    def vectorized_potential(self, r):
        """
        Vectorized pair potential function over array of distances r.
        """
        phi = np.zeros_like(r)
        phi[(r >= self.sigma_0) & (r < self.sigma_1)] = self.epsilon
        phi[r < self.sigma_0] = self.cap
        return phi


    def U_full(self):
        """
        Calculates the total potential energy of the whole system from the x and y coordinates with a cap value.
        """
        if self.x is None or self.y is None:
            raise ValueError("Coordinates not set yet. First run hot start or cold start.")
        dx = self.x[:, np.newaxis] - self.x[np.newaxis, :]
        dy = self.y[:, np.newaxis] - self.y[np.newaxis, :]

        dx -= self.L * np.round(dx / self.L)  # Update the distances according to the periodic boundary conditions.
        dy -= self.L * np.round(dy / self.L)

        euc_distance = dx**2 + dy**2

        i, j = np.triu_indices(len(self.x), k=1) # extract upper triangular matrix without the diagonal
        r = euc_distance[i, j]
        r_valid = np.sqrt(r[r <= self.sigma_1**2]) # extract distances inside sigma_1 radius

        if len(r_valid) == 0:
            return 0.0

        pair_potentials = self.vectorized_potential(r_valid)
        return np.sum(pair_potentials)


    def U_single(self, i):
        """
        Calculates potential of a single particle with idx i with respect to other particles. Coordinates of the particles are given by x and y.
        """
        mask = np.arange(self.N) != i # We filter out the particle we have moved
        dx = self.x[i] - self.x[mask] # Calculate the distance of its coordinate compared to every other particles
        dy = self.y[i] - self.y[mask]

        dx -= self.L * np.round(dx / self.L) # Apply periodic boundary
        dy -= self.L * np.round(dy / self.L)

        r = dx**2 + dy**2 # Calculate the euclidean distances
        r_valid = np.sqrt(r[r <= self.sigma_1**2]) # distances inside sigma_1 radius
        
        if len(r_valid) == 0:
            return 0.0
        
        phi = self.vectorized_potential(r_valid) # Finally calculate the energy of this particular particle layout
        return np.sum(phi)


    def average_radial_density(self, n_bins=100):
        """
        Computes the histogram of distances between particles and computes the average radial density.
        """
        dx = self.x[:, np.newaxis] - self.x[np.newaxis, :]
        dy = self.y[:, np.newaxis] - self.y[np.newaxis, :]
        dx -= self.L * np.round(dx / self.L)  # Update the distances according to the periodic boundary conditions.
        dy -= self.L * np.round(dy / self.L)

        r = np.reshape(np.sqrt(dx**2 + dy**2), self.N**2)
        r = r[r > 0] # discard the pairs of the same particle

        # Make a histogram that counts distances between particles and normalize it
        [g, bin_edges] = np.histogram(r, n_bins, range=(0, self.L / 2)) # Only count particles in radius L/2
        dr = bin_edges[1] - bin_edges[0] # bins of the same size
        r_vals = bin_edges[:-1] + (dr) / 2 # avg radii of the bins
        g_weights = 2 * np.pi * r_vals * dr # areas of circular strips
        n_b = self.N / self.L**2
        g_normalized = g / (n_b * g_weights)
        return g_normalized, r_vals


    def visualize(self, title="Particle Setup"):
        """
        Visualize particles with their radii.
        """

        R_core = self.sigma_0 / 2
        R_shoulder = self.sigma_1 / 2

        fig, ax = plt.subplots(figsize=(7, 7))
        for xi, yi in zip(self.x, self.y):
            # 1. Draw outer soft shoulder
            shoulder_circle = plt.Circle(
                (xi, yi),
                radius=R_shoulder,
                color="blue",
                alpha=0.15,
                ec="blue",
                ls="--",
            )
            ax.add_patch(shoulder_circle)

            # 2. Draw inner hard core
            circle = plt.Circle(
                (xi, yi), radius=R_core, color="red", alpha=0.5, ec="black"
            )
            ax.add_patch(circle)

        ax.set_aspect("equal")
        ax.set_xlim(0, self.L)
        ax.set_ylim(0, self.L)
        plt.xticks(np.arange(0, self.L + 1, step=5))
        plt.yticks(np.arange(0, self.L + 1, step=5))
        plt.title(title)
        plt.grid(True)
        plt.show()
