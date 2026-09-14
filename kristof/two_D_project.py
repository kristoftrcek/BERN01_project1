import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

class Two_D_test:
    """
    Class containing the parameters and results of one 2D test.
    """
    def __init__(self, config=None):
        self.config = config
        # copy the parameters that won't change
        self.N = config.N
        self.rho = config.rho
        self.sigma_0 = config.sigma_0
        self.sigma_1 = config.sigma_1
        self.epsilon = config.epsilon
        self.T_target = config.T_target
        self.cap = config.cap   # cap value for pre-equilibration
        self.L = config.L


    def make_step(self, i, max_step_size, T_accept, cap=None):
        if cap is None: # cap is given only when we want infinite cap
            cap = self.cap # if not given, set cap to finite preset cap value (only in pre-eq run)

        # 1. Calculate the particles energy contribution with its N-1 neighbours
        E_old = self.config.U_single(i, cap)
        x_old, y_old = self.config.x[i], self.config.y[i]

        # 2. Propose move with sigma_0/2 size in the positive/negative direction
        self.config.x[i] = (x_old + np.random.uniform(-max_step_size, max_step_size)) % self.L # While ensuring periodic boundary conditions
        self.config.y[i] = (y_old + np.random.uniform(-max_step_size, max_step_size)) % self.L

        # 3. Calculate the particles energy contribution after the move.
        E_new = self.config.U_single(i, cap)
        dE = E_new - E_old

        # 4. We accept the move according to the boltzman distirbution
        if np.isinf(E_new) or (dE > 0 and np.random.rand() >= np.exp(-dE / T_accept)):
            self.config.x[i], self.config.y[i] = x_old, y_old  # Reject move
            return False, 0.0
        return True, dE
    

    def pre_equilibrate(self, T_high=100.0, max_sweeps=300, check_interval=15):
        """
        Untangles overlapping particles of a givenconfiguration from a hot start using a capped potential.
        Parameters:
            T_high=100.0 (float): High temperature for pre-equilibration.
            max_sweeps=300 (int): Maximum number of sweeps to attempt.
            check_interval=15 (int): Interval at which to check for overlaps.
        Returns:
            success (bool): True if all hard-core overlaps (r < sigma_0) were resolved.
            sweeps_taken (int): Number of actual sweeps required to resolve overlaps.
        """
        if self.config.x is None or self.config.y is None:
            raise ValueError("Coordinates not set yet. First run hot start or cold start.")

        success = False
        max_step_size = self.config.max_step_size
        sweeps_taken = max_sweeps

        self.config.U_samples_preeq = np.zeros(max_sweeps)

        for sweep in tqdm(range(max_sweeps), desc="Pre-equilibration"):
            for i in range(self.N):
                # make a random step for each particle
                self.make_step(i, max_step_size, T_high)

            if sweep % check_interval == check_interval - 1:
                # Check if all hard-core overlaps are resolved across the box
                # after every 15 sweeps we run a U_full to check wheter the conflict caused by core overlaps have resolved.
                U_calc = self.config.U_full(cap=np.inf) # use infinite cap to check the actual energy
                self.config.U_samples_preeq[sweep] = U_calc

                if not np.isinf(U_calc):
                    success = True
                    sweeps_taken = sweep + 1
                    break

        return success, sweeps_taken
    

    def run_simulation(self, n_iter=1000000, sample_interval=1000, draw=False, new_start=True):
        """
        Run the MCMC simulation, take samples and draw the final results
        Parameters:
            n_iter=1000000 (int): number of total iterations
            sample_intervals=1000 (int): amount of iterations for one sample
            draw=False (bool): if True, on every sample we draw the particle configuration
            new_start=True (bool): if True, the saved U sample array will be discarded. If False, new U samples will be appended to an existing array
        Return:
            acceptance_rate (float)
            accumulation_err (float): U_accumulated - U_full
        """
        if self.config.x is None or self.config.y is None:
            raise ValueError("Coordinates not set yet. First run hot start or cold start.")

        max_step_size = self.config.max_step_size
        n_samples = n_iter // sample_interval

        # if sample array already exists, continue from the last sample, unless new_start=True
        if self.config.U_samples is None or new_start:
            start = 0
            self.config.U_samples = np.zeros(n_samples)
            # self.config.ard_samples = np.zeros((n_samples,n_bins))
            self.config.x_hist = np.zeros((n_samples, len(self.config.x)))
            self.config.y_hist = np.zeros((n_samples, len(self.config.y)))
        else:
            start = len(self.config.U_samples)
            self.config.U_samples = np.concatenate([self.config.U_samples, np.zeros(n_samples)])
            # self.config.ard_samples = np.concatenate([self.config.ard_samples, np.zeros(n_samples, n_bins)])
            self.config.x_hist = np.concatenate([self.config.x_hist, np.zeros(n_samples, len(self.config.x))])
            self.config.y_hist = np.concatenate([self.config.y_hist, np.zeros(n_samples, len(self.config.y))])

        U_accumulated = self.config.U_full(cap=np.inf)
        accepted = 0
        for i in tqdm(range(n_iter), desc="Running Simulation"):
            # Pick a random particle and make a step.
            idx = np.random.randint(0, self.config.N)
            step_accepted, dU = self.make_step(idx, max_step_size, self.config.T_target, cap=np.inf)
            if step_accepted:
                accepted += 1
                U_accumulated += dU

            if i % sample_interval == sample_interval-1:
                # Save the energy and radial density after every sample
                save_idx = start + i//sample_interval
                self.config.U_samples[save_idx] = self.config.U_full(cap=np.inf)
                # self.config.ard_samples[save_idx, :], _ = self.config.average_radial_density(n_bins)
                self.config.x_hist[save_idx, :] = self.config.x
                self.config.y_hist[save_idx, :] = self.config.y

                if draw:
                    self.config.visualize(title=f"Particle Setup after {i+1} samples, U={self.config.U_samples[i]:.2f}")

        acceptance_rate = accepted / n_iter
        accumulation_err = U_accumulated - self.config.U_full(cap=np.inf)
        
        return acceptance_rate, accumulation_err




class Particle_config:
    """
    Class containing particle config info
    """
     
    def __init__(self, N=100, rho=0.227, L=None, sigma_0=1.0, sigma_1=2.5, epsilon=1.0, T_target=0.1, cap=np.inf, max_step_size=1.0):
        """
        Parameters:
            N=100 (int): number of particles
            rho=0.227 (float): particle density
            L=None (float): length of the box, if None the length is calculated from N and rho
            sigma_0=1.0 (float): hard core diameter
            sigma_1=2.5 (float): soft core diameter
            epsilon=1.0 (float): potential if soft cores overlap
            T_target=0.1 (float): temperature of the simulation
            cap=np.inf (float): energy cap during pre-equilibrium step
            max_step_size=1.0 (float): maximal step in x and y direction
        """
        self.N = N
        self.rho = rho
        self.sigma_0 = sigma_0
        self.sigma_1 = sigma_1
        self.sigma_0_sq = sigma_0 * sigma_0
        self.sigma_1_sq = sigma_1 * sigma_1
        self.epsilon = epsilon
        self.T_target = T_target
        self.cap = cap  # cap value for pre-equilibration
        self.max_step_size = max_step_size
        if L is None:
            self.L = np.sqrt(N / rho)
        else:
            self.L = L
        self.x = None
        self.y = None
        self.x_hist = None
        self.y_hist = None
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

    
    def cold_start_hex_grid_config(self):
        """
        Create a cold start configuration of particles in an hexagonal grid
        """
        col_count = np.ceil(np.sqrt(2*self.N / np.sqrt(3))).astype(int)
        row_count = np.ceil(self.N / col_count).astype(int)
        remainder = self.N - col_count * (row_count-1)

        # Particles are filled in row by row
        x_vals = np.linspace(0, self.L, col_count, endpoint=False) # do not include the last sample
        x_vals_mat = np.repeat(x_vals[np.newaxis,:], row_count-1, axis=0)
        x_vals_mat[1::2, :] += self.L / (2 * col_count) # center each second row
        last_row = x_vals[:remainder] + self.L * ((row_count-1) % 2) / (2 * col_count) # center the last row if it is an even row
        self.x = np.concat([np.reshape(x_vals_mat, np.size(x_vals_mat)), last_row]) # repeat x values array for each row, add remaining particles
        
        y_vals = np.linspace(0, self.L, row_count, endpoint=False)
        y_vals += self.L / (2 * row_count) # center the values
        particles_per_row = col_count * np.ones(row_count, dtype=int)
        particles_per_row[-1] = remainder
        self.y = np.repeat(y_vals, particles_per_row) # repeat y value for the number of particles in each row


    def vectorized_potential(self, r):
        """
        Vectorized pair potential function over array of distances r.
        """
        phi = np.zeros_like(r)
        phi[(r >= self.sigma_0) & (r < self.sigma_1)] = self.epsilon
        phi[r < self.sigma_0] = self.cap
        return phi


    def U_full(self, cap=None):
        """
        Calculates the total potential energy of the whole system from the x and y coordinates
        with a preset cap value or given cap value.
        """
        if self.x is None or self.y is None:
            raise ValueError("Coordinates not set yet. First run hot start or cold start.")

        if cap is None: # cap is given only when we want infinite cap
            cap = self.cap # if not given, set cap to finite preset cap value (only in pre-eq run)
        

        dx = self.x[:, np.newaxis] - self.x[np.newaxis, :]
        dy = self.y[:, np.newaxis] - self.y[np.newaxis, :]

        half_L = self.L / 2
        dx = (dx + half_L) % self.L - half_L # Apply periodic boundary
        dy = (dy + half_L) % self.L - half_L

        i, j = np.triu_indices(len(self.x), k=1) # extract upper triangular matrix without the diagonal
        r = dx[i, j]**2 + dy[i, j]**2

        n_hard = np.count_nonzero(r < self.sigma_0_sq) # Counting overlaps to calculate energy accordingly
        if n_hard > 0 and np.isinf(cap):
            return np.inf

        n_soft = np.count_nonzero((r >= self.sigma_0_sq) & (r < self.sigma_1_sq))

        if n_hard > 0:
            return cap * n_hard + self.epsilon * n_soft
        return self.epsilon * n_soft



    def U_single(self, i, cap=None):
        """
        Calculates potential of a single particle with idx i with respect to other particles. Coordinates of the particles are given by x and y.
        """
        if self.x is None or self.y is None:
            raise ValueError("Coordinates not set yet. First run hot start or cold start.")
        
        if cap is None: # cap is given only when we want infinite cap
            cap = self.cap # if not given, set cap to finite preset cap value (only in pre-eq run)
        
        dx = self.x[i] - self.x # Calculate the distance of its coordinate compared to every other particles
        dy = self.y[i] - self.y

        half_L = self.L / 2
        dx = (dx + half_L) % self.L - half_L # Apply periodic boundary
        dy = (dy + half_L) % self.L - half_L

        r = dx*dx + dy*dy # Calculate the square of euclidean distances
        r[i] = np.inf  # Exclude self-interaction

        n_hard = np.count_nonzero(r < self.sigma_0_sq) # Count the number of hard-core overlaps
        if n_hard > 0 and np.isinf(cap): # If more than one and we are not in pre-eq we set energy to ininity.
                return np.inf

        n_soft = np.count_nonzero((r >= self.sigma_0_sq) & (r < self.sigma_1_sq)) # Count the number of soft-core overlaps
    
        if n_hard > 0: # if we had a cap calculate energy with this formula
            return cap * n_hard + self.epsilon * n_soft
        return self.epsilon * n_soft


    def average_radial_density(self, n_bins=300):
        """
        Computes the histogram of distances between particles and computes the average radial density over all the time steps.
        """
        if self.x_hist is None or self.y_hist is None:
            raise ValueError("Coordinate histories not set yet. First run the simulation.")

        n_samples, _ = np.shape(self.x_hist)
        g = np.zeros(n_bins)
        for i in range(n_samples):
            x = self.x_hist[i,:]
            y = self.y_hist[i,:]
            dx = x[:, np.newaxis] - x[np.newaxis, :]
            dy = y[:, np.newaxis] - y[np.newaxis, :]

            half_L = self.L / 2
            dx = (dx + half_L) % self.L - half_L # Apply periodic boundary
            dy = (dy + half_L) % self.L - half_L

            r = np.reshape(np.sqrt(dx**2 + dy**2), self.N**2)
            r = r[r > 0] # discard the pairs of the same particle

            # Make a histogram that counts distances between particles and normalize it
            [histogram, bin_edges] = np.histogram(r, n_bins, range=(0, self.L / 2)) # Only count particles in radius L/2
            g += histogram

        dr = bin_edges[1] - bin_edges[0] # bins of the same size
        r_vals = bin_edges[:-1] + (dr) / 2 # avg radii of the bins
        strip_areas = np.pi * (bin_edges[1:]**2 + bin_edges[:-1]**2)
        n_b = self.N / self.L**2

        g_normalized = g / (self.N * n_b * strip_areas * n_samples)
        return r_vals, g_normalized


    def get_parameters(self):
        """
        Return:
            self.x (1d array): current x coordinates of positions
            self.y (1d array): current y coordinates of positions
            self.x_hist (2d array): x coordinates history
            self.y_hist (2d array): y coordinates history
            self.U_samples (1d array): energy history
        """
        return self.x, self.y, self.x_hist, self.y_hist, self.U_samples


    def get_parameters_all(self):
        """
        Return:
            self.N (int): number of particles
            self.rho (float): particle density
            self.L (float): length of the box
            self.sigma_0 (float): hard core diameter
            self.sigma_1 (float): soft core diameter
            self.epsilon (float): potential if soft cores overlap
            self.T_target (float): temperature of the simulation
            self.cap (float): energy cap during pre-equilibrium step
            self.max_step_size (float): maximal step in x and y direction
            self.x (1d array): current x coordinates of positions
            self.y (1d array): current y coordinates of positions
            self.x_hist (2d array): x coordinates history
            self.y_hist (2d array): y coordinates history
            self.U_samples (1d array): energy history
        """
        return self.N, self.rho, self.L, self.sigma_0, self.sigma_1, self.epsilon, self.T_target, self.cap, self.max_step_size, self.x, self.y, self.x_hist, self.y_hist, self.U_samples


    def visualize_2d(self, title="Particle Setup"):
        """
        Visualize particles with their radii.
        """

        R_core = self.sigma_0 / 2
        R_shoulder = self.sigma_1 / 2

        fig, ax = plt.subplots(figsize=(5, 5))
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

    def visualise_U(self, title='Energy throughout the iterations'):
        """
        Plot U samples
        """

        fig, ax = plt.subplots(figsize=(7, 7))
        
        ax.plot(self.U_samples)
        plt.xlabel("number of iterations")
        plt.ylabel("U")
        plt.title(title)
        plt.grid(True)
        plt.show()


