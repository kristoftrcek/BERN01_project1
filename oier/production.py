import numpy as np
import pickle as pkl
from joblib import Parallel, delayed
from two_D_project import Particle_config, Two_D_test

#   Parameters
N = 100                 
sigma_0 = 1.0
sigma_1 = 2.5
epsilon = 1.0
cap = 800
eq_moves = 2_000_000    
prod_moves = 20_000_000 
sample_interval = 10_000

def save_pkl(obj, filename):
    with open(filename, 'wb') as outp:
        pkl.dump(obj, outp, pkl.HIGHEST_PROTOCOL)


# PART A: DENSITY

def run_density_sweep(rho, max_step):
    print(f"Starting Density rho*={rho}")
    L = np.sqrt(N / rho)
    pc = Particle_config(N=N, rho=rho, L=L, sigma_0=sigma_0, sigma_1=sigma_1,
                         epsilon=epsilon, T_target=0.1, cap=cap, max_step_size=max_step)
    
    pc.hot_start_config()
    sim = Two_D_test(pc)
    
    # Pre-equilibration
    sim.pre_equilibrate(T_high=100.0, max_sweeps=300, check_interval=15)
    
    # Equilibration
    sim.run_simulation(n_iter=eq_moves, sample_interval=sample_interval, draw=False, new_start=True)
    
    # Production
    acc_rate, u_err = sim.run_simulation(n_iter=prod_moves, sample_interval=sample_interval, draw=False, new_start=True)
    
    # Save results
    save_pkl({'config': pc, 'acceptance_rate': acc_rate, 'energy_drift': u_err}, f"results_rho_{rho}.pkl")
    print(f"Finished rho*={rho} | Acc Rate: {acc_rate:.3f} | U Drift: {u_err}")
    return rho

def run_all_densities():
    rhos = [0.10, 0.15, 0.227, 0.291, 0.380] #
    max_step_sizes = [1.1, 0.338, 0.200, 0.146, 0.107]
    
    print("--- PART A: DENSITY  ---")
    Parallel(n_jobs=len(rhos))(delayed(run_density_sweep)(rhos[i], max_step_sizes[i]) for i in range(len(rhos)))

# PART B: TEMPERATURE 

def run_temperature_ramp():
    print("\n--- PART B: TEMPERATURE  ---")
    rho_ramp = 0.291 #
    L = np.sqrt(N / rho_ramp)
    max_step = 0.146 
    temps = [0.25, 0.24, 0.23, 0.22, 0.21, 0.20, 0.19, 0.18, 0.17, 0.16, 0.15] #
    
    # Config
    pc = Particle_config(N=N, rho=rho_ramp, L=L, sigma_0=sigma_0, sigma_1=sigma_1,
                         epsilon=epsilon, T_target=temps[0], cap=cap, max_step_size=max_step)
    pc.hot_start_config()
    sim = Two_D_test(pc)
    
    # Pre-equilibrate once
    sim.pre_equilibrate(T_high=100.0, max_sweeps=300, check_interval=15)
    
    for T in temps:
        print(f"Running Temperature T*={T}")
        pc.T_target = T
        
        # Equilibrate at new temperature
        # The final coordinates of the previous step are used as the starting point for the next temperature
        sim.run_simulation(n_iter=eq_moves, sample_interval=sample_interval, draw=False, new_start=True)
        
        # Produce 
        acc_rate, u_err = sim.run_simulation(n_iter=prod_moves, sample_interval=sample_interval, draw=False, new_start=True)
        
        # Save results 
        save_pkl({'config': pc, 'acceptance_rate': acc_rate, 'energy_drift': u_err}, f"results_ramp_T_{T}.pkl")

if __name__ == '__main__':
    run_all_densities()
    run_temperature_ramp()