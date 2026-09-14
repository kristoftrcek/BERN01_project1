import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mc_simulation as mc
import pickle



# 1. VG Parameters
N = 1000 
T_star = 0.1 
eq_moves = 2_000 
prod_moves = 1_000_000 
sample_interval = 500 
densities = [0.10, 0.15, 0.227, 0.291, 0.380] 

# Pre-calculated by Botond
optimal_deltas = {0.1: 1.116, 0.15: 0.449, 0.227: 0.284, 0.291: 0.224, 0.38: 0.171}

def save_snapshot(x, y, Length, title, filename):
    
    R_core, R_shoulder = mc.sigma_0 / 2.0, mc.sigma_1 / 2.0
    fig, ax = plt.subplots(figsize=(7, 7))
    for xi, yi in zip(x, y):
        ax.add_patch(plt.Circle((xi, yi), radius=R_shoulder, color="blue", alpha=0.15, ec="blue", ls="--"))
        ax.add_patch(plt.Circle((xi, yi), radius=R_core, color="red", alpha=0.5, ec="black"))
    ax.set_aspect("equal")
    ax.set_xlim(0, Length)
    ax.set_ylim(0, Length)
    plt.title(title)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def create_animation(trajectory, Length, filename):
    
    R_core, R_shoulder = mc.sigma_0 / 2.0, mc.sigma_1 / 2.0
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, Length)
    ax.set_ylim(0, Length)
    ax.set_aspect("equal")
    
    patches = []
    for _ in range(N):
        shoulder = plt.Circle((0, 0), radius=R_shoulder, color="blue", alpha=0.15, ec="blue", ls="--")
        core = plt.Circle((0, 0), radius=R_core, color="red", alpha=0.5, ec="black")
        ax.add_patch(shoulder)
        ax.add_patch(core)
        patches.extend([shoulder, core])
        
    def update(frame):
        coords = trajectory[frame]
        for i, (xi, yi) in enumerate(coords):
            patches[i*2].center = (xi, yi)    
            patches[i*2+1].center = (xi, yi)  
        return patches

    ani = animation.FuncAnimation(fig, update, frames=len(trajectory), blit=True)
    ani.save(filename, writer='ffmpeg', fps=30)
    plt.close()

# 2. Loop
for rho in densities:
    print(f"\n--- Running Density rho* = {rho} ---")
    Length = np.sqrt(N / rho)
    delta = optimal_deltas[rho]
    
    x, y = mc.hot_start(N=N, rho=rho)
    x, y, _, _ = mc.pre_equilibrate(x, y, T_high=100, U_cap=800, delta_pre_eq=0.50, rho=rho, N=N)
    
    print("Equilibrating...")
    x, y, _, _, _ = mc.equilibrate_continuous(x, y, target_T=T_star, total_moves=eq_moves, delta=delta, sample_interval=sample_interval, rho=rho, N=N)
    save_snapshot(x, y, Length, f"Eq. Snapshot (rho={rho})", f"eq_snapshot_rho_{rho}.png")
    
    print("Production...")
    x, y, _, _, trajectory, _ = mc.run_production_sampling(x, y, target_T=T_star, total_moves=prod_moves, delta=delta, sample_interval=sample_interval, rho=rho, N=N)
    save_snapshot(x, y, Length, f"Prod. Snapshot (rho={rho})", f"prod_snapshot_rho_{rho}.png")
    
    print("Animating...")
    create_animation(trajectory, Length, f"trajectory_rho_{rho}.gif")