import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from two_D_project import Particle_config

def analyze_pickle(filename):
    print(f"--- Loading and Analyzing: {filename} ---")
    
    #Load 
    with open(filename, "rb") as f:
        data = pkl.load(f)

    pc = data['config']
    acc_rate = data['acceptance_rate']
    u_err = data['energy_drift']

    # Output
    print(f"Density (rho*): {pc.rho}")
    print(f"Typical Acceptance Ratio: {acc_rate:.3f}")
    print(f"Energy Drift (U_accumulated - U_full): {u_err:.2e}")

    
    #  Energy vs Configurations Plot
    
    plt.figure(figsize=(8, 5))
    plt.plot(pc.U_samples, color='purple', alpha=0.8)
    plt.title(f"Energy vs. Sampled Configurations ($\\rho^*$ = {pc.rho})")
    plt.xlabel("Sample Index")
    plt.ylabel("Potential Energy $U$")
    plt.grid(True)
    plt.savefig(f"energy_trace_rho_{pc.rho}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: energy_trace_rho_{pc.rho}.png")

    
    # 3. Radial Distribution Function g(r)
    
   
    r_vals, g_normalized = pc.average_radial_density(n_bins=300)
    
    plt.figure(figsize=(8, 5))
    plt.plot(r_vals, g_normalized, color='green')
    plt.title(f"Radial Distribution Function $g(r)$ ($\\rho^*$ = {pc.rho})")
    plt.xlabel("Distance $r$")
    plt.ylabel("$g(r)$")
    plt.grid(True)
    plt.savefig(f"g_r_rho_{pc.rho}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: g_r_rho_{pc.rho}.png")

    
    # 4. Final 2D Configurational Snapshot
    
    R_core, R_shoulder = pc.sigma_0 / 2.0, pc.sigma_1 / 2.0
    fig, ax = plt.subplots(figsize=(7, 7))
    for xi, yi in zip(pc.x, pc.y):
        ax.add_patch(plt.Circle((xi, yi), radius=R_shoulder, color="blue", alpha=0.15, ec="blue", ls="--"))
        ax.add_patch(plt.Circle((xi, yi), radius=R_core, color="red", alpha=0.5, ec="black"))
    ax.set_aspect("equal")
    ax.set_xlim(0, pc.L)
    ax.set_ylim(0, pc.L)
    plt.title(f"Final Equilibrium Structure ($\\rho^*$ = {pc.rho})")
    plt.savefig(f"snapshot_rho_{pc.rho}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: snapshot_rho_{pc.rho}.png")

    
    # 5. Trajectory Animation
    
    print("Generating animation...")
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, pc.L)
    ax.set_ylim(0, pc.L)
    ax.set_aspect("equal")
    
    patches = []
    for _ in range(pc.N):
        shoulder = plt.Circle((0, 0), radius=R_shoulder, color="blue", alpha=0.15, ec="blue", ls="--")
        core = plt.Circle((0, 0), radius=R_core, color="red", alpha=0.5, ec="black")
        ax.add_patch(shoulder)
        ax.add_patch(core)
        patches.extend([shoulder, core])
        
    def update(frame):
        x_coords = pc.x_hist[frame]
        y_coords = pc.y_hist[frame]
        for i, (xi, yi) in enumerate(zip(x_coords, y_coords)):
            patches[i*2].center = (xi, yi)     # Soft corona
            patches[i*2+1].center = (xi, yi)   # Hard core
        return patches

    # Use pillow instead of ffmpeg
    ani = animation.FuncAnimation(fig, update, frames=len(pc.x_hist), blit=True)
    gif_filename = f"trajectory_rho_{pc.rho}.gif"
    ani.save(gif_filename, writer='pillow', fps=30)
    plt.close()
    print(f"Saved: {gif_filename}")
    print("--- Analysis Complete! ---")

if __name__ == '__main__':
    # You can change this to point to any of your density or temperature pickles
    analyze_pickle("results_rho_0.1.pkl")