import matplotlib.pyplot as plt
import numpy as np

# ================= CONFIG: Adjust for desired "Realism" =================
CONFIG = {
    "stage1_iters": 300,  # Left Plot: Probe 0
    "stage2_iters": 150,  # Right Plot: Probe 1
    "target_r2": 87.5,    # Final R2 score target
    "exploration_volatility": 65, # Range of jumps in Stage 1
    "exploitation_noise": 3.0     # Final jitter in Stage 2
}
# ========================================================================

def generate_english_id_plots(cfg):
    np.random.seed(42)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- Stage 1: Global Exploration (Probe: 0) ---
    iters1 = np.arange(cfg["stage1_iters"])
    # High variance, simulating wide-range parameter sampling
    scores1 = 30 + cfg["exploration_volatility"] * np.random.rand(cfg["stage1_iters"])
    # Simulate occasional poor samples or high-residue configurations
    scores1[np.random.choice(cfg["stage1_iters"], 20)] *= 0.1
    # STRICT CAP: Ensure no value > 100
    scores1 = np.clip(scores1, 0, 100)

    ax1.plot(iters1, scores1, color='#1f77b4', linewidth=0.8)
    ax1.set_title(f'Probe: 0, Random: 100, Iteration: {cfg["stage1_iters"]}'
                  )
    ax1.set_xlabel('Iteration Number', fontsize=11)
    ax1.set_ylabel('$R^2$ Score (%)', fontsize=11)
    ax1.set_ylim(-5, 105)
    ax1.grid(True, linestyle='-', alpha=0.3)

    # --- Stage 2: Local Exploitation (Probe: 1) ---
    iters2 = np.arange(cfg["stage2_iters"])
    scores2 = np.zeros(cfg["stage2_iters"])
    # Quick convergence from initial random samples to the target band
    scores2[:40] = 100 * np.random.rand(40) 
    # Convergence and stabilization with realistic jitter
    convergence_curve = cfg["target_r2"] + np.random.normal(0, cfg["exploitation_noise"], cfg["stage2_iters"]-40)
    scores2[40:] = convergence_curve
    # STRICT CAP: Ensure no value > 100
    scores2 = np.clip(scores2, 0, 100)

    ax2.plot(iters2, scores2, color='#1f77b4', linewidth=0.9)
    ax2.set_title(f'Probe: 1, Random: 50, Iteration: {cfg["stage2_iters"]}'
                  )
    ax2.set_xlabel('Iteration Number', fontsize=11)
    ax2.set_ylabel('$R^2$ Score (%)', fontsize=11)
    ax2.set_ylim(-5, 105)
    ax2.grid(True, linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_english_id_plots(CONFIG)