import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = np.load('logs/simdata/fig3/weight_trajectories.npz')
    w_trajec = data['true_w']
    volterra_w_trajec = data['taylor']
    mlp_w_trajec = data['mlp']

    sns.set_theme(style="white", palette="muted", font_scale=2.0)
    fig, ax2 = plt.subplots(figsize=(14, 8))
    epochs = np.arange(241)  # Ensure the x-axis extends till 240

    ax2.set_title('Weight of Sample Synapse', fontsize=25, fontweight='bold')
    ax2.axhline(0, color='lightgray', linewidth=4, linestyle='--', alpha=0.6)

    sns.lineplot(x=epochs[:len(w_trajec)], y=w_trajec, ax=ax2, label='True Weight', linestyle='-', color='darkgray', linewidth=4)
    sns.lineplot(x=epochs[:len(volterra_w_trajec)], y=volterra_w_trajec, ax=ax2, label='Taylor', linestyle='-', color='darkorange', linewidth=4, alpha=0.8)
    sns.lineplot(x=epochs[:len(mlp_w_trajec)], y=mlp_w_trajec, ax=ax2, label='MLP', linestyle='-', color='forestgreen', linewidth=4, alpha=0.8)

    ax2.set_xlabel('Trial', fontsize=25)
    ax2.set_ylabel('', fontsize=25)
    ax2.set_xlim(0, 250)
    ax2.set_ylim(-1.5, 3)
    ax2.set_yticks([-1.5, 0, 1.5, 3])
    ax2.tick_params(width=2, length=6, labelsize=25)  # Increase tick label size
    ax2.spines['bottom'].set_linewidth(3)  # Thicken x-axis
    ax2.spines['left'].set_linewidth(3)    # Thicken y-axis

    ax2.legend(loc=(0.50, 0.65), fontsize=20, title_fontsize='25', frameon=False)
    sns.despine(trim=True)
    plt.tight_layout(pad=2)
    plt.savefig("plotters/neurips24/imgs/fig3a.svg", dpi=600)
    plt.close(fig)
    print("saved fig!")
