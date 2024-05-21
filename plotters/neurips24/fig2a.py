import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.5)

    data = np.load('logs/simdata/fig2/weight_trajectories.npz')
    w_trajec = data['true_w']
    volterra_w_trajec = data['taylor']
    mlp_w_trajec = data['mlp']

    fig, ax2 = plt.subplots(figsize=(12, 6))
    epochs = np.arange(len(w_trajec))

    sns.lineplot(x=epochs, y=w_trajec, ax=ax2, label='True Weight', linestyle='-', color='blue', linewidth=2.5)
    sns.lineplot(x=epochs, y=volterra_w_trajec, ax=ax2, label='Taylor', linestyle='-', color='orange', linewidth=2.5)
    sns.lineplot(x=epochs, y=mlp_w_trajec, ax=ax2, label='MLP', linestyle='-', color='green', linewidth=2.5)

    ax2.set_xlabel('Trial', fontsize=14)
    ax2.set_ylabel('Weight Trajectory', fontsize=14)
    ax2.legend(loc='upper right', fontsize=12, title='Legend', title_fontsize='13')
    ax2.set_title('Evolution of Weight Over Trials', fontsize=16, pad=20)
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig("plotters/neurips24/fig2a.png", dpi=500)
    plt.close(fig)
    print("saved fig!")
