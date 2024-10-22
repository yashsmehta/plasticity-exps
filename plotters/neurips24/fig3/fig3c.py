import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd

if __name__ == "__main__":
    csv_dir_taylor = 'logs/simdata/fig3/volterra'
    csv_files_taylor = [f for f in os.listdir(csv_dir_taylor) if f.endswith('.csv')]
    df_list_taylor = [pd.read_csv(os.path.join(csv_dir_taylor, f)) for f in csv_files_taylor]
    df_taylor = pd.concat(df_list_taylor, ignore_index=True)
    df_taylor = df_taylor[df_taylor['epoch'] == 350]

    csv_dir_mlp = 'logs/simdata/fig3/mlp'
    csv_files_mlp = [f for f in os.listdir(csv_dir_mlp) if f.endswith('.csv')]
    df_list_mlp = [pd.read_csv(os.path.join(csv_dir_mlp, f)) for f in csv_files_mlp]
    df_mlp = pd.concat(df_list_mlp, ignore_index=True)
    df_mlp = df_mlp[df_mlp['epoch'] == 350]

    df_taylor['model'] = 'Taylor'
    df_mlp['model'] = 'MLP'
    df_combined = pd.concat([df_taylor, df_mlp], ignore_index=True)

    sns.set_theme(style="white", palette="muted", font_scale=2.0)
    fig, ax = plt.subplots(figsize=(5, 6))

    boxplot = sns.boxplot(x='model', y='percent_deviance', hue='model', data=df_combined, ax=ax, palette={"Taylor": "darkorange", "MLP": "forestgreen"}, fliersize=0, width=0.5)
    for patch in boxplot.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .3))

    sns.stripplot(x='model', y='percent_deviance', hue='model', data=df_combined, ax=ax, palette={"Taylor": "darkorange", "MLP": "forestgreen"}, legend=False, size=6, edgecolor='w', linewidth=1, jitter=True, zorder=2)

    ax.set_xlabel('', fontsize=20)
    ax.set_ylabel('', fontsize=20)  # Remove y-axis label
    ax.set_title('% Deviance\nExplained', fontsize=25, fontweight='bold', ha='center')  # Set title instead
    ax.tick_params(width=2, length=6, labelsize=25)  # Increase tick label size
    ax.spines['bottom'].set_linewidth(3)  # Thicken x-axis
    ax.spines['left'].set_linewidth(3)    # Thicken y-axis
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.set_yticklabels(['0', '50', '100'])

    sns.despine(trim=True)
    plt.tight_layout(pad=2)
    plt.savefig("plotters/neurips24/imgs/fig3c.svg", dpi=600)
    plt.close(fig)
    print("saved fig!")
