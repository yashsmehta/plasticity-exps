import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
import time

if __name__ == "__main__":
    csv_dir_taylor = 'logs/simdata/fig3/volterra'
    csv_files_taylor = [f for f in os.listdir(csv_dir_taylor) if f.endswith('.csv')]
    df_taylor = pd.concat((pd.read_csv(os.path.join(csv_dir_taylor, f)) for f in csv_files_taylor), ignore_index=True)

    volterra_coefficients = [
        {
            'epoch': row['epoch'],
            'Coefficient': f"θ_{i}{j}{k}{l}",
            'Value': row[f"A_{i}{j}{k}{l}"]
        }
        for _, row in df_taylor.iterrows()
        for i in range(3)
        for j in range(3)
        for k in range(3)
        for l in range(3)
    ]

    df_coefficients = pd.DataFrame(volterra_coefficients)
    print("now plotting...")
    start_time = time.time()
    sns.set_theme(style="white", palette="muted", font_scale=2.0)
    fig, ax = plt.subplots(figsize=(9, 7))

    red_shade = '#B22222'
    custom_palette = {label: red_shade if label == 'θ_1001' else 'grey' for label in df_coefficients['Coefficient'].unique()}

    ax.set_title('Taylor Coefficients During Training', fontsize=25, fontweight='bold')
    sns.lineplot(x='epoch', y='Value', hue='Coefficient', data=df_coefficients, ax=ax, palette=custom_palette, linewidth=4, legend=False)

    ax.axhline(y=1, color=red_shade, linestyle='--', linewidth=4, alpha=0.4)

    ax.set_xlabel('Epoch', fontsize=20)
    ax.set_ylabel('', fontsize=20)
    ax.tick_params(width=2, length=6, labelsize=20)
    ax.spines['bottom'].set_linewidth(3)  # Thicken x-axis
    ax.spines['left'].set_linewidth(3)    # Thicken y-axis
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(['0', '0.5', '1'])
    ax.set_xlim(left=0)  # Ensure y-axis starts when x=0

    sns.despine(trim=True)
    plt.tight_layout(pad=2)
    plt.savefig("plotters/neurips24/imgs/fig3e.svg", dpi=600)
    plt.close(fig)
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
    print("saved fig!")
