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

    volterra_coefficients = []
    for index, row in df_taylor.iterrows():
        coefficients = [
            row[f"A_{i}{j}{k}{l}"]
            for i in range(3)
            for j in range(3)
            for k in range(3)
            for l in range(3)
        ]
        volterra_coefficients.append(coefficients)

    df_coefficients = pd.DataFrame(volterra_coefficients, columns=[f"A_{i}{j}{k}{l}" for i in range(3) for j in range(3) for k in range(3) for l in range(3)])
    df_coefficients = df_coefficients.melt(var_name='Coefficient', value_name='Value')

    df_coefficients['Coefficient'] = df_coefficients['Coefficient'].apply(lambda x: x.replace('A_', 'θ_'))

    sns.set_theme(style="white", palette="muted", font_scale=2.0)
    fig, ax = plt.subplots(figsize=(12, 7))

    red_shade = '#B22222'  # Changed to another shade of red
    custom_palette = {label: red_shade if label == 'θ_1001' else 'grey' for label in df_coefficients['Coefficient'].unique()}

    ax.set_title('Learned Taylor Coefficients', fontsize=25, fontweight='bold')
    boxplot = sns.boxplot(x='Coefficient', y='Value', hue='Coefficient', data=df_coefficients, ax=ax, palette=custom_palette, fliersize=0, width=3, dodge=False, legend=False)
    for patch in boxplot.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .3))
    sns.stripplot(x='Coefficient', y='Value', hue='Coefficient', data=df_coefficients, ax=ax, palette=custom_palette, size=5, edgecolor='w', linewidth=0.7, jitter=0.4, zorder=2, legend=False)
    xtick_labels = ['θ_0000', 'θ_0001', 'θ_1001', 'θ_2222']
    xtick_labels_display = [r'$\theta_{0000}$', r'$\theta_{0001}$', r'$\theta_{1001}$', r'$\theta_{2222}$']
    xtick_indices = [df_coefficients['Coefficient'].unique().tolist().index(label) for label in xtick_labels]
    ax.set_xticks(xtick_indices)
    ax.set_xticklabels(xtick_labels_display, fontsize=25)
    ax.set_xlabel('', fontsize=20)
    ax.set_ylabel('', fontsize=20)
    ax.tick_params(axis='x', width=2, length=6, labelsize=18, rotation=90)  # Increase tick label size and rotate x-axis labels
    ax.tick_params(axis='y', width=2, length=6, labelsize=25)  # Increase tick label size for y-axis without rotation
    ax.spines['bottom'].set_linewidth(3)  # Thicken x-axis
    ax.spines['left'].set_linewidth(3)    # Thicken y-axis

    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(['0', '0.5', '1'])

    sns.despine(trim=True)
    plt.tight_layout(pad=2)
    plt.savefig("plotters/neurips24/imgs/fig3d.svg", dpi=600)
    plt.close(fig)
    print("saved fig!")
