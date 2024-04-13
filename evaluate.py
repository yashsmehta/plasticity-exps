
import plasticity.synapse as synapse
import plasticity.data_loader as data_loader
import plasticity.model as model
import plasticity.utils as utils
import jax
from jax.random import split
import numpy as np
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


def simulate_model(cfg):
    cfg = utils.validate_config(cfg)
    np.set_printoptions(suppress=True, threshold=sys.maxsize)
    key = jax.random.PRNGKey(99)
    key, subkey = split(key)

    plasticity_coeff, plasticity_func = synapse.init_plasticity(
        subkey, cfg, mode="plasticity_model"
    )

    params = model.initialize_params(key, cfg)
    # Get all the csv files in the current directory
    csv_file = "logs/simdata/eval/volterra/exp_1.csv"
    df = pd.read_csv(csv_file)
    df = df.loc[(df['l1_regularization'] == 1e-2) & (df['epoch'] == 2500)]
    plasticity_coeff = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                plasticity_coeff[i,j,k] = df[f"A_{i}{j}{k}"].values

    (
        resampled_xs,
        neural_recordings,
        decisions,
        rewards,
        expected_rewards,
    ) = data_loader.load_data(key, cfg, mode="eval")

    trial_lengths = data_loader.get_trial_lengths(decisions["0"])
    logits_mask = data_loader.get_logits_mask(decisions["0"])

    # simulate model with learned plasticity coefficients (plasticity_coeff)
    model_params_trajec, model_activations = model.simulate(
        params,
        plasticity_coeff,
        plasticity_func,
        resampled_xs["0"],
        rewards["0"],
        expected_rewards["0"],
        trial_lengths,
    )

    volterra_weight_trajec = model_params_trajec[0][0]
    print("model params trajec shape: ", volterra_weight_trajec.shape)
    volterra_w_trajec = volterra_weight_trajec[:, 0, 0]
    print("volterra_w_trajec shape: ", volterra_w_trajec.shape)

    volterra_activity_trajec = jax.nn.sigmoid(model_activations[-1])
    print("activity trajec shape: ", volterra_activity_trajec.shape)
    volterra_neuron = np.array([volterra_activity_trajec[i, trial_lengths[i] - 1, 0] for i in range(len(trial_lengths))])
    print("model neuron shape: ", volterra_neuron.shape)

    generation_coeff, generation_func = synapse.init_plasticity(
        key, cfg, mode="generation_model"
    )

    params_trajec, activations = model.simulate(
        params,
        generation_coeff,
        generation_func,
        resampled_xs["0"],
        rewards["0"],
        expected_rewards["0"],
        trial_lengths,
    )
    weight_trajec = params_trajec[0][0]
    print("params trajec shape: ", weight_trajec.shape)
    w_trajec = weight_trajec[:, 0, 0]
    print("w_trajec shape: ", w_trajec.shape)

    activity_trajec = jax.nn.sigmoid(activations[-1])
    print("activity trajec shape: ", activity_trajec.shape)
    neuron = np.array([activity_trajec[i, trial_lengths[i] - 1, 0] for i in range(len(trial_lengths))])

    # now get MLP data
    cfg.plasticity_model = 'mlp'
    _, mlp_plasticity_func = synapse.init_plasticity(
        subkey, cfg, mode="plasticity_model"
    )
    file = "logs/simdata/eval/mlp/mlp_params_1.pkl"

    with open(file, 'rb') as f:
        data = pickle.load(f)
    mlp_plasticity_coeff = data[-1]

    mlp_params_trajec, mlp_activations = model.simulate(
        params,
        mlp_plasticity_coeff,
        mlp_plasticity_func,
        resampled_xs["0"],
        rewards["0"],
        expected_rewards["0"],
        trial_lengths,
    )
    mlp_weight_trajec = mlp_params_trajec[0][0]
    print("params trajec shape: ", mlp_weight_trajec.shape)
    mlp_w_trajec = mlp_weight_trajec[:, 0, 0]
    print("w_trajec shape: ", mlp_w_trajec.shape)

    mlp_activity_trajec = jax.nn.sigmoid(mlp_activations[-1])
    print("activity trajec shape: ", mlp_activity_trajec.shape)
    mlp_neuron = np.array([mlp_activity_trajec[i, trial_lengths[i] - 1, 0] for i in range(len(trial_lengths))])
    w_dict = {'true w':w_trajec, 'mlp':mlp_w_trajec, 'taylor':volterra_w_trajec}

    np.savez('logs/simdata/fig2/weight_trajectories.npz', **w_dict)

    # now plot!
    sns.set(font_scale = 1.2)
    sns.set_style("white")

    fig, ax2 = plt.subplots(figsize=(10, 5))
    epochs = np.arange(len(neuron))

    # sns.lineplot(x=epochs, y=neuron, ax=axs, label='true neuron', linestyle='-', color='gray')
    # sns.lineplot(x=epochs, y=volterra_neuron, ax=axs, label='Taylor', linestyle='-', color='red')
    # sns.lineplot(x=epochs, y=mlp_neuron, ax=axs, label='mlp', linestyle='-', color='orange')
    sns.lineplot(x=epochs, y=w_trajec, ax=ax2, label='true weight', linestyle='--', color='gray')
    sns.lineplot(x=epochs, y=volterra_w_trajec, ax=ax2, label='Taylor', linestyle='--', color='purple')
    sns.lineplot(x=epochs, y=mlp_w_trajec, ax=ax2, label='mlp', linestyle='--', color='green')

    ax2.set_xlabel('Trial')
    ax2.set_ylabel('Weight Trajectory', color='green')
    ax2.legend(loc='upper right')
    plt.title('Evolution of Weight Over Tials')
    plt.tight_layout()
    plt.savefig("fig2a.svg", dpi=500)




