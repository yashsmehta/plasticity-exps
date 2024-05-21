import jax
from jax.random import split
import numpy as np
import pandas as pd
import sys
import pickle
import os
import plasticity.synapse as synapse
import plasticity.data_loader as data_loader
import plasticity.model as model
import plasticity.utils as utils


def load_volterra_coefficients(file_path, shape):
    df = pd.read_csv(file_path)
    df = df.loc[(df["l1_regularization"] == 0.01) & (df["epoch"] == 100)]
    return np.array(
        [
            df[f"A_{i}{j}{k}{l}"].values
            for i in range(3)
            for j in range(3)
            for k in range(3)
            for l in range(3)
        ]
    ).reshape(shape)


def simulate_and_extract_trajectories(
    params,
    plasticity_coeff,
    plasticity_func,
    resampled_xs,
    rewards,
    expected_rewards,
    trial_lengths,
):
    model_params_trajec, model_activations = model.simulate(
        params,
        plasticity_coeff,
        plasticity_func,
        resampled_xs,
        rewards,
        expected_rewards,
        trial_lengths,
    )
    synapse_trajec = model_params_trajec[0][0][:, 0, 0]
    neuron_trajec = np.array(
        [
            jax.nn.sigmoid(model_activations[-1])[i, trial_lengths[i] - 1, 0]
            for i in range(len(trial_lengths))
        ]
    )
    return synapse_trajec, neuron_trajec


def simulate_model(cfg):
    cfg = utils.validate_config(cfg)
    np.set_printoptions(suppress=True, threshold=sys.maxsize)
    key, subkey = split(jax.random.PRNGKey(99))

    cfg.plasticity_model = "volterra"
    _, plasticity_func = synapse.init_plasticity(subkey, cfg, mode="plasticity_model")
    params = model.initialize_params(key, cfg)
    plasticity_coeff = load_volterra_coefficients(
        "logs/simdata/eval/volterra/exp_1.csv", (3, 3, 3, 3)
    )

    (
        resampled_xs,
        neural_recordings,
        decisions,
        rewards,
        expected_rewards,
    ) = data_loader.load_data(key, cfg, mode="eval")
    trial_lengths = data_loader.get_trial_lengths(decisions["0"])

    volterra_w_trajec, _ = simulate_and_extract_trajectories(
        params,
        plasticity_coeff,
        plasticity_func,
        resampled_xs["0"],
        rewards["0"],
        expected_rewards["0"],
        trial_lengths,
    )
    print("got volterra weight trajectory")

    generation_coeff, generation_func = synapse.init_plasticity(
        key, cfg, mode="generation_model"
    )
    true_w_trajec, _ = simulate_and_extract_trajectories(
        params,
        generation_coeff,
        generation_func,
        resampled_xs["0"],
        rewards["0"],
        expected_rewards["0"],
        trial_lengths,
    )
    print("got true weight trajectory")

    cfg.plasticity_model = "mlp"
    _, mlp_plasticity_func = synapse.init_plasticity(
        subkey, cfg, mode="plasticity_model"
    )
    with open("logs/simdata/eval/mlp/mlp_params_1.pkl", "rb") as f:
        mlp_plasticity_coeff = pickle.load(f)[-1]

    mlp_w_trajec, _ = simulate_and_extract_trajectories(
        params,
        mlp_plasticity_coeff,
        mlp_plasticity_func,
        resampled_xs["0"],
        rewards["0"],
        expected_rewards["0"],
        trial_lengths,
    )
    print("got mlp weight trajectory")

    w_dict = {"true_w": true_w_trajec, "mlp": mlp_w_trajec, "taylor": volterra_w_trajec}

    output_dir = "logs/simdata/fig2"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.savez(os.path.join(output_dir, "weight_trajectories.npz"), **w_dict)
    print("saved weight trajectories!")
