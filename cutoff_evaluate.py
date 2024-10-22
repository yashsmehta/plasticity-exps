import logging
import time
from typing import Dict, Tuple

import jax
import numpy as np
import optax
import pandas as pd
from jax.random import split

import plasticity.data_loader as data_loader
import plasticity.losses as losses
import plasticity.synapse as synapse
import plasticity.utils as utils
from plasticity.model import (
    evaluate_percent_deviance,
    initialize_params,
    simulate,
)


def split_data(
    data: Dict[str, np.ndarray], cutoff: float
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Splits data into training and evaluation sets based on a cutoff fraction.

    Parameters:
    - data (dict): Dictionary containing data arrays to split.
    - cutoff (float): Fraction of data to be used for training.

    Returns:
    - Tuple containing training data and evaluation data (both dicts).
    """
    if not isinstance(data, dict):
        raise ValueError("Data must be a dictionary.")

    train_data = {}
    eval_data = {}
    for key, value in data.items():
        split_index = int(len(value) * cutoff)
        train_data[key] = value[:split_index]
        eval_data[key] = value[split_index:]
    return train_data, eval_data


def evaluate_model(
    key: jax.random.PRNGKey,
    cfg: Dict,
    plasticity_coeff: np.ndarray,
    plasticity_func,
    xs_eval: Dict[str, np.ndarray],
    decisions_eval: Dict[str, np.ndarray],
    rewards_eval: Dict[str, np.ndarray],
    expected_rewards_eval: Dict[str, np.ndarray],
) -> Tuple[Dict[str, float], float]:
    """
    Evaluates the model performance on the evaluation dataset.

    Parameters:
    - key (jax.random.PRNGKey): Random number generator key.
    - cfg (dict): Configuration object containing model settings.
    - plasticity_coeff (np.ndarray): Array of plasticity coefficients.
    - plasticity_func (function): Plasticity function.
    - xs_eval (dict): Input data for evaluation.
    - decisions_eval (dict): Decisions for evaluation.
    - rewards_eval (dict): Rewards for evaluation.
    - expected_rewards_eval (dict): Expected rewards for evaluation.

    Returns:
    - percent_deviance (float): Median percent deviance explained across all experiments.
    """

    percent_deviances = []

    for exp_i in decisions_eval:
        key, subkey = jax.random.split(key)
        params = initialize_params(subkey, cfg)
        trial_lengths = data_loader.get_trial_lengths(decisions_eval[exp_i])

        _, model_activations = simulate(
            params,
            plasticity_coeff,
            plasticity_func,
            xs_eval[exp_i],
            rewards_eval[exp_i],
            expected_rewards_eval[exp_i],
            trial_lengths,
        )

        # Simulate null model with zero plasticity coefficients
        zero_plasticity_coeff, zero_plasticity_func = synapse.init_plasticity_volterra(
            None, init="zeros"
        )
        _, null_model_activations = simulate(
            params,
            zero_plasticity_coeff,
            zero_plasticity_func,
            xs_eval[exp_i],
            rewards_eval[exp_i],
            expected_rewards_eval[exp_i],
            trial_lengths,
        )

        percent_deviance = evaluate_percent_deviance(
            decisions_eval[exp_i], model_activations, null_model_activations
        )
        percent_deviances.append(percent_deviance)

    median_percent_deviance = np.median(percent_deviances)
    logging.info(f"Median percent deviance explained: {median_percent_deviance}")

    return median_percent_deviance


def train_model(cfg: Dict) -> Tuple[np.ndarray, Dict, float, Tuple]:
    """
    Trains the model using the provided configuration.

    Parameters:
    - cfg (dict): Configuration object containing model settings.

    Returns:
    - plasticity_coeff (np.ndarray): Trained plasticity coefficients.
    - expdata (dict): Dictionary containing training logs.
    - train_time (float): Total training time in seconds.
    - evaluation_data (tuple): Tuple containing evaluation datasets.
    """
    key = jax.random.PRNGKey(cfg.expid)
    noise_key = jax.random.PRNGKey(10 * cfg.expid)

    # Load and split data
    xs, _, decisions, rewards, expected_rewards = data_loader.load_fly_expdata(
        key=key, cfg=cfg, mode="train"
    )
    cutoff = cfg.trajectory_cutoff
    xs_train, xs_eval = split_data(xs, cutoff)
    decisions_train, decisions_eval = split_data(decisions, cutoff)
    rewards_train, rewards_eval = split_data(rewards, cutoff)
    expected_rewards_train, expected_rewards_eval = split_data(
        expected_rewards, cutoff
    )

    # Initialize model parameters and plasticity coefficients
    key, subkey = split(key)
    params = initialize_params(key, cfg)

    plasticity_coeff, plasticity_func = synapse.init_plasticity(
        subkey, cfg, mode="plasticity_model"
    )

    # Set up optimizer
    optimizer = optax.adam(learning_rate=cfg.learning_rate)
    opt_state = optimizer.init(plasticity_coeff)
    loss_value_and_grad = jax.value_and_grad(losses.loss, argnums=2)

    expdata = {}
    start_time = time.time()

    # Training loop
    for epoch in range(cfg.num_epochs + 1):
        epoch_losses = []
        for exp_i in decisions_train:
            noise_key, sub_noise_key = split(noise_key)

            # Compute loss and gradients
            loss, meta_grads = loss_value_and_grad(
                sub_noise_key,
                params,
                plasticity_coeff,
                plasticity_func,
                xs_train[exp_i],
                rewards_train[exp_i],
                expected_rewards_train[exp_i],
                None,
                decisions_train[exp_i],
                cfg,
            )

            # Update plasticity coefficients
            updates, opt_state = optimizer.update(meta_grads, opt_state)
            plasticity_coeff = optax.apply_updates(plasticity_coeff, updates)
            epoch_losses.append(loss)

        # Logging
        if epoch % cfg.log_interval == 0:
            avg_loss = np.mean(epoch_losses)
            expdata = utils.print_and_log_training_info(
                cfg, expdata, plasticity_coeff, epoch, avg_loss
            )
            logging.info(f"Epoch {epoch}, Average Loss: {avg_loss}")

    train_time = round(time.time() - start_time, 3)
    logging.info(f"Training completed in {train_time}s")

    evaluation_data = (
        xs_eval,
        decisions_eval,
        rewards_eval,
        expected_rewards_eval,
    )

    return plasticity_coeff, plasticity_func, expdata, train_time, evaluation_data


def save_results(
    cfg: Dict, expdata: Dict, train_time: float, evaluation_results: Tuple[Dict, float]
) -> pd.DataFrame:
    """
    Saves training and evaluation results to a DataFrame and logs them.

    Parameters:
    - cfg (dict): Configuration object containing model settings.
    - expdata (dict): Dictionary containing training logs.
    - train_time (float): Total training time in seconds.
    - evaluation_results (tuple): Contains evaluation metrics.

    Returns:
    - df (pd.DataFrame): DataFrame containing logged data.
    """
    percent_deviance = evaluation_results

    df = pd.DataFrame.from_dict(expdata)
    df["train_time"] = train_time
    df["percent_deviance"] = percent_deviance

    for key_cfg, value in cfg.items():
        if isinstance(value, (float, int, str)):
            df[key_cfg] = value
    df["layer_sizes"] = str(cfg.layer_sizes)

    utils.save_logs(cfg, df)

    return df


def setup_environment(cfg: Dict) -> Dict:
    """
    Sets up the environment based on the configuration.

    Parameters:
    - cfg (dict): Configuration object containing model settings.

    Returns:
    - cfg (dict): Validated and possibly updated configuration object.
    """
    cfg = utils.validate_config(cfg)
    try:
        jax.config.update("jax_platform_name", cfg.device)
    except Exception as e:
        logging.warning(f"Could not set platform to {cfg.device}: {e}")
    device = jax.lib.xla_bridge.get_backend().platform
    logging.info(f"Platform: {device}")
    logging.info(f"Layer sizes: {cfg.layer_sizes}")
    return cfg


def main(cfg: Dict):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename="training.log",
        filemode="w",
    )

    cfg = setup_environment(cfg)
    logging.info(f"Training trajectory_cutoff: {cfg.trajectory_cutoff}")

    # Train the model
    (
        plasticity_coeff,
        plasticity_func,
        expdata,
        train_time,
        evaluation_data,
    ) = train_model(cfg)

    # Evaluate the model
    (
        xs_eval,
        decisions_eval,
        rewards_eval,
        expected_rewards_eval,
    ) = evaluation_data
    key = jax.random.PRNGKey(cfg.expid)

    percent_deviance = evaluate_model(
        key,
        cfg,
        plasticity_coeff,
        plasticity_func,
        xs_eval,
        decisions_eval,
        rewards_eval,
        expected_rewards_eval,
    )
    print("percent_deviance", percent_deviance)

    # Save results
    df = save_results(cfg, expdata, train_time, percent_deviance)
    logging.info("Final results:")
    logging.info(df.tail(5))
