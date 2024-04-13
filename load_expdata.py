import scipy.io as sio
import numpy as np
import os
import collections

data_dir = "../data/"
num_exps = 1
num_blocks = 3
element_dim = 2

xs, decisions, rewards, expected_rewards = {}, {}, {}, {}


def expected_reward_for_exp_data(R, moving_avg_window):
    r_history = collections.deque(moving_avg_window * [0], maxlen=moving_avg_window)
    expected_rewards = []
    for r in R:
        expected_rewards.append(np.mean(r_history))
        r_history.appendleft(r)
    return expected_rewards


for exp_i, file in enumerate(os.listdir(data_dir)):
    if exp_i >= num_exps:
        break
    print(exp_i, file)
    data = sio.loadmat(data_dir + file)
    X, Y, R = data["X"], data["Y"], data["R"]
    Y = np.squeeze(Y)
    R = np.squeeze(R)
    num_trials = np.sum(Y)
    assert num_trials == R.shape[0], "Y and R should have the same number of trials"

    print("R.shape", R.shape)
    print("Y.shape", Y.shape)
    print("X.shape", X.shape)

    # remove last element, and append left to get indices.
    indices = np.cumsum(Y)
    indices = np.insert(indices, 0, 0)
    indices = np.delete(indices, -1)
    print("first few Y", Y[:30])

    exp_decisions = [[] for _ in range(num_trials)]
    exp_xs = [[] for _ in range(num_trials)]

    for index, decision, x in zip(indices, Y, X):
        exp_decisions[index].append(decision)
        exp_xs[index].append(x)

    trial_lengths = [len(exp_decisions[i]) for i in range(num_trials)]
    longest_trial_length = np.max(np.array(trial_lengths))
    print("trial_lengths", trial_lengths)
    print("longest_trial_length", longest_trial_length)

    tensor = np.full((num_trials, longest_trial_length), np.nan)
    for i in range(num_trials):
        for j in range(trial_lengths[i]):
            tensor[i][j] = exp_decisions[i][j]
    decisions[str(exp_i)] = tensor

    tensor = np.full((num_trials, longest_trial_length, element_dim), 0)
    for i in range(num_trials):
        for j in range(trial_lengths[i]):
            tensor[i][j] = exp_xs[i][j]
    xs[str(exp_i)] = tensor

    rewards[str(exp_i)] = R
    expected_rewards[str(exp_i)] = expected_reward_for_exp_data(R, 10)
    print("rewards", rewards[str(exp_i)])
    print("expected_rewards", expected_rewards[str(exp_i)])
    print("decision dict", decisions)
    # print("xs dict", xs)
