import subprocess
import itertools
import re


if __name__ == "__main__":
    exec_file = "base.py"
    seeds = 3
    queue, cores, use_gpu = "local", 4, False

    if queue == "local" and use_gpu:
        raise Exception("No GPUs available on this partition!")

    coeff_inits = ["XR", "X2R", "R-0.05W","R-0.05XW", "XR-0.05W", "XR-0.05XW", "X2R-0.05W", "X2R2-0.05W2"]
    layer_sizes = ["[2, 10, 1]"]

    configs = {
        "exp_name": ["table1"],
        "num_train": [18],
        "num_eval": [7],
        "num_epochs": [100],
        "l1_regularization": [1e-2],
        "layer_sizes": layer_sizes,
        "plasticity_model": ["volterra", "mlp"],
        "generation_coeff_init": coeff_inits,
        "use_experimental_data": [False],
        "log_expdata": [True],
    }

    combinations = list(itertools.product(*configs.values()))
    use_gpu = str(use_gpu).lower()

    for combination in combinations:
        execstr = "python " + exec_file
        for idx, key in enumerate(configs.keys()):
            execstr += " " + key + "=" + str(combination[idx])

        execstr = re.sub(r'layer_sizes=\[([^\]]+)\]',
                         lambda m: f'"layer_sizes=[{m.group(1)}]"',
                         execstr)

        cmd = ["scripts/submit_job.sh", str(cores), str(seeds), queue, execstr, use_gpu]
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True)
        print(output)
