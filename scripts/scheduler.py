import subprocess
import itertools
import re

exec_file = "base.py"
# these are the number of parallel jobs that will be launched for each config (with different seeds)
seeds = 3

# queue, cores, use_gpu = "gpu_rtx", 5, True
# queue, cores, use_gpu = "gpu_a100", 12, True
queue, cores, use_gpu = "local", 4, False

if(queue == "local" and use_gpu == True):
    raise Exception ("No GPUs available on this partition!")

coeff_inits = [f"X{i}Y{j}R1W{k}" for i in range(3) for j in range(3) for k in range(2)]
coeff_inits += [f"{f}-0.05X{i}Y{j}W1R{k}" for f in coeff_inits for i in range(2) for j in range(2) for k in range(2)]

# coeff_inits = [f"X{i}Y{j}R0W0" for i in range(3) for j in range(3)]
# coeff_inits = [f"X{i}Y{j}W0R{k}" for i in range(3) for j in range(3) for k in range(3)]
# coeff_inits = ["X1Y0W0R1", "X0Y1W0R1", "X1Y1W0R1", "X1Y0W0R2", "X0Y1W0R2", "X1Y1W0R2"]
# coeff_inits = ["X1Y0W0R1"]
layer_sizes = ["[2, 10, 1]"]
# layer_sizes = ["[2, 10, 1]", "[10, 100, 1]", "[50, 2000, 1]"]

configs = {
    "exp_name": ["recoverability"],
    "num_train": [18],
    "num_eval": [7],
    "num_epochs": [100],
    "l1_regularization": [5e-2],
    "layer_sizes": layer_sizes,
    "plasticity_model": ["volterra", "mlp"],
    "generation_coeff_init": coeff_inits,
    "use_experimental_data": [False],
    "log_expdata": [True],
}

# function to iterate through all values of dictionary:
combinations = list(itertools.product(*configs.values()))

# generate config string to pass to bash script
use_gpu = str(use_gpu).lower()
for combination in combinations:
    execstr = "python " + f"{exec_file}"
    for idx, key in enumerate(configs.keys()):
        execstr += " " + key + "=" + str(combination[idx])

    execstr = re.sub(r'layer_sizes=\[([^\]]+)\]',
                 lambda m: f'"layer_sizes=[{m.group(1)}]"',
                 execstr)

    cmd = ["scripts/submit_job.sh", str(cores), str(seeds), queue, execstr, use_gpu]

    # Run the command and capture the output
    output = subprocess.check_output(
        cmd, stderr=subprocess.STDOUT, universal_newlines=True
    )
    print(output)