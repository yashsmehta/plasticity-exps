import subprocess
import itertools
import re

exec_file = "base.py"
# these are the number of parallel jobs that will be launched for each config (with different seeds)
flies = 10

# queue, cores, use_gpu = "gpu_rtx", 5, True
# queue, cores, use_gpu = "gpu_tesla", 12, True
queue, cores, use_gpu = "local", 4, False

if(queue == "local" and use_gpu == True):
    raise Exception ("No GPUs available on this partition!")

# coeff_inits = [f"X{i}R{j}W0" for i in range(3) for j in range(3)]
# coeff_inits += [f"{f}-0.05X0R0W1" for f in coeff_inits]
# coeff_inits = [f"X{i}Y{j}W0R{k}" for i in range(3) for j in range(3) for k in range(3)]
coeff_inits = [f"X{i}Y{j}W0R{k}" for i in range(1) for j in range(1) for k in range(1)]

configs = {
    "exp_name": ["recoverability"],
    "num_train": [50],
    "l1_regularization": [5e-2],
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
    execstr = re.sub(r'layer_sizes=(\[\d+,\s*\d+\])',
                        lambda m: f'"layer_sizes={m.group(1)}"',
                        execstr)
    cmd = ["scripts/submit_job.sh", str(cores), str(flies), queue, execstr, use_gpu]

    # Run the command and capture the output
    output = subprocess.check_output(
        cmd, stderr=subprocess.STDOUT, universal_newlines=True
    )
    print(output)