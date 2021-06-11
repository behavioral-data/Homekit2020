# slurm-launch.py
# Usage:
# python slurm-launch.py --exp-name test \
#     --command "rllib train --run PPO --env CartPole-v0"

import argparse
import subprocess
import sys
import time
import os

from pathlib import Path
this_dir = Path(__file__).parents[0]
template_file = this_dir / "slurm-template.sh"

DIR = "${DIR}"
JOB_NAME = "${JOB_NAME}"
ACCOUNT = "${ACCOUNT}"
PARTITION = "${PARTITION}"
NUM_NODES = "${NUM_NODES}"
NUM_GPUS_PER_NODE = "${NUM_GPUS_PER_NODE}"
COMMAND_PLACEHOLDER = "${COMMAND_PLACEHOLDER}"
CONDA_PATH = "${CONDA_PATH}"
GIVEN_NODE = "${GIVEN_NODE}"
CONDA_ENV = "${CONDA_ENV}"
LOG_PATH = "${LOG_PATH}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="The job's run directory")
    parser.add_argument(
        "--exp-name",
        type=str,
        required=True,
        help="The job name and path to logging file (exp_name.log).")
    parser.add_argument(
        "--num-nodes",
        "-n",
        type=int,
        default=1,
        help="Number of nodes to use."),
    parser.add_argument(
        "--account",
        "-a",
        type=str,
        default="cse",
        help="Account to use")
    parser.add_argument(
        "--partition",
        "-p",
        type=str,
        default="cse-gpu",
        help="Parition to run on")
    parser.add_argument(
        "--node",
        "-w",
        type=str,
        help="The specified nodes to use. Same format as the "
        "return of 'sinfo'. Default: ''.")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="Number of GPUs to use in each node. (Default: 0)")
    parser.add_argument(
        "--conda-path",
        type=str,
        default="/gscratch/bdata/$USER/anaconda3/bin/conda"
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default=None,
        help="The name of the conda environment")
    parser.add_argument(
        "--command",
        type=str,
        required=True,
        help="The command you wish to execute. For example: "
        " --command 'python test.py'. "
        "Note that the command must be a string.")
    parser.add_argument(
        "--dry-run",
        action="store_true")
    args = parser.parse_args()

    if args.node:
        # assert args.num_nodes == 1
        node_info = "#SBATCH -w {}".format(args.node)
    else:
        node_info = ""

    job_name = "{}_{}".format(args.exp_name,
                              time.strftime("%m%d-%H%M", time.localtime()))

    if args.conda_path:
        conda_path_option = "source " +  os.path.join(args.conda_path,"etc","profile.d","conda.sh")
    else:
        conda_path_option = ""

    jobs_dir = os.path.join(this_dir,"jobs")
    this_jobs_dir = os.path.join(jobs_dir,job_name)
    os.makedirs(this_jobs_dir,exist_ok=True)
    log_path = os.path.join(this_jobs_dir,"slurm.log")

    # ===== Modified the template script =====
    with open(template_file, "r") as f:
        text = f.read()
    text = text.replace(DIR, args.dir)
    text = text.replace(JOB_NAME, job_name)
    text = text.replace(NUM_NODES, str(args.num_nodes))
    text = text.replace(PARTITION, args.partition)
    text = text.replace(ACCOUNT,args.account)
    text = text.replace(LOG_PATH, log_path)
    text = text.replace(CONDA_PATH,conda_path_option)
    text = text.replace(CONDA_ENV,args.conda_env)
    text = text.replace(NUM_GPUS_PER_NODE, str(args.num_gpus))
    text = text.replace(COMMAND_PLACEHOLDER, str(args.command))
    text = text.replace(GIVEN_NODE, node_info)
    text = text.replace(
        "# THIS FILE IS A TEMPLATE AND IT SHOULD NOT BE DEPLOYED TO "
        "PRODUCTION!",
        "# THIS FILE IS MODIFIED AUTOMATICALLY FROM TEMPLATE AND SHOULD BE "
        "RUNNABLE!")

    # ===== Save the script =====
    script_file = os.path.join(this_jobs_dir,"{}.sh".format(job_name))
    with open(script_file, "w") as f:
        f.write(text)

    if not args.dry_run:
        # ===== Submit the job =====
        print("Starting to submit job!")
        subprocess.Popen(["sbatch", script_file])
        print(
            "Job submitted! Script file is at: <{}>. Log file is at: <{}>".format(
                script_file, "{}.log".format(job_name)))
    else:
        print(f"Dry run! Not submitting job. Script file is at {script_file}")
    sys.exit(0)
