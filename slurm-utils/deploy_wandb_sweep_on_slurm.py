import click
import subprocess

@click.command()
@click.argument("n_jobs")
@click.argument("sweep_address")
def main(n_jobs, sweep_address):
    for i in range(n_jobs):
        command1 = subprocess.Popen(['srun','slurm-utils/slurm_wandb_sweep.sh', sweep_address])

if __name__ == "__main__":
    main()