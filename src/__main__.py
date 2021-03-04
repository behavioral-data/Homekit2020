import warnings
from ast import literal_eval
warnings.filterwarnings("ignore")

import click

from src.models.tasks import get_task_with_name
from src.models.train_model import train_neural_baseline
from src.utils import get_logger
logger = get_logger()

@click.group()
def cli():
    pass

@cli.command(context_settings=dict(
    ignore_unknown_options=True,
))
@click.argument("task_name")
@click.argument("model_name")
@click.argument('model_args', nargs=-1, type=click.UNPROCESSED)
@click.option('--dataset_args')
def train(task_name,model_name,model_args,dataset_args={}):
    logger.info(f"Training {model_name} on {task_name}")

    dataset_args = literal_eval(dataset_args)
    task = get_task_with_name(task_name)("2020-02-01",dataset_args=dataset_args)
    if model_name in ["cnn","lstm"]:
        train_neural_baseline(model_name,task, *model_args)
    else:
        raise NotImplementedError(f"{model_name} not supported")

if __name__ == "__main__":
    cli()