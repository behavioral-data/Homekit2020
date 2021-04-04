import faulthandler; faulthandler.enable()
import warnings
# from ast import literal_eval
from json import loads
import argparse
warnings.filterwarnings("ignore")

import click

from src.models.train_model import train_neural_baseline, train_bert, train_longformer, train_autoencoder, train_sand
from src.utils import get_logger
logger = get_logger()

@click.group()
def cli():
    pass
cli.add_command(train_neural_baseline)
cli.add_command(train_bert)
cli.add_command(train_longformer)
cli.add_command(train_autoencoder)
cli.add_command(train_sand)


# @cli.command(context_settings=dict(
#     ignore_unknown_options=True,
# ))
# @click.argument("task_name")
# @click.argument("model_name")
# @click.argument('model_args', nargs=-1, type=click.UNPROCESSED)
# @click.option('--dataset_args')
# @click.pass_context
# def train(ctx,task_name,model_name,model_args,dataset_args={}):
#     logger.info(f"Training {model_name} on {task_name}")

#     dataset_args = loads(dataset_args)
#     # model_args = process_model_args(model_args)

#     task = get_task_with_name(task_name)(dataset_args=dataset_args)
#     if model_name in ["cnn","lstm"]:
#         train_neural_baseline(model_name,task,model_args)
#     else:
#         raise NotImplementedError(f"{model_name} not supported")


if __name__ == "__main__":
    print("hey!")
    cli()