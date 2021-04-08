import faulthandler; faulthandler.enable()
import warnings
# from ast import literal_eval
from json import loads
import argparse
warnings.filterwarnings("ignore")

import click

from src.models.train_model import (train_neural_baseline, train_bert, train_longformer, 
                                    train_autoencoder, train_sand, train_cnn_transformer)
from src.models.baselines import select_random
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
cli.add_command(train_cnn_transformer)

cli.add_command(select_random)



if __name__ == "__main__":
    cli()