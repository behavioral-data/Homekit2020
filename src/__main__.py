import faulthandler; faulthandler.enable()
import warnings
# from ast import literal_eval
from json import loads
import argparse
warnings.filterwarnings("ignore")

import click
from src.models.baselines import select_random
from src.utils import get_logger
from src.models.commands import MODEL_COMMANDS
logger = get_logger(__name__)

@click.group()
def cli():
    pass

for command in MODEL_COMMANDS:
    cli.add_command(command)

cli.add_command(select_random)


if __name__ == "__main__":
    cli()