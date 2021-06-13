import os
import pickle

import click
import yaml

from src.utils import read_yaml, get_logger
logger = get_logger(__name__)

from src.data.task_datasets import MinuteLevelActivityReader
from src.data.utils import get_cached_datareader_path
from src.models.tasks import get_task_with_name




@click.command()
@click.argument("config_path", type= click.Path(exists=True))
@click.option("--task_name", type=str,help="If provided, cache a task and not a datareader")
@click.option('--data_location', default=None,type=click.Path(exists=True))
@click.option('--cache_path', default=None)
@click.option('--activity_level', default="minute")
@click.option('--postfix', default="",type=str)
def main(config_path, task_name=None, cache_path=None, data_location=None, postfix="",
         activity_level="minute"):

    reader_args = read_yaml(config_path)
    if data_location:
        reader_args["data_location"] = data_location
    
    if not task_name:
        resource = MinuteLevelActivityReader(**reader_args)
        name = os.path.split(os.path.splitext(config_path)[0])[-1] + postfix
    else:
        resource = get_task_with_name(task_name)(dataset_args=reader_args,
                                        activity_level=activity_level)
    
    if not cache_path:
        cache_path = get_cached_datareader_path(name)
    logger.info(f"Dumping pickle to {cache_path}...")
    pickle.dump(resource,open(cache_path,"wb"), protocol=4)

if __name__ == "__main__":
    main()