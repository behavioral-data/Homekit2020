import os
import pickle
from re import match

import click
from dask.base import dont_optimize
from pyarrow.dataset import dataset
import yaml

from src.utils import read_yaml, get_logger
logger = get_logger()

from src.data.task_datasets import MinuteLevelActivityReader
from src.data.utils import get_cached_datareader_path


def validate_reader(dataset_args,data_reader):
    props = ["min_date","max_date","split_date","min_windows",
            "day_window_size","max_missing_days_in_window"]
    dont_match = []
    
    for prop in props:
        args_prop = dataset_args.get(prop,None)
        #Assume (maybe a bad assumption) that default args are preserved
        if args_prop is None:
            continue
        reader_prop = getattr(data_reader,prop)
        prop_match = args_prop == reader_prop

        if not prop_match:
            dont_match.append((prop,args_prop,reader_prop))
    return dont_match

def load_cached_activity_reader(name, dataset_args=None,
                                fail_if_mismatched=False,
                                activity_level="minute"):
    if not activity_level == "minute":
        raise NotImplementedError("Can only cache minute level activities")
        
    cache_path = get_cached_datareader_path(name)
    reader = pickle.load(open(cache_path, "rb" ) )
    if dataset_args:
        dont_match = validate_reader(dataset_args,reader)
        if len(dont_match) != 0:
            message = f"Mismatch between cached data reader and dataset args:{dont_match}"
            if fail_if_mismatched:
                raise ValueError(message)
            else: 
                logger.warning(message)

    elif fail_if_mismatched:
        raise(ValueError("In order to check for match with cached activity_reader must pass dataset_args"))
    return reader

@click.command()
@click.argument("config_path", type= click.Path(exists=True))
def main(config_path):
    args =  read_yaml(config_path)
    datareader = MinuteLevelActivityReader(**args)
    name = os.path.split(os.path.splitext(config_path)[0])[-1]
    cache_path = get_cached_datareader_path(name)
    print(name)
    logger.info(f"Dumping pickle to {cache_path}...")
    pickle.dump(datareader,open(cache_path,"wb"), protocol=4)
    
    args["is_cached"] = True
    with open(config_path, 'w') as outfile:
        yaml.dump(args, outfile, default_flow_style=False)
 
if __name__ == "__main__":
    main()