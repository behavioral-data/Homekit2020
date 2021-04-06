import os
import json 
import logging

import dotenv
import yaml
import numpy as np

def load_dotenv():
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    dotenv_path = os.path.join(project_dir, '.env')
    dotenv.load_dotenv(dotenv_path)

def read_yaml(path):
    with open(path, 'r') as stream:
        return yaml.safe_load(stream)

def clean_datum_for_serialization(datum):
    for k, v in datum.items():
        if isinstance(v, (np.ndarray, np.generic)):
            datum[k] = v.tolist()
    return datum

def write_jsonl(open_file, data, mode="a"):
    for datum in data:
        clean_datum = clean_datum_for_serialization(datum)
        open_file.write(json.dumps(clean_datum))
        open_file.write("\n")


def get_logger():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    return logger

def check_for_wandb_run():
    try:
        import wandb
    except ImportError:
        return None
    return wandb.run