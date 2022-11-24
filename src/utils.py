import os
import json
import logging
import gc
import argparse
import click

import wandb
import pandas as pd
import dotenv
import yaml
import numpy as np
import torch
from torchviz import make_dot
import subprocess
from scipy.special import softmax

from dotenv import dotenv_values
config = dotenv_values(".env")

def validate_yaml_or_json(ctx, param, value):
    if value is None:
        return
    try:
        return read_yaml(value)
    except FileNotFoundError:
        try:
            return json.loads(value)
        except json.decoder.JSONDecodeError:
            raise click.BadParameter('dataset_args needs to be either a json string or a path to a config .yaml')

def load_dotenv():
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    dotenv_path = os.path.join(project_dir, '.env')
    dotenv.load_dotenv(dotenv_path)

def read_yaml(path):
    with open(path, 'r') as stream:
        return yaml.safe_load(stream)

def write_yaml(data,path):
    with open(path, 'w') as stream:
        yaml.dump(data, stream)

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

def read_jsonl(path,line=None):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_logger(name):
    logger = logging.getLogger(name)
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

def render_network_plot(var,dir,filename="model",params=None):
    graph = make_dot(var,params=params)
    graph.format = "png"
    return graph.render(filename=filename,directory=dir)


def get_unused_gpus():
    result=subprocess.getoutput("nvidia-smi -q -d PIDS |grep -A4 GPU | grep Processes").split("\n")
    return [str(i) for i in range(len(result)) if "None" in result[i]]

def set_gpus_automatically(n):
    free_devices = get_unused_gpus()
    n_free = len(free_devices)
    if n_free < n:
        raise ValueError(f"Insufficent GPUs available for automatic allocation: {n_free} available, {n} requested.")
    devices = free_devices[:n]

    logger = get_logger(__name__)
    logger.info(f"Auto setting gpus to: {devices}")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)


def visualize_model(model,dir="."):
    """
    Returns the path to an image of a model
    """
    x_dummy = torch.rand(model.input_dim).unsqueeze(0)
    y_dummy = torch.tensor(1).unsqueeze(0)
    pred_dummy = model(inputs_embeds=x_dummy, labels = y_dummy)[0]

    params = dict(model.named_parameters())
    model_img_path = render_network_plot(pred_dummy,dir,params=params)
    return model_img_path

def describe_resident_tensors():
    tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                tensors.append((type(obj), obj.size()))
        except:
            pass
    return tensors


def update_wandb_run(run_id,vals):
    project = config["WANDB_PROJECT"]
    entity = config["WANDB_USERNAME"]
    api = wandb.Api()
    run_url = f"{entity}/{project}/{run_id}"
    run = api.run(run_url)
    for k,v in vals.items():
        update_run(run,k,v)
    run.summary.update()
    return  f"https://wandb.ai/{entity}/{project}/runs/{run_id}"


def update_run(run, k, v):
    if (isinstance(run.summary, wandb.old.summary.Summary) and k not in run.summary):
        run.summary._root_set(run.summary._path, [(k, {})])
    run.summary[k] = v

def get_wandb_summaries(runids, project=None, entity=None):
    results = []
    api = wandb.Api()
    if not project:
        project = config["WANDB_PROJECT"]
    if not entity:
        entity = config["WANDB_USERNAME"]

    for run_id in runids:
        run_url = f"{entity}/{project}/{run_id}"
        run = api.run(run_url)
        summary = run.summary._json_dict

        meta = json.load(run.file("wandb-metadata.json").download(replace=True))
        summary["command"] = " ".join(["python", meta["program"]] +  meta["args"])

        summary["id"] = run_id
        results.append(summary)

    return results


def upload_pandas_df_to_wandb(run_id,table_name,df,run=None):

    model_table = wandb.Table(dataframe=df)
    if run:
        run.log({table_name:model_table})
    else:
        with get_historical_run(run_id) as run:
            run.log({table_name:model_table})

def get_historical_run(run_id: str):
    """Allows restoring an historical run to a writable state
    """
    return wandb.init(id=run_id, resume='allow', settings=wandb.Settings(start_method='fork'))


def binary_logits_to_pos_probs(arr,pos_index=-1):
    probs = softmax(arr,axis=1)
    return probs[:,pos_index]

def download_table(run_id, table_name,v="latest"):
    api = wandb.Api()
    entity = config["WANDB_USERNAME"]
    project = config["WANDB_PROJECT"]
    artifact = api.artifact(f"{entity}/{project}/run-{run_id}-{table_name}:{v}")
    table = artifact.get(table_name)
    print(table)
    return pd.DataFrame(table.data, columns=table.columns)

def argparse_to_groups(args,parser):
    """
    Takes argparse args and a parser and returns results seperated by groups.
    Taken from:
    https://stackoverflow.com/questions/38884513/python-argparse-how-can-i-get-namespace-objects-for-argument-groups-separately
    """
    arg_groups={}

    for group in parser._action_groups:
        group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
        arg_groups[group.title]=argparse.Namespace(**group_dict)

    return arg_groups