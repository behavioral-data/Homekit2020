from json import load
import os
import glob
import torch

import numpy as np

from src.data.utils import load_json

def get_most_recent_checkpoint(checkpoint_dir):
    subfolders = [ f.path for f in os.scandir(checkpoint_dir) if f.is_dir() ]
    checkpoint_paths = [x for x in subfolders if "checkpoint-" in x]
    checkpoint_number = [int(x.split("checkpoint-")[-1]) for x in checkpoint_paths]
    return checkpoint_paths[np.argmax(checkpoint_number)]

def load_model_from_huggingface_checkpoint(path):
    config_path = os.path.join(path,"model_config.json")
    config = load_json(config_path)
    base_class = globals()[config.pop("model_base_class")]
    model = base_class(**config)
    
    most_recent_checkpoint = get_most_recent_checkpoint(path)
    state_dict_path = os.path.join(most_recent_checkpoint,"pytorch_model.bin")
    
    model.load_state_dict(torch.load(state_dict_path))
    return model



