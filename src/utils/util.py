import json
from pathlib import Path
import os
import pickle as p
import numpy as np
import datetime
import torch
from src.models import VADA


def parse_config(config_fp="src/configs/config.json"):
    config_file_path = os.path.join(get_project_root(), config_fp)
    print(f"Using configuration file: {config_fp}")
    with open(config_file_path) as f:
        return json.load(f)


def get_project_root():
    return Path(__file__).parent.parent.parent


def join_root_path_str(path_str):
    return Path(os.path.join(get_project_root(), path_str))


def pickle_save(obj, filename):
    with open(filename, "wb") as f:
        p.dump(obj, f)


def pickle_load(filename):
    with open(filename, "rb") as f:
        return p.load(f)


def load_np_arr(path, allow_pickle=True):
    return np.load(file=path, allow_pickle=allow_pickle)


def save_np_arr(path, arr, allow_pickle=True):
    return np.save(file=path, arr=arr, allow_pickle=allow_pickle)


def get_time_str():
    """
    Generate a unique file name based on the current date and time.

    Returns:
    - A short unique file name string.
    """
    # Get the current date and time
    current_datetime = datetime.datetime.now()

    # Format the datetime as a string (e.g., "20230921_1530" for September 21, 2023, 15:30)
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M")

    return formatted_datetime


def get_run_str(config_path):
    checkpoint_path = parse_config(config_path)['resume_checkpoint_path']
    return "-".join(checkpoint_path.split('/')[-1].split("-")[0:2])

def load_vada_checkpoint(config_path):
    config = parse_config(config_path)

    model = VADA(device='cpu', **config['model']['args'])

    checkpoint = torch.load(os.path.join(get_project_root(), config['resume_checkpoint_path']),
                            map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to('cpu')
    return model

if __name__ == "__main__":
    config = parse_config()
    print(get_project_root())
    print(config)
