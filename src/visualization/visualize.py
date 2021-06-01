import json
import os

from matplotlib import pyplot as plt
import wandb
import pandas as pd

from src.data.utils import download_wandb_table

def plot_roc_curve(run_id,table_name="roc_table",plt_kwargs={},
                 entity="mikeamerrill", project="flu"):
    data = download_wandb_table(run_id,table_name=table_name,
                                entity=entity, project=project)
    plt.plot(data["fpr"],data["tpr"],**plt_kwargs)
    plt.plot([0,1],[0,1], linestyle = "--", color="grey")
    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.gca().set_aspect("equal")
