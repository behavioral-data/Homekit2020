from sys import path_importer_cache
import multiprocessing

import click
import pandas as pd
from distributed import Client
from tqdm import tqdm
tqdm.pandas()

from src.utils import read_yaml, get_logger
logger = get_logger(__name__)

from src.data.utils import  read_parquet_to_pandas
from src.models.features import get_feature_with_name


def feature_generator_from_config_path(feature_config_path,return_meta=True):
    feature_config = read_yaml(feature_config_path)
    print(feature_config["feature_names"])
    feature_fns = [(name,get_feature_with_name(name)) for name in feature_config["feature_names"]]

    meta = {name:float for name in feature_config["feature_names"]}
    def gen_features(partiton):
        result = {}
        for name, fn in feature_fns:
            result[name] = fn(partiton)
            meta[name] = float
        return pd.Series(result)
    
    if return_meta:
        return gen_features, meta
    else:
        return gen_features

@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.argument("out_path", type=click.Path(exists=False))
@click.argument("in_path", type=click.Path(exists=True))
def main(config_path,out_path, in_path):
    generator, meta = feature_generator_from_config_path(config_path)
    logger.info("Genrating features")
    logger.info(f"Features : {list(meta.keys())}")

    raw = read_parquet_to_pandas(in_path)
    raw["date"] = raw["timestamp"].dt.date
    features = raw.groupby(["participant_id","date"]).progress_apply(generator)
    features.dropna().to_csv(out_path)

if __name__ == "__main__":
    main()