import glob
import os

import click

import pandas as pd
import numpy as np

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

import pyspark.sql.types as sql_types
from pyspark.sql import functions as f
from pyspark.sql.functions import col

from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark import SparkContext

# TODO Maybe want to support loading this from a file
# although this hardcoded config works pretty well for
# bdata's darkwing machine
SPARK_CONFIG = [ 
    ("spark.master","local[16]"),
    ("spark.ui.port","4050"),
    ("spark.executor.memory","750g"),
    ('spark.driver.memory',  '2000g'),
    ("spark.driver.maxResultSize", '0'), # unlimited
    ("spark.network.timeout",            "10000001"),
    ("spark.executor.heartbeatInterval", "10000000")]   

@click.command()
@click.argument("in_path", type=click.Path())
@click.argument("out_path", type=click.Path(file_okay=False))
@click.option("--split_date",default=None)
@click.option("--end_date",default=None)
@click.option("--eval_frac",default=None)
@click.option("--test_frac", default=0.5, help="Fraction of eval set that's reserved for testing")
@click.option("--activity_level", type=click.Choice(["day","minute"]), default="minute")
def main(in_path, out_path, split_date=None, end_date=None,
        test_frac = 0.5, eval_frac = None, activity_level="minute",
        timestamp_col = "timestamp"):

        if not activity_level == "minute":
            raise NotImplementedError("This script only supports minute-level data")
        
        conf = SparkConf().setAll(SPARK_CONFIG)
        sc = SparkContext(conf=conf, appName="PetaStorm Conversion")
        spark = SparkSession(sc)

        df = spark.read.parquet(in_path)
        if end_date:
            df = df.where(col("date") < pd.to_datetime(end_date))

        train_df = df.where(col("date") < pd.to_datetime(split_date))
        test_eval = df.where(col("date") >= pd.to_datetime(split_date))

        test_eval_split = (test_eval.select("participant_id")
                      .distinct()  # removing duplicate account_ids
                      .withColumn("rand_val", f.rand())
                      .withColumn("data_type", f.when(f.col("rand_val") < test_frac, "test")
                                                .otherwise("eval")))

        eval_df = (test_eval_split.filter(f.col("data_type") == "eval")
                            .join(test_eval, on="participant_id") # inner join removes all rows other than train
                            .drop("data_type","rand_val"))

        test_df = (test_eval_split.filter(f.col("data_type") == "test")
                           .join(test_eval, on="participant_id")
                           .drop("data_type","rand_val"))
        
        test_df.printSchema()
        dfs_to_write = [train_df,eval_df,test_df]
        prefixes = ["train","eval","test"]

        for df,prefix in zip(dfs_to_write,prefixes):
            path = os.path.join(out_path,prefix)
            df.write \
                .mode('overwrite') \
                .parquet(path, partitionBy="date")
            df.unpersist(blocking = True)

if __name__ == "__main__":
    main()