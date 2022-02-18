"""
========================
Model Training Utilities 
========================
`Project repository available here  <https://github.com/behavioral-data/SeattleFluStudy>`_

The main() function of the module takes care of saving a transformed Parquet dataset given one in input. 
Transformations include scaling and grouping of data to time windows of specified size. 


"""
from os import name
import numpy as np
import click

from pyarrow.parquet import ParquetDataset
import pyspark

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

import pyspark.sql.types as sql_types
from pyspark.sql import functions as f
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark import SparkContext

from petastorm.codecs import ScalarCodec, CompressedImageCodec, NdarrayCodec
from petastorm.etl.dataset_metadata import materialize_dataset, get_schema_from_dataset_url
from petastorm.unischema import dict_to_spark_row, Unischema, UnischemaField, _numpy_to_spark_mapping

import pandas as pd

MINS_IN_DAY = 60*24


from contextlib import contextmanager
from pyspark.sql import SparkSession

@contextmanager
def spark_timezone(timezone: str):
    """Context manager to temporarily set spark timezone during context manager
    life time while preserving original timezone. This is especially
    meaningful in conjunction with casting timestamps when automatic timezone
    conversions are applied by spark.

    Please be aware that the timezone property should be adjusted during DAG
    creation and execution (including both spark transformations and actions).
    Changing the timezone while adding filter/map tasks might not be
    sufficient. Be sure to change the timezone when actually executing a spark
    action like collect/save etc.

    Parameters
    ----------
    timezone: str
        Name of the timezone (e.g. 'UTC' or 'Europe/Berlin').

    Examples
    --------
    >>> with spark_timezone("Europe/Berlin"):
    >>>     df.select(df["ts_col"].cast("timestamp")).show()

    """

    spark = get_active_spark_context()
    current = spark.conf.get("spark.sql.session.timeZone")
    spark.conf.set("spark.sql.session.timeZone", timezone)

    try:
        yield None
    finally:
        spark.conf.set("spark.sql.session.timeZone", current)


def get_active_spark_context() -> SparkSession:
    """Helper function to return the currently active spark context.

    """

    return SparkSession.builder.getOrCreate()


def lr_schmea(schema):
    new_fields = []
    for field in schema.fields.values():
        old_name = field.name
        shape = field.shape
        np_dtype = field.numpy_dtype
        if np_dtype is np.float64:
            np_dtype = np.float32

        new_fields.append(UnischemaField(old_name+"_l",np_dtype,nullable=False,shape=shape))
        new_fields.append(UnischemaField(old_name+"_r",np_dtype,nullable=False,shape=shape))
    
    return Unischema("homekit",new_fields)
        

@click.command()
@click.argument("input_path", type=click.Path(file_okay=True,exists=True))
@click.argument("output_path", type=click.Path(file_okay=False,exists=False))
@click.option("--task",type=click.Choice(["same_participant","sequential"]), default="same_participant")
@click.option("--sequential_offset", default=7, help="Number of days to offset the following window")
def main(input_path, output_path, task="same_participant",sequential_offset=7):
    configuation_properties = [
    ("spark.master","local[64]"),
    ("spark.ui.port","4050"),
    ("spark.executor.memory","128g"),
    ('spark.driver.memory',  '2000g'),
    ("spark.driver.maxResultSize", '0'), # unlimited
    ("spark.network.timeout",            "10000001"),
    ("spark.executor.heartbeatInterval", "10000000")]   
    
    conf = SparkConf().setAll(configuation_properties)
    sc = SparkContext(conf=conf, appName="PetaStorm Conversion")
    spark = SparkSession(sc)
                
    if not "file://" in output_path:
        output_path = "file://"+ output_path
    
    if not "file://" in input_path:
        input_url = "file://"+ input_path
    else:
        input_url = input_path
    
    input = spark.read.parquet(input_path)
    input_schema = get_schema_from_dataset_url(input_url)
    rowgroup_size_mb = 128
    schema = lr_schmea(input_schema)
    with materialize_dataset(spark, output_path, schema, rowgroup_size_mb):

        MAX_NEG=1e6
        
        if task == "same_participant":



            
            input_r = input.select(*(f.col(x).alias(x + '_r') for x in input.columns))
            input_l = input.select(*(f.col(x).alias(x + '_l') for x in input.columns))
            

            diff_participant_r  = input_r.orderBy(f.rand()).withColumn('row_id', f.monotonically_increasing_id())
            diff_participant_l  = input_l.withColumn('row_id', f.monotonically_increasing_id())
            diff_participant  =  diff_participant_l.join(diff_participant_r,on="row_id").drop("row_id")
            for _i in range(10):
                diff_participant_r  = input_r.orderBy(f.rand()).withColumn('row_id', f.monotonically_increasing_id())
                diff_participant_l  = input_l.withColumn('row_id', f.monotonically_increasing_id())
                diff_participant_prime  =  diff_participant_l.join(diff_participant_r,on="row_id").drop("row_id")
                diff_participant = diff_participant.union(diff_participant_prime)

            # Would be good to check number of unique particpants in each group
            same_participant =  input_l.join(input_r,input_r["participant_id_r"] == input_l["participant_id_l"])\
                                                .limit(int(1e6))

            output = diff_participant.union(same_participant).orderBy(f.rand())
            
                
        elif task == "sequential":
            split_size = input.count() // 2
            df = input.withColumn("end_delta",input['end'] - f.expr(f'INTERVAL {sequential_offset} DAYS')).orderBy(f.rand())
            
            df_r = df.select(*(f.col(x).alias(x + '_r') for x in df.columns))
            df_l = df.select(*(f.col(x).alias(x + '_l') for x in df.columns))
            
            pos, neg = df_l.join(df_r, (df_l["end_l"] == df_r["end_delta_r"]) &
                                        (df_l["participant_id_l"] == df_r["participant_id_r"])).randomSplit([0.5,0.5])
            
            neg_l = neg.select(*(x for x in neg.columns if x[-2:]=="_l"))
            neg_r = neg.select(*(x for x in neg.columns if x[-2:]=="_r"))
            
            neg_l  = neg_l.orderBy(f.rand()).withColumn('row_id', f.monotonically_increasing_id())
            neg_r  = neg_r.withColumn('row_id', f.monotonically_increasing_id())
            neg = neg_l.join(neg_r,on="row_id").drop("row_id")
            
            output = pos.union(neg).orderBy(f.rand())

        output.write \
                .mode('overwrite') \
                .parquet(output_path,mode="overwrite")

def rename_columns(df, columns):
    if isinstance(columns, dict):
        for old_name, new_name in columns.items():
            df = df.withColumnRenamed(old_name, new_name)
        return df
    else:
        raise ValueError("'columns' should be a dict, like {'old_name_1':'new_name_1', 'old_name_2':'new_name_2'}")


def _field_spark_dtype(field):
    mapping = _numpy_to_spark_mapping()
    mapping[np.datetime64] = sql_types.TimestampType()
    #Cast doubles to floats since pytorch uses float tensors
    mapping[np.float64] = sql_types.FloatType()

    if field.codec is None:
        if field.shape == ():
            spark_type = mapping.get(field.numpy_dtype, None)
            if not spark_type:
                raise ValueError('Was not able to map type {} to a spark type.'.format(str(field.numpy_dtype)))
        else:
            raise ValueError('An instance of non-scalar UnischemaField \'{}\' has codec set to None. '
                             'Don\'t know how to guess a Spark type for it'.format(field.name))
    else:
        spark_type = field.codec.spark_dtype()

    return spark_type

def get_spark_schema(schema):
    
    schema_entries = []
    for field in schema._fields.values():
        spark_type = _field_spark_dtype(field)
        schema_entries.append(sql_types.StructField(field.name, spark_type, field.nullable))

    return sql_types.StructType(schema_entries)
    

if __name__ == "__main__":
    main()