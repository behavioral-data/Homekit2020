from os import name
import numpy as np
import click

from pyarrow.parquet import ParquetDataset

from pyspark.sql import SparkSession
import pyspark.sql.types as sql_types
from pyspark.sql import functions as f
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler, VectorAssembler


from petastorm.codecs import ScalarCodec, CompressedImageCodec, NdarrayCodec
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import dict_to_spark_row, Unischema, UnischemaField, _numpy_to_spark_mapping

MINS_IN_DAY = 60*24

@click.command()
@click.argument("input_path", type=click.Path(file_okay=False,exists=True))
@click.argument("output_path", type=click.Path(file_okay=False,exists=False))
@click.option("--max_missing_days_in_window", type=int, default=2)
@click.option("--min_windows", type=int, default=1)
@click.option("--day_window_size", type=int, default=4)
def main(input_path, output_path, max_missing_days_in_window, 
                    min_windows, day_window_size):
                
    if not "file://" in output_path:
        output_path = "file://"+ output_path
    
    pyarrow_dataset = ParquetDataset(input_path)
    schema = Unischema.from_arrow_schema(pyarrow_dataset)
    
    expected_length = day_window_size*MINS_IN_DAY
    new_fields = []
    for field in schema.fields.values():
        name = field.name
        if name in ["date","timestamp"]:
            continue

        np_dtype = field.numpy_dtype
        if np_dtype is np.float64:
            new_fields.append(UnischemaField(name,np.float32,nullable=False,shape=(expected_length,)))
        else:
            new_fields.append(UnischemaField(name,np_dtype,nullable=False,shape=(expected_length,)))
        
    new_fields.append(UnischemaField("start",np.datetime64,nullable=False,shape=None))
    new_fields.append(UnischemaField("end",np.datetime64,nullable=False,shape=None))

    schema = Unischema("homekit",new_fields)
    rowgroup_size_mb = 256
    spark = SparkSession.builder.master("local[64]")\
                                .config("spark.executor.memory", "32g")\
                                .config("spark.driver.memory", "32g")\
                                .appName("Petastorm Conversion").getOrCreate()
    sc = spark.sparkContext

    # Wrap dataset materialization portion. Will take care of setting up spark environment variables as
    # well as save petastorm specific metadata
    with materialize_dataset(spark, output_path, schema, rowgroup_size_mb):

        # rows_rdd = \
        #     .map(row_generator)\
        #     .map(lambda x: dict_to_spark_row(schema, x))

        df = spark.read.parquet(input_path)
        dbl_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, sql_types.DoubleType)]
        non_dbl_cols = [f.name for f in df.schema.fields if not isinstance(f.dataType, sql_types.DoubleType)]
        # w = Window.partitionBy(input.id).orderBy(input.timestamp)
        assemblers = [VectorAssembler(inputCols=[col], outputCol=col + "_vec") for col in dbl_cols]
        scalers = [StandardScaler(inputCol=col + "_vec", outputCol=col + "_scaled") for col in dbl_cols]
        pipeline = Pipeline(stages=assemblers + scalers)
        scalerModel = pipeline.fit(df)
        scaledData = scalerModel.transform(df)
        
        names = {x + "_scaled": x for x in dbl_cols}
        firstelement=f.udf(lambda v:float(v[0]),sql_types.DoubleType())
        # df = df.select([firstelement(c).alias(c) for c in df.columns])
        scaled_cols = [firstelement(f.col(c)).alias(names[c]) for c in names.keys()]
        old_cols = [f.col(c) for c in non_dbl_cols]
        scaledData = scaledData.select(old_cols + scaled_cols)
       
        
        window_duration = f"{day_window_size} days"
        slide_duration = f"1 days"
        
        # w = Window.partitionBy(scaledData.participant_id,scaledData.date).orderBy(scaledData.timestamp)
        grouped = scaledData.groupBy("participant_id",f.window("timestamp", window_duration, 
                                                            slide_duration, startTime="0 minutes"))

        feature_columns = [x for x in scaledData.columns if not x in ["participant_id","timestamp","date"]]
        aggs = [f.collect_list(colName) for colName in feature_columns] + [f.count(feature_columns[0]).alias("count_col")]
        # vecAssembler = VectorAssembler(outputCol="features", inputCols=[f"collect_list({x})" for x in feature_columns])
        
        result = grouped.agg(*aggs)
        result  = result.filter(result.count_col == expected_length)
        result.drop("count_col")
        result = rename_columns(result,{f"collect_list({x})" : x for x in feature_columns})
        # result = result.withColumn()

        result = result.withColumn("start",result.window.start)
        result = result.withColumn("end",result.window.end)

        result.write \
            .mode('overwrite') \
            .parquet(output_path)

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