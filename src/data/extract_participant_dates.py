import click

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

from pyspark.sql import functions as f
from pyspark.sql.functions import col

from pyspark import SparkContext

SPARK_CONFIG = [ 
    ("spark.master","local[95]"),
    ("spark.ui.port","4050"),
    ("spark.executor.memory","750g"),
    ('spark.driver.memory',  '2000g'),
    ("spark.driver.maxResultSize", '0'), # unlimited
    ("spark.network.timeout",            "10000001"),
    ("spark.executor.heartbeatInterval", "10000000")]   


@click.command()
@click.argument("input_path", type=click.Path(file_okay=False,exists=True))
@click.option("-o","--output_path", type=click.Path(file_okay=False,exists=False))
def main(input_path, output_path=None):
    """
    Takes as input a spark minute dataset 
    and generates a csv where the first
    column is the participant id and the second column
    is the date in YYYY-MM-DD format.
    """
    conf = SparkConf().setAll(SPARK_CONFIG)
    sc = SparkContext(conf=conf, appName="PetaStorm Conversion")
    spark = SparkSession(sc)

    df = spark.read.parquet(input_path)
    df = (df.select("participant_id","date")
                           .distinct())
    if output_path:
        df.toPandas().to_csv(output_path,index=False)
    else:
        print("participant_id","date")
        for row in df.collect():
            d = row.asDict()
            participant_id = d['participant_id']
            date = d["date"]
            print(f"{participant_id},{date}")

if __name__ == "__main__":
    main()
