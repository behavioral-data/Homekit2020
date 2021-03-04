import os

import click
from src.data.make_dataset import lab_results, lab_updates, process_surveys, process_minute_level
from src.utils import get_logger
logger = get_logger()

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    os.environ["DEBUG_DATA"] = "1"
    logger.info('making debug data set from raw data')

    # Lab updates:
    lab_updates(output_filepath)
    results = lab_results(output_filepath,return_result=True)
    pos_participant_ids = results[results["result"] == 'Detected']["participant_id"].sample(10)

    # Surveys
    process_surveys(output_filepath)

    # Minute Level Activity
    process_minute_level(output_filepath,participant_ids=pos_participant_ids)

if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
