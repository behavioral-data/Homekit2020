import os

import click
from src.data.make_dataset import lab_results, lab_updates, process_surveys, process_minute_level, process_day_level
from src.utils import get_logger
logger = get_logger()

@click.command()
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    os.environ["DEBUG_DATA"] = "1"
    logger.info('making debug data set from raw data')

    # Lab updates:
    lab_updates()
    results = lab_results(return_result=True)
    pos_participant_ids = results[results["result"] == 'Detected']["participant_id"].sample(10)

    # Surveys
    process_surveys()

    # Minute Level Activity
    process_minute_level(participant_ids=pos_participant_ids)
    process_day_level(participant_ids=pos_participant_ids)

if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
