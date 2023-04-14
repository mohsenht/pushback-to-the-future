import os
from unittest import TestCase

import pandas as pd

from src.constants import COLUMN_NAME_TIMESTAMP, SEPARATOR
from src.prediction.CSVReader import CSVReader


class TestCSVReader(TestCase):

    def test_csv_reader(self):
        print(os.getcwd())
        csv_reader = CSVReader(f"data{SEPARATOR}CSVReader{SEPARATOR}etd.csv", 4, [COLUMN_NAME_TIMESTAMP])
        timestamps = pd.read_csv(
            f"data{SEPARATOR}CSVReader{SEPARATOR}timestamp.csv",
            parse_dates=[
                'start_time',
                'end_time'
            ]
        )

        for index, timestamp in timestamps.iterrows():
            data = csv_reader.get_data(
                timestamp['start_time'],
                timestamp['end_time']
            )
        csv_reader.close_file()
