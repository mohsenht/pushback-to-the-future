from unittest import TestCase
import pandas as pd

from constants import separator
from loader.implementation.AircraftInfoLoader import AircraftInfoLoader


class TestAircraftInfoLoader(TestCase):

    def setUp(self):
        self.data = pd.read_csv(f"data{separator}result_2_days.csv", parse_dates=["timestamp"])
        self.now = self.data.iloc[0].timestamp

    def test_load_data(self):
        mfs_path = f"data{separator}train{separator}KCLT{separator}KCLT_mfs.csv"
        type_file_path = f"data{separator}model{separator}KCLT{separator}types.json"
        asghar = AircraftInfoLoader(mfs_path, type_file_path).load_data(self.now, self.data)