from unittest import TestCase
import pandas as pd

from loader.implementation.AircraftInfoLoader import AircraftInfoLoader


class TestAircraftInfoLoader(TestCase):

    def setUp(self):
        self.data = pd.read_csv('data\\result_2_days.csv', parse_dates=["timestamp"])
        self.now = self.data.iloc[0].timestamp

    def test_load_data(self):
        mfs_path = 'D:\\competetion\\KCLT\\KCLT_mfs.csv'
        type_file_path = 'D:\\projects\\pushbackToTheFuture\\pushback\\data\\model\\KCLT\\types.json'
        asghar = AircraftInfoLoader(mfs_path, type_file_path).load_data(self.now, self.data)

