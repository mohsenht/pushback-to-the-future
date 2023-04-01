from unittest import TestCase
import pandas as pd

from loader.implementation.LastWeatherLoader import LastWeatherLoader


class TestLastWeatherLoader(TestCase):

    def setUp(self):
        self.data = pd.read_csv('data\\result_2_days.csv', parse_dates=["timestamp"])
        self.now = self.data.iloc[0].timestamp

    def test_load_data(self):
        lamp_path = 'data\\lamp_2_days.csv'
        data = LastWeatherLoader(lamp_path).load_data(self.now, self.data)
        self.fail()
