from unittest import TestCase
import pandas as pd

from constants import separator
from loader.implementation.LastWeatherExtractor import LastWeatherExtractor


class TestLastWeatherLoader(TestCase):

    def setUp(self):
        self.data = pd.read_csv(f"data{separator}result_2_days.csv", parse_dates=["timestamp"])
        self.now = self.data.iloc[0].timestamp

    def test_load_data(self):
        lamp_path = f"data{separator}lamp_2_days.csv"
        data = LastWeatherExtractor(lamp_path).load_data(self.now, self.data)
        self.fail()
