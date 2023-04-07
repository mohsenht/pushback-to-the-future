from unittest import TestCase
import pandas as pd

from constants import SEPARATOR, FILE_NAME_MFS
from loader.implementation.AircraftInfoExtractor import AircraftInfoExtractor
from path_generator_utility import types_path_generator, path_generator


class TestAircraftInfoLoader(TestCase):

    def setUp(self):
        self.data = pd.read_csv(f"data{SEPARATOR}result_2_days.csv", parse_dates=["timestamp"])
        self.now = self.data.iloc[0].timestamp

    def test_load_data(self):
        mfs_path = path_generator("KCLT", FILE_NAME_MFS)
        type_file_path = types_path_generator("KCLT")
        test_result = AircraftInfoExtractor(mfs_path, type_file_path).load_data(self.now, self.data)