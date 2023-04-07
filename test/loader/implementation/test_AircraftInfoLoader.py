from unittest import TestCase
import pandas as pd

from constants import separator, file_name_mfs
from loader.implementation.AircraftInfoExtractor import AircraftInfoExtractor
from path_generator_utility import types_path_generator, path_generator


class TestAircraftInfoLoader(TestCase):

    def setUp(self):
        self.data = pd.read_csv(f"data{separator}result_2_days.csv", parse_dates=["timestamp"])
        self.now = self.data.iloc[0].timestamp

    def test_load_data(self):
        mfs_path = path_generator("KCLT", file_name_mfs)
        type_file_path = types_path_generator("KCLT")
        test_result = AircraftInfoExtractor(mfs_path, type_file_path).load_data(self.now, self.data)