from unittest import TestCase

import pandas as pd

from clean.TypeContainer import TypeContainer
from constants import SEPARATOR, FEATURE_COLUMN_WEIGHTED_MEAN_ARRIVAL_TO_GATE
from loader.implementation.ArrivalToGateTimeWeightedMeanExtractor import ArrivalToGateTimeWeightedMeanExtractor
from model.Input import Input
from path_generator_utility import types_path_generator
from prediction.utility import crop_data_in_30h
from test.constants import PROJECT_PATH, AIRPORT


class TestArrivalToGateTimeWeightedMeanExtractor(TestCase):
    def test_load_data(self):
        now = pd.Timestamp('2020-11-08 05:30:00')
        data = pd.read_csv(
            f"data{SEPARATOR}ArrivalToGateTimeWeightedMeanExtractor{SEPARATOR}data.csv",
            parse_dates=["timestamp"]
        ).sort_values("timestamp")
        runways = pd.read_csv(
            f"data{SEPARATOR}ArrivalToGateTimeWeightedMeanExtractor{SEPARATOR}runways.csv",
            parse_dates=["arrival_runway_actual_time", "timestamp"],
        ).sort_values("timestamp")
        standtimes = pd.read_csv(
            f"data{SEPARATOR}ArrivalToGateTimeWeightedMeanExtractor{SEPARATOR}standtimes.csv",
            parse_dates=["arrival_stand_actual_time", "timestamp"],
        ).sort_values("timestamp")
        input_data = Input(
            config=pd.DataFrame(),
            etd=pd.DataFrame(),
            first_position=pd.DataFrame(),
            lamp=pd.DataFrame(),
            mfs=pd.DataFrame(),
            runways=crop_data_in_30h(now, runways),
            standtimes=crop_data_in_30h(now, standtimes),
            tbfm=pd.DataFrame(),
            tfm=pd.DataFrame(),
        )
        type_container = TypeContainer.from_file(f"{PROJECT_PATH}{types_path_generator(AIRPORT)}")

        loaded_data = ArrivalToGateTimeWeightedMeanExtractor().load_data(now, data, input_data,
                                                                         type_container)

        assert not loaded_data.empty, "loaded_data is not empty"
        assert hasattr(loaded_data, FEATURE_COLUMN_WEIGHTED_MEAN_ARRIVAL_TO_GATE), "loaded_data does not have column ''"
        assert loaded_data.iloc[-1][FEATURE_COLUMN_WEIGHTED_MEAN_ARRIVAL_TO_GATE] != 0
