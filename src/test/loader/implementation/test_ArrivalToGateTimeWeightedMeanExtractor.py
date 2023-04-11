from unittest import TestCase

import cudf

from src.clean.TypeContainer import TypeContainer
from src.constants import SEPARATOR, FEATURE_COLUMN_WEIGHTED_MEAN_ARRIVAL_TO_GATE, \
    RUNWAYS_COLUMN_ARRIVAL_RUNWAY_ACTUAL_TIME, STANDTIMES_COLUMN_ARRIVAL_STAND_ACTUAL_TIME, \
    COLUMN_NAME_TIMESTAMP
from src.hyper_parameters import ARRIVAL_TO_GATE_WEIGHTED_MEAN_TIME_CONSTANT_FOR_EMPTY_ARRIVAL
from src.loader.implementation.ArrivalToGateTimeWeightedMeanExtractor import ArrivalToGateTimeWeightedMeanExtractor
from src.model.Input import Input
from src.path_generator_utility import types_path_generator
from src.prediction.utility import crop_data_in_30h
from src.test.constants import PROJECT_PATH, AIRPORT


class TestArrivalToGateTimeWeightedMeanExtractor(TestCase):
    def test_load_data(self):
        now = cudf.Timestamp('2020-11-08 05:30:00')
        data = cudf.read_csv(
            f"data{SEPARATOR}ArrivalToGateTimeWeightedMeanExtractor{SEPARATOR}data.csv",
            parse_dates=[COLUMN_NAME_TIMESTAMP]
        ).sort_values(COLUMN_NAME_TIMESTAMP)
        runways = cudf.read_csv(
            f"data{SEPARATOR}ArrivalToGateTimeWeightedMeanExtractor{SEPARATOR}runways.csv",
            parse_dates=[RUNWAYS_COLUMN_ARRIVAL_RUNWAY_ACTUAL_TIME, COLUMN_NAME_TIMESTAMP],
        ).sort_values(COLUMN_NAME_TIMESTAMP)
        standtimes = cudf.read_csv(
            f"data{SEPARATOR}ArrivalToGateTimeWeightedMeanExtractor{SEPARATOR}standtimes.csv",
            parse_dates=[STANDTIMES_COLUMN_ARRIVAL_STAND_ACTUAL_TIME, COLUMN_NAME_TIMESTAMP],
        ).sort_values(COLUMN_NAME_TIMESTAMP)
        input_data = Input(
            config=cudf.DataFrame(),
            etd=cudf.DataFrame(),
            first_position=cudf.DataFrame(),
            lamp=cudf.DataFrame(),
            mfs=cudf.DataFrame(),
            runways=crop_data_in_30h(now, runways),
            standtimes=crop_data_in_30h(now, standtimes),
            tbfm=cudf.DataFrame(),
            tfm=cudf.DataFrame(),
        )
        type_container = TypeContainer.from_file(f"{PROJECT_PATH}{types_path_generator(AIRPORT)}")

        loaded_data = ArrivalToGateTimeWeightedMeanExtractor().load_data(now, data, input_data,
                                                                         type_container)

        assert not loaded_data.empty, "loaded_data is empty"
        assert hasattr(loaded_data, FEATURE_COLUMN_WEIGHTED_MEAN_ARRIVAL_TO_GATE), "loaded_data does not have column ''"
        assert loaded_data.iloc[-1][FEATURE_COLUMN_WEIGHTED_MEAN_ARRIVAL_TO_GATE] != 0

    def test_load_data_without_any_arrival_to_gate_data(self):
        now = cudf.Timestamp('2020-11-08 05:30:00')
        data = cudf.read_csv(
            f"data{SEPARATOR}ArrivalToGateTimeWeightedMeanExtractor{SEPARATOR}data.csv",
            parse_dates=[COLUMN_NAME_TIMESTAMP]
        ).sort_values(COLUMN_NAME_TIMESTAMP)
        runways = cudf.read_csv(
            f"data{SEPARATOR}ArrivalToGateTimeWeightedMeanExtractor{SEPARATOR}one_item_runways.csv",
            parse_dates=[RUNWAYS_COLUMN_ARRIVAL_RUNWAY_ACTUAL_TIME, COLUMN_NAME_TIMESTAMP],
        ).sort_values(COLUMN_NAME_TIMESTAMP)
        standtimes = cudf.read_csv(
            f"data{SEPARATOR}ArrivalToGateTimeWeightedMeanExtractor{SEPARATOR}one_item_standtimes.csv",
            parse_dates=[STANDTIMES_COLUMN_ARRIVAL_STAND_ACTUAL_TIME, COLUMN_NAME_TIMESTAMP],
        ).sort_values(COLUMN_NAME_TIMESTAMP)

        input_data = Input(
            config=cudf.DataFrame(),
            etd=cudf.DataFrame(),
            first_position=cudf.DataFrame(),
            lamp=cudf.DataFrame(),
            mfs=cudf.DataFrame(),
            runways=runways,
            standtimes=standtimes,
            tbfm=cudf.DataFrame(),
            tfm=cudf.DataFrame(),
        )
        type_container = TypeContainer.from_file(f"{PROJECT_PATH}{types_path_generator(AIRPORT)}")

        loaded_data = ArrivalToGateTimeWeightedMeanExtractor().load_data(
            now,
            data,
            input_data,
            type_container
        )

        assert not loaded_data.empty, "loaded_data is not empty"
        assert hasattr(loaded_data, FEATURE_COLUMN_WEIGHTED_MEAN_ARRIVAL_TO_GATE), "loaded_data does not have column ''"
        assert loaded_data.iloc[-1][
                   FEATURE_COLUMN_WEIGHTED_MEAN_ARRIVAL_TO_GATE
               ] == ARRIVAL_TO_GATE_WEIGHTED_MEAN_TIME_CONSTANT_FOR_EMPTY_ARRIVAL
