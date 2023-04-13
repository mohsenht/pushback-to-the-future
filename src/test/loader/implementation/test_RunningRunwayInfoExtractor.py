from unittest import TestCase

import cudf
import pandas as pd

from src.clean.TypeContainer import TypeContainer
from src.constants import SEPARATOR, COLUMN_NAME_TIMESTAMP, ETD_COLUMN_TIMESTAMP, \
    ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME
from src.loader.implementation.LastETDExtractor import LastETDExtractor
from src.loader.implementation.RunningRunwayInfoExtractor import RunningRunwayInfoExtractor
from src.model.Input import Input
from src.path_generator_utility import types_path_generator
from src.test.constants import PROJECT_PATH, AIRPORT


class TestRunningRunwayInfoExtractor(TestCase):

    def test_load_data_with_regular_input(self):
        expected = pd.read_csv("/home/sara/expected.csv")
        actual = pd.read_csv("/home/sara/actual.csv")
        assert actual.equals(expected)
        now = pd.Timestamp('2020-11-08 05:30:00')
        data = cudf.read_csv(
            f"data{SEPARATOR}RunningRunwayInfoExtractor{SEPARATOR}data.csv",
            parse_dates=[COLUMN_NAME_TIMESTAMP]
        ).sort_values(COLUMN_NAME_TIMESTAMP)

        config = cudf.read_csv(
            f"data{SEPARATOR}RunningRunwayInfoExtractor{SEPARATOR}config.csv",
            parse_dates=[
                ETD_COLUMN_TIMESTAMP
            ],
        ).sort_values(ETD_COLUMN_TIMESTAMP)

        input_data = Input(
            config=config,
            etd=cudf.DataFrame(),
            first_position=cudf.DataFrame(),
            lamp=cudf.DataFrame(),
            mfs=cudf.DataFrame(),
            runways=cudf.DataFrame(),
            standtimes=cudf.DataFrame(),
            tbfm=cudf.DataFrame(),
            tfm=cudf.DataFrame(),
        )
        type_container = TypeContainer.from_file(f"{PROJECT_PATH}{types_path_generator(AIRPORT)}")

        loaded_data = RunningRunwayInfoExtractor().load_data(
            now,
            data,
            input_data,
            type_container
        )

        expected_loaded_data = pd.read_csv(
            f"data{SEPARATOR}RunningRunwayInfoExtractor{SEPARATOR}expected_data.csv"
        )

        assert not loaded_data.empty, "loaded_data is empty"
        loaded_data.to_csv(f"data{SEPARATOR}RunningRunwayInfoExtractor{SEPARATOR}loaded_data.csv", index=False)
        loaded_data = pd.read_csv(f"data{SEPARATOR}RunningRunwayInfoExtractor{SEPARATOR}loaded_data.csv")
        assert loaded_data.equals(expected_loaded_data)
