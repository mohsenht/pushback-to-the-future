from unittest import TestCase

import cudf

from src.clean.TypeContainer import TypeContainer
from src.constants import SEPARATOR, COLUMN_NAME_TIMESTAMP, ETD_COLUMN_TIMESTAMP, \
    ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME
from src.loader.implementation.LastETDExtractor import LastETDExtractor
from src.model.Input import Input
from src.path_generator_utility import types_path_generator
from src.test.constants import PROJECT_PATH, AIRPORT


class TestLastETDExtractor(TestCase):

    def test_load_data_with_empty_edt(self):
        now = cudf.Timestamp('2020-11-08 05:30:00')

        data = cudf.read_csv(
            f"data{SEPARATOR}LastETDExtractor{SEPARATOR}data.csv",
            parse_dates=[
                COLUMN_NAME_TIMESTAMP
            ]
        ).sort_values(COLUMN_NAME_TIMESTAMP)

        etd = cudf.read_csv(
            f"data{SEPARATOR}LastETDExtractor{SEPARATOR}empty_etd.csv",
            parse_dates=[
                ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME,
                ETD_COLUMN_TIMESTAMP
            ],
        ).sort_values(ETD_COLUMN_TIMESTAMP)

        input_data = Input(
            config=cudf.DataFrame(),
            etd=etd,
            first_position=cudf.DataFrame(),
            lamp=cudf.DataFrame(),
            mfs=cudf.DataFrame(),
            runways=cudf.DataFrame(),
            standtimes=cudf.DataFrame(),
            tbfm=cudf.DataFrame(),
            tfm=cudf.DataFrame(),
        )
        type_container = TypeContainer.from_file(f"{PROJECT_PATH}{types_path_generator(AIRPORT)}")

        loaded_data = LastETDExtractor().load_data(
            now,
            data,
            input_data,
            type_container
        )

        assert not loaded_data.empty, "loaded_data is empty"

    def test_load_data_with_regular_input(self):
        now = cudf.Timestamp('2020-11-08 05:30:00')
        data = cudf.read_csv(
            f"data{SEPARATOR}LastETDExtractor{SEPARATOR}data.csv",
            parse_dates=[COLUMN_NAME_TIMESTAMP]
        ).sort_values(COLUMN_NAME_TIMESTAMP)

        etd = cudf.read_csv(
            f"data{SEPARATOR}LastETDExtractor{SEPARATOR}etd.csv",
            parse_dates=[
                ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME,
                ETD_COLUMN_TIMESTAMP
            ],
        ).sort_values(ETD_COLUMN_TIMESTAMP)

        input_data = Input(
            config=cudf.DataFrame(),
            etd=etd,
            first_position=cudf.DataFrame(),
            lamp=cudf.DataFrame(),
            mfs=cudf.DataFrame(),
            runways=cudf.DataFrame(),
            standtimes=cudf.DataFrame(),
            tbfm=cudf.DataFrame(),
            tfm=cudf.DataFrame(),
        )
        type_container = TypeContainer.from_file(f"{PROJECT_PATH}{types_path_generator(AIRPORT)}")

        loaded_data = LastETDExtractor().load_data(
            now,
            data,
            input_data,
            type_container
        )

        expected_loaded_data = cudf.read_csv(
            f"data{SEPARATOR}LastETDExtractor{SEPARATOR}expected_data.csv",
            parse_dates=[COLUMN_NAME_TIMESTAMP]
        ).sort_values(COLUMN_NAME_TIMESTAMP)
        expected_loaded_data["last_etd"] = expected_loaded_data.last_etd.clip(lower=0).astype(int)

        assert not loaded_data.empty, "loaded_data is empty"
        assert loaded_data.equals(expected_loaded_data)
