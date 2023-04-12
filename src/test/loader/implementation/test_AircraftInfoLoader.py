from unittest import TestCase

import cudf
import pandas as pd

from src.clean.TypeContainer import TypeContainer
from src.constants import SEPARATOR, COLUMN_NAME_TIMESTAMP, MFS_COLUMN_FLIGHT_ID, MFS_COLUMN_AIRCRAFT_TYPE, \
    MFS_COLUMN_AIRCRAFT_ENGINE_CLASS, MFS_COLUMN_MAJOR_CARRIER, MFS_COLUMN_FLIGHT_TYPE, MFS_COLUMN_IS_DEPARTURE
from src.loader.implementation.AircraftInfoExtractor import AircraftInfoExtractor
from src.model.Input import Input
from src.path_generator_utility import types_path_generator
from src.test.constants import PROJECT_PATH, AIRPORT


class TestAircraftInfoLoader(TestCase):

    def test_load_data_with_empty_mfs(self):
        now = pd.Timestamp('2020-11-08 05:30:00')
        data = cudf.read_csv(
            f"data{SEPARATOR}AircraftInfoExtractor{SEPARATOR}data.csv",
            parse_dates=[COLUMN_NAME_TIMESTAMP]
        ).sort_values(COLUMN_NAME_TIMESTAMP)

        mfs_columns = [
            MFS_COLUMN_FLIGHT_ID,
            MFS_COLUMN_AIRCRAFT_ENGINE_CLASS,
            MFS_COLUMN_AIRCRAFT_TYPE,
            MFS_COLUMN_MAJOR_CARRIER,
            MFS_COLUMN_FLIGHT_TYPE,
            MFS_COLUMN_IS_DEPARTURE
        ]

        input_data = Input(
            config=cudf.DataFrame(),
            etd=cudf.DataFrame(),
            first_position=cudf.DataFrame(),
            lamp=cudf.DataFrame(),
            mfs=cudf.DataFrame(columns=mfs_columns),
            runways=cudf.DataFrame(),
            standtimes=cudf.DataFrame(),
            tbfm=cudf.DataFrame(),
            tfm=cudf.DataFrame(),
        )
        type_container = TypeContainer.from_file(f"{PROJECT_PATH}{types_path_generator(AIRPORT)}")

        loaded_data = AircraftInfoExtractor().load_data(
            now,
            data,
            input_data,
            type_container
        )

        mfs_features = [
            MFS_COLUMN_AIRCRAFT_TYPE,
            MFS_COLUMN_AIRCRAFT_ENGINE_CLASS,
            MFS_COLUMN_MAJOR_CARRIER,
            MFS_COLUMN_FLIGHT_TYPE
        ]

        assert not loaded_data.empty, "loaded_data is empty"
        assert all(col in loaded_data.columns for col in mfs_features)

    def test_load_data_with_regular_input(self):
        now = pd.Timestamp('2020-11-08 05:30:00')
        data = cudf.read_csv(
            f"data{SEPARATOR}AircraftInfoExtractor{SEPARATOR}data.csv",
            parse_dates=[COLUMN_NAME_TIMESTAMP]
        ).sort_values(COLUMN_NAME_TIMESTAMP)

        mfs = cudf.read_csv(
            f"data{SEPARATOR}AircraftInfoExtractor{SEPARATOR}mfs.csv"
        )

        input_data = Input(
            config=cudf.DataFrame(),
            etd=cudf.DataFrame(),
            first_position=cudf.DataFrame(),
            lamp=cudf.DataFrame(),
            mfs=mfs,
            runways=cudf.DataFrame(),
            standtimes=cudf.DataFrame(),
            tbfm=cudf.DataFrame(),
            tfm=cudf.DataFrame(),
        )
        type_container = TypeContainer.from_file(f"{PROJECT_PATH}{types_path_generator(AIRPORT)}")

        loaded_data = AircraftInfoExtractor().load_data(
            now,
            data,
            input_data,
            type_container
        )

        expected_loaded_data = cudf.read_csv(
            f"data{SEPARATOR}AircraftInfoExtractor{SEPARATOR}expected_data.csv",
            parse_dates=[COLUMN_NAME_TIMESTAMP]
        ).sort_values(COLUMN_NAME_TIMESTAMP)

        assert not loaded_data.empty, "loaded_data is empty"
        assert loaded_data.equals(expected_loaded_data)
