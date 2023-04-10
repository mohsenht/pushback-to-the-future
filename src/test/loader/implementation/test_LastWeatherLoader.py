from unittest import TestCase

import pandas as pd

from src.clean.TypeContainer import TypeContainer
from src.constants import SEPARATOR, LAMP_COLUMN_TEMPERATURE, LAMP_COLUMN_FORECAST_TIMESTAMP, \
    LAMP_COLUMN_WIND_DIRECTION, \
    LAMP_COLUMN_WIND_SPEED, LAMP_COLUMN_WIND_GUST, LAMP_COLUMN_CLOUD_CEILING, LAMP_COLUMN_VISIBILITY, LAMP_COLUMN_CLOUD, \
    LAMP_COLUMN_LIGHTNING_PROB, LAMP_COLUMN_PRECIP, LAMP_COLUMN_TIMESTAMP, COLUMN_NAME_TIMESTAMP
from src.loader.implementation.LastWeatherExtractor import LastWeatherExtractor
from src.model.Input import Input
from src.path_generator_utility import types_path_generator
from src.prediction.utility import crop_data_in_30h
from src.test.constants import AIRPORT, PROJECT_PATH


class TestLastWeatherLoader(TestCase):

    def test_last_weather_when_data_and_input_are_empty(self):
        now = pd.Timestamp.now()
        lamp_columns = [
            LAMP_COLUMN_TIMESTAMP,
            LAMP_COLUMN_FORECAST_TIMESTAMP,
            LAMP_COLUMN_TEMPERATURE,
            LAMP_COLUMN_WIND_DIRECTION,
            LAMP_COLUMN_WIND_SPEED,
            LAMP_COLUMN_WIND_GUST,
            LAMP_COLUMN_CLOUD_CEILING,
            LAMP_COLUMN_VISIBILITY,
            LAMP_COLUMN_CLOUD,
            LAMP_COLUMN_LIGHTNING_PROB,
            LAMP_COLUMN_PRECIP
        ]
        input_data = Input(
            config=pd.DataFrame(),
            etd=pd.DataFrame(),
            first_position=pd.DataFrame(),
            lamp=pd.DataFrame(columns=lamp_columns),
            mfs=pd.DataFrame(),
            runways=pd.DataFrame(),
            standtimes=pd.DataFrame(),
            tbfm=pd.DataFrame(),
            tfm=pd.DataFrame(),
        )
        data_columns = ['last_etd']
        data = pd.DataFrame(columns=data_columns)
        type_container = TypeContainer.from_file(f"{PROJECT_PATH}{types_path_generator(AIRPORT)}")

        loaded_data = LastWeatherExtractor().load_data(now, data, input_data, type_container)

        assert loaded_data.empty, "loaded_data is not empty"

    def test_data_is_empty_but_last_weather_has_data(self):
        now = pd.Timestamp('2020-11-08 05:30:00')
        lamp = pd.read_csv(
            f"data{SEPARATOR}weather{SEPARATOR}lamp.csv",
            parse_dates=[COLUMN_NAME_TIMESTAMP]
        ).sort_values(COLUMN_NAME_TIMESTAMP)
        input_data = Input(
            config=pd.DataFrame(),
            etd=pd.DataFrame(),
            first_position=pd.DataFrame(),
            lamp=crop_data_in_30h(now, lamp),
            mfs=pd.DataFrame(),
            runways=pd.DataFrame(),
            standtimes=pd.DataFrame(),
            tbfm=pd.DataFrame(),
            tfm=pd.DataFrame(),
        )
        data_columns = ['last_etd']
        data = pd.DataFrame(columns=data_columns)
        type_container = TypeContainer.from_file(f"{PROJECT_PATH}{types_path_generator(AIRPORT)}")

        loaded_data = LastWeatherExtractor().load_data(now, data, input_data, type_container)

        assert loaded_data.empty, "loaded_data is not empty"

    def test_data_and_weather_loader(self):
        now = pd.Timestamp('2020-11-08 05:30:00')
        lamp = pd.read_csv(
            f"data{SEPARATOR}weather{SEPARATOR}lamp.csv",
            parse_dates=[COLUMN_NAME_TIMESTAMP]
        ).sort_values(COLUMN_NAME_TIMESTAMP)
        data = pd.read_csv(
            f"data{SEPARATOR}weather{SEPARATOR}data.csv",
            parse_dates=[COLUMN_NAME_TIMESTAMP]
        ).sort_values(COLUMN_NAME_TIMESTAMP)
        input_data = Input(
            config=pd.DataFrame(),
            etd=pd.DataFrame(),
            first_position=pd.DataFrame(),
            lamp=crop_data_in_30h(now, lamp),
            mfs=pd.DataFrame(),
            runways=pd.DataFrame(),
            standtimes=pd.DataFrame(),
            tbfm=pd.DataFrame(),
            tfm=pd.DataFrame(),
        )
        type_container = TypeContainer.from_file(f"{PROJECT_PATH}{types_path_generator(AIRPORT)}")

        actual_loaded_data = LastWeatherExtractor().load_data(now, data, input_data, type_container)

        expected_loaded_data = pd.read_csv(
            f"data{SEPARATOR}weather{SEPARATOR}expected_data.csv",
            parse_dates=[COLUMN_NAME_TIMESTAMP]
        ).sort_values(COLUMN_NAME_TIMESTAMP)

        assert not actual_loaded_data.empty, "loaded_data is empty"
        assert actual_loaded_data.equals(expected_loaded_data)
