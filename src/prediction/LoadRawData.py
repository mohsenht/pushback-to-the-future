import cudf
import pandas as pd

from src.constants import FILE_NAME_ETD, FILE_NAME_RUNWAYS, FILE_NAME_STANDTIMES, FILE_NAME_LAMP, FILE_NAME_CONFIG, \
    FILE_NAME_MFS, LAMP_COLUMN_FORECAST_TIMESTAMP, STANDTIMES_COLUMN_ARRIVAL_STAND_ACTUAL_TIME, \
    RUNWAYS_COLUMN_ARRIVAL_RUNWAY_ACTUAL_TIME, ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME, ETD_COLUMN_TIMESTAMP, \
    RUNWAYS_COLUMN_TIMESTAMP, STANDTIMES_COLUMN_TIMESTAMP, LAMP_COLUMN_TIMESTAMP, CONFIG_COLUMN_TIMESTAMP
from src.model.Input import Input
from src.path_generator_utility import path_generator
from src.prediction.utility import crop_data_in_30h


class LoadRawData:

    def __init__(self, airport):
        self.etd = cudf.read_csv(
            path_generator(airport, FILE_NAME_ETD),
            parse_dates=[
                ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME,
                ETD_COLUMN_TIMESTAMP
            ],
        ).sort_values(ETD_COLUMN_TIMESTAMP)

        self.runways = cudf.read_csv(
            path_generator(airport, FILE_NAME_RUNWAYS),
            parse_dates=[
                RUNWAYS_COLUMN_ARRIVAL_RUNWAY_ACTUAL_TIME,
                RUNWAYS_COLUMN_TIMESTAMP
            ],
        ).sort_values(RUNWAYS_COLUMN_TIMESTAMP)

        self.standtimes = cudf.read_csv(
            path_generator(airport, FILE_NAME_STANDTIMES),
            parse_dates=[
                STANDTIMES_COLUMN_ARRIVAL_STAND_ACTUAL_TIME,
                STANDTIMES_COLUMN_TIMESTAMP
            ],
        ).sort_values(STANDTIMES_COLUMN_TIMESTAMP)

        self.weather = cudf.read_csv(
            path_generator(airport, FILE_NAME_LAMP),
            parse_dates=[
                LAMP_COLUMN_FORECAST_TIMESTAMP,
                LAMP_COLUMN_TIMESTAMP
            ],
        ).sort_values(LAMP_COLUMN_TIMESTAMP)

        self.config = cudf.read_csv(
            path_generator(airport, FILE_NAME_CONFIG),
            parse_dates=[
                CONFIG_COLUMN_TIMESTAMP
            ],
        ).sort_values(CONFIG_COLUMN_TIMESTAMP)

        self.mfs = cudf.read_csv(path_generator(airport, FILE_NAME_MFS))

    def get_input(self, now: pd.Timestamp):
        return Input(
            config=crop_data_in_30h(now, self.config),
            etd=crop_data_in_30h(now, self.etd),
            first_position=cudf.DataFrame(),
            lamp=crop_data_in_30h(now, self.weather),
            mfs=self.mfs,
            runways=crop_data_in_30h(now, self.runways),
            standtimes=crop_data_in_30h(now, self.standtimes),
            tbfm=cudf.DataFrame(),
            tfm=cudf.DataFrame()
        )
