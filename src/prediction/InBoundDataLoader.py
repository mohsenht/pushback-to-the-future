import gc

import pandas as pd

from src.constants import FILE_NAME_ETD, FILE_NAME_RUNWAYS, FILE_NAME_STANDTIMES, FILE_NAME_LAMP, FILE_NAME_CONFIG, \
    FILE_NAME_MFS, LAMP_COLUMN_FORECAST_TIMESTAMP, STANDTIMES_COLUMN_ARRIVAL_STAND_ACTUAL_TIME, \
    RUNWAYS_COLUMN_ARRIVAL_RUNWAY_ACTUAL_TIME, ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME, ETD_COLUMN_TIMESTAMP, \
    RUNWAYS_COLUMN_TIMESTAMP, STANDTIMES_COLUMN_TIMESTAMP, LAMP_COLUMN_TIMESTAMP, CONFIG_COLUMN_TIMESTAMP, \
    FILE_NAME_TFM, FILE_NAME_TBFM, FILE_NAME_FIRST_POSITION, TBFM_COLUMN_SCHEDULED_RUNWAY_ESTIMATED_TIME, \
    COLUMN_NAME_TIMESTAMP, TFM_COLUMN_ARRIVAL_RUNWAY_ESTIMATED_TIME, TFM_COLUMN_TIMESTAMP, FILE_NAME_LABELS
from src.path_generator_utility import path_generator, labels_path_generator
from src.prediction.CSVReader import CSVReader
from src.prediction.constant_chunk_size import ETD_CHUNK, TBFM_CHUNK, TFM_CHUNK, STANDTIMES_CHUNK, RUNWAYS_CHUNK, \
    LAMP_CHUNK, CONFIG_CHUNK, LABELS_CHUNK


class InBoundDataLoader:

    def __init__(self, airport):
        self.airport = airport

        self.data = self.config_reader = CSVReader(
            labels_path_generator(airport),
            LABELS_CHUNK,
            [
                CONFIG_COLUMN_TIMESTAMP
            ]
        )

        self.config_reader = CSVReader(
            path_generator(airport, FILE_NAME_CONFIG),
            CONFIG_CHUNK,
            [
                CONFIG_COLUMN_TIMESTAMP
            ]
        )

        self.etd_reader = CSVReader(
            path_generator(airport, FILE_NAME_ETD),
            ETD_CHUNK,
            [
                ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME,
                ETD_COLUMN_TIMESTAMP
            ]
        )

        self.first_position = pd.DataFrame()

        self.lamp_reader = CSVReader(
            path_generator(airport, FILE_NAME_LAMP),
            LAMP_CHUNK,
            [
                LAMP_COLUMN_FORECAST_TIMESTAMP,
                LAMP_COLUMN_TIMESTAMP
            ]
        )

        self.mfs = pd.read_csv(path_generator(airport, FILE_NAME_MFS))

        self.runways_reader = CSVReader(
            path_generator(airport, FILE_NAME_RUNWAYS),
            RUNWAYS_CHUNK,
            [
                RUNWAYS_COLUMN_ARRIVAL_RUNWAY_ACTUAL_TIME,
                RUNWAYS_COLUMN_TIMESTAMP
            ]
        )

        self.standtimes_reader = CSVReader(
            path_generator(airport, FILE_NAME_STANDTIMES),
            STANDTIMES_CHUNK,
            [
                STANDTIMES_COLUMN_ARRIVAL_STAND_ACTUAL_TIME,
                STANDTIMES_COLUMN_TIMESTAMP
            ]
        )

        self.tbfm_reader = CSVReader(
            path_generator(airport, FILE_NAME_TBFM),
            TBFM_CHUNK,
            [
                TBFM_COLUMN_SCHEDULED_RUNWAY_ESTIMATED_TIME,
                COLUMN_NAME_TIMESTAMP
            ]
        )

        self.tfm_reader = CSVReader(
            path_generator(airport, FILE_NAME_TFM),
            TFM_CHUNK,
            [
                TFM_COLUMN_ARRIVAL_RUNWAY_ESTIMATED_TIME,
                TFM_COLUMN_TIMESTAMP
            ]
        )

    def get_data(self, start_time, end_time, data_name):
        if data_name == FILE_NAME_LABELS:
            return self.data.get_data(start_time, end_time)
        if data_name == FILE_NAME_CONFIG:
            return self.config_reader.get_data(start_time, end_time)
        if data_name == FILE_NAME_ETD:
            return self.etd_reader.get_data(start_time, end_time)
        if data_name == FILE_NAME_FIRST_POSITION:
            return self.first_position
        if data_name == FILE_NAME_LAMP:
            return self.lamp_reader.get_data(start_time, end_time)
        if data_name == FILE_NAME_MFS:
            return self.mfs
        if data_name == FILE_NAME_RUNWAYS:
            return self.runways_reader.get_data(start_time, end_time)
        if data_name == FILE_NAME_STANDTIMES:
            return self.standtimes_reader.get_data(start_time, end_time)
        if data_name == FILE_NAME_TBFM:
            return self.tbfm_reader.get_data(start_time, end_time)
        if data_name == FILE_NAME_TFM:
            return self.tfm_reader.get_data(start_time, end_time)

    def closeAll(self):
        self.config_reader.close_file()
        self.etd_reader.close_file()
        self.lamp_reader.close_file()
        self.runways_reader.close_file()
        self.standtimes_reader.close_file()
        self.tbfm_reader.close_file()
        self.tfm_reader.close_file()
        gc.collect()
