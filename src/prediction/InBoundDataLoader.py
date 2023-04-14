import pandas as pd

from src.constants import FILE_NAME_ETD, ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME, ETD_COLUMN_TIMESTAMP
from src.model.Input import Input
from src.path_generator_utility import path_generator
from src.prediction.CSVReader import CSVReader
from src.prediction.constant_chunk_size import ETD_CHUNK
from src.prediction.utility import crop_data_in_30h


class InBoundDataLoader:

    def __init__(self, airport):
        self.airport = airport
        self.etd_reader = CSVReader(
            path_generator(airport, FILE_NAME_ETD),
            ETD_CHUNK,
            [
                ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME,
                ETD_COLUMN_TIMESTAMP
            ]
        )

    def get_data(self, start_time, end_time):
        return self.etd_reader.get_data(start_time, end_time)
