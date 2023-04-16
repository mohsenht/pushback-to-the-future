import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd

from src.prediction.InBoundDataLoader import InBoundDataLoader
from src.prediction.LoadRawData import LoadRawData
from src.prediction.PredictInterface import PredictInterface
from src.constants import NUMBER_OF_PROCESSORS, COLUMN_NAME_TIMESTAMP
from src.prediction.constant_chunk_size import TIMESTAMP_CHUNK


class UnseenDataRunner:

    def __init__(self, path, airport, predict_interface: PredictInterface):
        self.path = path
        self.airport = airport
        self.predict_interface = predict_interface

    def run(self) -> pd.DataFrame:
        model = self.predict_interface.load_model(Path(os.getcwd()))
        results = []
        start_time = time.time()
        print(f"Loading airport features: {self.airport}")
        in_bound_data = InBoundDataLoader(self.airport, self.path)

        airport_submission_format = pd.read_csv(
            self.path,
            parse_dates=[COLUMN_NAME_TIMESTAMP]).sort_values(COLUMN_NAME_TIMESTAMP)
        airport_submission_format = airport_submission_format.loc[
            airport_submission_format.airport == self.airport
            ]
        timestamps = pd.to_datetime(airport_submission_format.timestamp.unique()).sort_values(COLUMN_NAME_TIMESTAMP)
        timestamps = timestamps[0]

        start_index = 0
        timestamps_length = len(timestamps)

        while start_index < timestamps_length:
            end_index = start_index + TIMESTAMP_CHUNK
            current_chunk = timestamps[start_index:end_index]
            current_chunk_len = len(current_chunk)

            sys.stdout.write('\r')
            # the exact output you're looking for:
            sys.stdout.write(f"{self.airport}: end_index: {end_index} - timestamps_length: {timestamps_length}")

            if current_chunk_len > 0:
                start_time = time.time()
                start_current = current_chunk[0]
                end_current = current_chunk[-1]
                raw_data = LoadRawData(in_bound_data, start_current - timedelta(hours=30), end_current)
                for ts in current_chunk:
                    results.append(
                        self.predict_interface.predict(
                            ts,
                            raw_data.get_data(),
                            raw_data,
                            self.airport,
                            model
                        )
                    )
                start_index = end_index

            sys.stdout.flush()

        in_bound_data.closeAll()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{self.airport} features loaded time: {elapsed_time:.2f} seconds")

        return pd.concat(results, axis=0, ignore_index=True)
