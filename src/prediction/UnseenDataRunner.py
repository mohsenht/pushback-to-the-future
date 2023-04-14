import multiprocessing as mp
import os
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

    def __init__(self, unlableled_data, predict_interface: PredictInterface):
        self.unlabeled_data = unlableled_data
        self.predict_interface = predict_interface

    def run(self, airports) -> pd.DataFrame:
        model = self.predict_interface.load_model(Path(os.getcwd()))
        results = []
        for airport in airports:
            start_time = time.time()
            print(f"Loading airport features: {airport}")
            in_bound_data = InBoundDataLoader(airport)

            airport_submission_format = self.unlabeled_data.loc[
                self.unlabeled_data.airport == airport
                ]
            timestamps = pd.to_datetime(airport_submission_format.timestamp.unique()).sort_values(COLUMN_NAME_TIMESTAMP)
            timestamps = timestamps[0]

            start_index = 0
            while start_index < len(timestamps):
                end_index = start_index + TIMESTAMP_CHUNK
                current_chunk = timestamps[start_index:end_index]
                current_chunk_len = len(current_chunk)
                if current_chunk_len > 0:
                    raw_data = LoadRawData(in_bound_data, current_chunk[0] - timedelta(hours=30), current_chunk[-1])
                    pool = mp.Pool(processes=NUMBER_OF_PROCESSORS)
                    results.append(pd.concat(pool.starmap(self.predict_interface.predict,
                                                          [(ts, airport_submission_format, raw_data, airport, model) for
                                                           ts in timestamps]), axis=0, ignore_index=True))
                    pool.close()
                    pool.join()
                start_index = end_index

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"{airport} features loaded time: {elapsed_time:.2f} seconds")

        return pd.concat(results, axis=0, ignore_index=True)
