from multiprocessing import get_context
import os
import time
from pathlib import Path

import cudf

from src.prediction.LoadRawData import LoadRawData
from src.prediction.PredictInterface import PredictInterface
from src.constants import NUMBER_OF_PROCESSORS


class UnseenDataRunner:

    def __init__(self, unlableled_data, predict_interface: PredictInterface):
        self.unlabeled_data = unlableled_data
        self.predict_interface = predict_interface

    def run(self, airports) -> cudf.DataFrame:
        model = self.predict_interface.load_model(Path(os.getcwd()))
        results = []
        for airport in airports:
            start_time = time.time()
            print(f"Loading airport features: {airport}")
            raw_data = LoadRawData(airport)

            airport_submission_format = self.unlabeled_data.loc[
                self.unlabeled_data.airport == airport
                ]
            timestamps = cudf.to_datetime(airport_submission_format.timestamp.unique())
            timestamps = timestamps.to_pandas()

            ctx = get_context("spawn")
            pool = ctx.Pool(processes=NUMBER_OF_PROCESSORS)
            results.append(cudf.concat(
                pool.starmap(
                    self.predict_interface.predict,
                    [(ts, airport_submission_format, raw_data, airport, model) for ts in timestamps]), axis=0,
                ignore_index=True))
            pool.close()
            pool.join()

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"{airport} features loaded time: {elapsed_time:.2f} seconds")

        return cudf.concat(results, axis=0, ignore_index=True)
