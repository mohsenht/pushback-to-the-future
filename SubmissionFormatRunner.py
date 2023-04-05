import multiprocessing as mp
import os
from pathlib import Path

import pandas as pd

from LoadRawData import LoadRawData
from PredictInterface import PredictInterface
from constants import number_of_processors


class SubmissionFormatRunner:

    def __init__(self, file_path, predict_interface: PredictInterface):
        self.submission_format = pd.read_csv(file_path, parse_dates=["timestamp"]) \
            .sort_values("timestamp")
        self.predict_interface = predict_interface

    def run(self, airports) -> pd.DataFrame:
        model = self.predict_interface.load_model(Path(os.getcwd()))
        result = pd.DataFrame()
        for airport in airports:
            raw_data = LoadRawData(airport)

            airport_submission_format = self.submission_format.loc[
                self.submission_format.airport == airport
                ]
            timestamps = pd.to_datetime(airport_submission_format.timestamp.unique())

            pool = mp.Pool(processes=number_of_processors)
            results = pool.starmap(self.predict_interface.predict,
                                   [(ts, airport_submission_format, raw_data, airport, model) for ts in timestamps])
            pool.close()
            pool.join()

            result = pd.concat(results, ignore_index=True)

        return result

