import multiprocessing as mp
import os
from pathlib import Path

import pandas as pd

from prediction.LoadRawData import LoadRawData
from prediction.PredictInterface import PredictInterface
from constants import number_of_processors


class UnseenDataRunner:

    def __init__(self, unlableled_data, predict_interface: PredictInterface):
        self.unlabeled_data = unlableled_data
        self.predict_interface = predict_interface

    def run(self, airports) -> pd.DataFrame:
        model = self.predict_interface.load_model(Path(os.getcwd()))
        result = pd.DataFrame()
        for airport in airports:
            raw_data = LoadRawData(airport)

            airport_submission_format = self.unlabeled_data.loc[
                self.unlabeled_data.airport == airport
                ]
            timestamps = pd.to_datetime(airport_submission_format.timestamp.unique())

            pool = mp.Pool(processes=number_of_processors)
            results = pool.starmap(self.predict_interface.predict,
                                   [(ts, airport_submission_format, raw_data, airport, model) for ts in timestamps])
            pool.close()
            pool.join()

            result = pd.concat(results, ignore_index=True)

        return result

