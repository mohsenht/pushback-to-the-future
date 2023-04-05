import os
from pathlib import Path
from typing import Any

import pandas as pd

from LoadRawData import LoadRawData
from Model import Model
from PredictInterface import PredictInterface
from solution import predict, load_model


class Predictor(PredictInterface):

    def load_model(self, solution_directory: Path) -> Any:
        return load_model(solution_directory)

    def predict(self,
                now: pd.Timestamp,
                data: pd.DataFrame,
                raw_data: LoadRawData,
                airport: str,
                model: Model) -> pd.DataFrame:
        input_data = raw_data.get_input(now)
        return predict(
            input_data.config,
            input_data.etd,
            input_data.first_position,
            input_data.lamp,
            input_data.mfs,
            input_data.runways,
            input_data.standtimes,
            input_data.tbfm,
            input_data.tfm,
            airport,
            now,
            data,
            model,
            Path(os.getcwd()),
        )
