from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


from src.model.Model import Model
from src.prediction.LoadRawData import LoadRawData


class PredictInterface(ABC):

    @abstractmethod
    def load_model(self, solution_directory: Path) -> Any:
        pass

    @abstractmethod
    def predict(self,
                now: pd.Timestamp,
                data: pd.DataFrame,
                raw_data: LoadRawData,
                airport: str,
                model: Model) -> pd.DataFrame:
        pass
