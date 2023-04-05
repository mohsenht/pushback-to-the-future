from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from Input import Input
from LoadRawData import LoadRawData
from Model import Model


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
