from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cudf


from src.model.Model import Model
from src.prediction.LoadRawData import LoadRawData


class PredictInterface(ABC):

    @abstractmethod
    def load_model(self, solution_directory: Path) -> Any:
        pass

    @abstractmethod
    def predict(self,
                now: pd.Timestamp,
                data: cudf.DataFrame,
                raw_data: LoadRawData,
                airport: str,
                model: Model) -> cudf.DataFrame:
        pass
