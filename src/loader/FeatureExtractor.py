from abc import ABC, abstractmethod

import pandas as pd

from src.model.Input import Input
from src.clean.TypeContainer import TypeContainer


class FeatureExtractor(ABC):

    @abstractmethod
    def load_data(self,
                  now: pd.Timestamp,
                  data: pd.DataFrame,
                  input_data: Input,
                  type_container: TypeContainer) -> pd.DataFrame:
        pass
