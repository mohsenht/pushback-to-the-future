from abc import ABC, abstractmethod

import pandas as pd

from model.Input import Input
from clean.TypeContainer import TypeContainer


class FeatureExtractor(ABC):

    @abstractmethod
    def load_data(self,
                  now: pd.Timestamp,
                  data: pd.DataFrame,
                  input_data: Input,
                  type_container: TypeContainer) -> pd.DataFrame:
        pass
