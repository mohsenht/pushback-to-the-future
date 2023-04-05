from abc import ABC, abstractmethod
import pandas as pd

from Input import Input
from clean.extract.TypeContainer import TypeContainer


class DataLoader(ABC):

    @abstractmethod
    def load_data(self, now: pd.Timestamp, data: pd.DataFrame, input: Input, type_container: TypeContainer) -> pd.DataFrame:
        pass
