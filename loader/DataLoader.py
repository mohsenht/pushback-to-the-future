from abc import ABC, abstractmethod
import pandas as pd


class DataLoader(ABC):

    @abstractmethod
    def load_data(self, now: pd.Timestamp, data: pd.DataFrame) -> pd.DataFrame:
        pass