import pandas as pd

from Input import Input
from clean.extract.TypeContainer import TypeContainer
from loader.DataLoader import DataLoader


class DataGatherer():
    def __init__(self):
        self.data_loaders = []

    def add_feature(self, data_loader: DataLoader):
        self.data_loaders.append(data_loader)

    def load_features(self, now: pd.Timestamp, data: pd.DataFrame, input: Input, type_container: TypeContainer) -> pd.DataFrame:
        now_data = data.loc[
            data.timestamp == now
            ].reset_index(drop=True)
        for loader in self.data_loaders:
            now_data = loader.load_data(now, now_data, input, type_container)

        return now_data
