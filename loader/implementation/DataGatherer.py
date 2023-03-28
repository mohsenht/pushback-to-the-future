import pandas as pd

from loader.DataLoader import DataLoader


class DataGatherer(DataLoader):
    def __init__(self):
        self.data_loaders = []

    def add_feature(self, data_loader: DataLoader):
        self.data_loaders.append(data_loader)

    def load_data(self, now: pd.Timestamp, data: pd.DataFrame) -> pd.DataFrame:
        now_data = data.loc[
            data.timestamp == now
            ].reset_index(drop=True)
        for loader in self.data_loaders:
            loader.load_data(now, now_data)

        return now_data
