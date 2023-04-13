import cudf
import pandas

from src.model.Input import Input
from src.clean.TypeContainer import TypeContainer
from src.loader.FeatureExtractor import FeatureExtractor


class FeatureExtractRunner:
    def __init__(self):
        self.data_loaders = []

    def add_feature(self, data_loader: FeatureExtractor):
        self.data_loaders.append(data_loader)

    def load_features(self,
                      now: pandas.Timestamp,
                      data: cudf.DataFrame,
                      input_data: Input,
                      type_container: TypeContainer) -> cudf.DataFrame:
        print(now)
        now_data = data.loc[
            data.timestamp == now
            ].reset_index(drop=True)
        for loader in self.data_loaders:
            now_data = loader.load_data(now, now_data, input_data, type_container)

        return now_data
