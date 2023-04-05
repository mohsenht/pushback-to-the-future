from pathlib import Path
from typing import Any

import pandas as pd

from AirportModel import AirportModel
from FeatureExtractor import FeatureExtractor
from LoadRawData import LoadRawData
from Model import Model
from PredictInterface import PredictInterface
from clean.extract.TypeContainer import TypeContainer
from constants import airports, separator
from path_generator import types_path_generator


class FeatureLoader(PredictInterface):

    def load_model(self, solution_directory: Path) -> Any:
        airport_dict = {}

        for airport in airports:
            type_container = TypeContainer.from_file(f"{solution_directory}{separator}{types_path_generator(airport)}")
            airport_dict[airport] = AirportModel(model=None, type_container=type_container)

        return Model(airport_dict, FeatureExtractor().data_gatherer)

    def predict(self,
                now: pd.Timestamp,
                data: pd.DataFrame,
                raw_data: LoadRawData,
                airport: str,
                model: Model) -> pd.DataFrame:
        input_data = raw_data.get_input(now)
        return model.data_gatherer.load_features(now, data, input_data, model.airport_dict[airport].type_container)
