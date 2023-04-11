from pathlib import Path
from typing import Any

import cudf

from src.model.AirportModel import AirportModel
from src.loader.FeatureExtractorContainer import FeatureExtractorContainer
from src.prediction.LoadRawData import LoadRawData
from src.model.Model import Model
from src.prediction.PredictInterface import PredictInterface
from src.clean.TypeContainer import TypeContainer
from src.constants import AIRPORTS, SEPARATOR
from src.path_generator_utility import types_path_generator


class FeatureLoader(PredictInterface):

    def load_model(self, solution_directory: Path) -> Any:
        airport_dict = {}

        for airport in AIRPORTS:
            type_container = TypeContainer.from_file(f"{solution_directory}{SEPARATOR}{types_path_generator(airport)}")
            airport_dict[airport] = AirportModel(model=None, type_container=type_container)

        return Model(airport_dict, FeatureExtractorContainer().data_gatherer)

    def predict(self,
                now: pd.Timestamp,
                data: cudf.DataFrame,
                raw_data: LoadRawData,
                airport: str,
                model: Model) -> cudf.DataFrame:
        input_data = raw_data.get_input(now)
        return model.data_gatherer.load_features(
            now,
            data,
            input_data,
            model.airport_dict[airport].type_container
        )
