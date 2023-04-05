from pathlib import Path
from typing import Any

import pandas as pd
import xgboost as xgb

from AirportModel import AirportModel
from FeatureExtractor import FeatureExtractor
from Input import Input
from Model import Model
from clean.extract.TypeContainer import TypeContainer
from constants import airports, separator
from path_generator import model_path_generator, types_path_generator


def load_model(solution_directory: Path) -> Any:
    airport_dict = {}

    for airport in airports:
        xgboost_model = xgb.XGBRegressor()
        xgboost_model.load_model(f"{solution_directory}{separator}{model_path_generator(airport)}")
        type_container = TypeContainer.from_file(f"{solution_directory}{separator}{types_path_generator(airport)}")
        airport_dict[airport] = AirportModel(xgboost_model, type_container)

    return Model(airport_dict, FeatureExtractor().data_gatherer)


def predict(
        config: pd.DataFrame,
        etd: pd.DataFrame,
        first_position: pd.DataFrame,
        lamp: pd.DataFrame,
        mfs: pd.DataFrame,
        runways: pd.DataFrame,
        standtimes: pd.DataFrame,
        tbfm: pd.DataFrame,
        tfm: pd.DataFrame,
        airport: str,
        prediction_time: pd.Timestamp,
        partial_submission_format: pd.DataFrame,
        model: Any,
        solution_directory: Path,
) -> pd.DataFrame:
    input = Input(
        config,
        etd,
        first_position,
        lamp,
        mfs,
        runways,
        standtimes,
        tbfm,
        tfm,
    )
    prediction = partial_submission_format.copy()
    data = model.data_gatherer.load_features(prediction_time, partial_submission_format, input,
                                             model.airport_dict[airport].type_container)
    features = data.iloc[:, 4:]
    y_pred = model.airport_dict[airport].model.predict(features)

    prediction["minutes_until_pushback"] = y_pred
    return prediction
