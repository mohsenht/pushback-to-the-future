from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from model.AirportModel import AirportModel
from loader.FeatureExtractorContainer import FeatureExtractorContainer
from model.Input import Input
from model.Model import Model
from clean.TypeContainer import TypeContainer
from constants import AIRPORTS, SEPARATOR
from path_generator_utility import model_path_generator, types_path_generator


def load_model(solution_directory: Path) -> Any:
    airport_dict = {}

    for airport in AIRPORTS:
        xgboost_model = xgb.XGBRegressor()
        xgboost_model.load_model(f"{solution_directory}{SEPARATOR}{model_path_generator(airport)}")
        type_container = TypeContainer.from_file(f"{solution_directory}{SEPARATOR}{types_path_generator(airport)}")
        airport_dict[airport] = AirportModel(xgboost_model, type_container)

    return Model(airport_dict, FeatureExtractorContainer().data_gatherer)


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
    data = model.data_gatherer.load_features(
        prediction_time,
        partial_submission_format,
        input,
        model.airport_dict[airport].type_container
    )
    features = data.iloc[:, 4:]
    y_pred = model.airport_dict[airport].model.predict(features)

    prediction["minutes_until_pushback"] = np.maximum(y_pred.round(), 0).astype(int)
    return prediction
