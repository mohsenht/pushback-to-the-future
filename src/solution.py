from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from src.model.AirportModel import AirportModel
from src.loader.FeatureExtractorContainer import FeatureExtractorContainer
from src.model.Input import Input
from src.model.Model import Model
from src.clean.TypeContainer import TypeContainer
from src.constants import AIRPORTS, SEPARATOR, SUBMISSION_FORMAT_MINUTES_UNTIL_PUSHBACK, RUNWAYS_COLUMN_TIMESTAMP, \
    ETD_COLUMN_TIMESTAMP, LAMP_COLUMN_TIMESTAMP, STANDTIMES_COLUMN_TIMESTAMP, CONFIG_COLUMN_TIMESTAMP, \
    ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME, RUNWAYS_COLUMN_ARRIVAL_RUNWAY_ACTUAL_TIME, \
    LAMP_COLUMN_FORECAST_TIMESTAMP, STANDTIMES_COLUMN_ARRIVAL_STAND_ACTUAL_TIME
from src.path_generator_utility import model_path_generator, types_path_generator

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
    input_data = prepareData(config,
                             etd,
                             first_position,
                             lamp,
                             mfs,
                             runways,
                             standtimes,
                             tbfm,
                             tfm)
    prediction = partial_submission_format.copy()
    data = model.data_gatherer.load_features(
        prediction_time,
        partial_submission_format,
        input_data,
        model.airport_dict[airport].type_container
    )
    features = data.iloc[:, 4:]
    y_pred = model.airport_dict[airport].model.predict(features)

    prediction[SUBMISSION_FORMAT_MINUTES_UNTIL_PUSHBACK] = np.maximum(y_pred.round(), 0).astype(int)
    return prediction


def prepareData(
        config: pd.DataFrame,
        etd: pd.DataFrame,
        first_position: pd.DataFrame,
        lamp: pd.DataFrame,
        mfs: pd.DataFrame,
        runways: pd.DataFrame,
        standtimes: pd.DataFrame,
        tbfm: pd.DataFrame,
        tfm: pd.DataFrame,
) -> Input:
    etd = etd.copy()
    runways = runways.copy()
    standtimes = standtimes.copy()
    lamp = lamp.copy()
    config = config.copy()

    etd[ETD_COLUMN_TIMESTAMP] = pd.to_datetime(etd[ETD_COLUMN_TIMESTAMP])
    etd[ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME] = pd.to_datetime(
        etd[ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME])

    runways[RUNWAYS_COLUMN_TIMESTAMP] = pd.to_datetime(runways[RUNWAYS_COLUMN_TIMESTAMP])
    runways[RUNWAYS_COLUMN_ARRIVAL_RUNWAY_ACTUAL_TIME] = pd.to_datetime(
        runways[RUNWAYS_COLUMN_ARRIVAL_RUNWAY_ACTUAL_TIME])

    standtimes[STANDTIMES_COLUMN_TIMESTAMP] = pd.to_datetime(standtimes[STANDTIMES_COLUMN_TIMESTAMP])
    standtimes[STANDTIMES_COLUMN_ARRIVAL_STAND_ACTUAL_TIME] = pd.to_datetime(
        standtimes[STANDTIMES_COLUMN_ARRIVAL_STAND_ACTUAL_TIME])

    lamp[LAMP_COLUMN_TIMESTAMP] = pd.to_datetime(lamp[LAMP_COLUMN_TIMESTAMP])
    lamp[LAMP_COLUMN_FORECAST_TIMESTAMP] = pd.to_datetime(lamp[LAMP_COLUMN_FORECAST_TIMESTAMP])

    config[CONFIG_COLUMN_TIMESTAMP] = pd.to_datetime(config[CONFIG_COLUMN_TIMESTAMP])

    return Input(
        config.sort_values(CONFIG_COLUMN_TIMESTAMP),
        etd.sort_values(ETD_COLUMN_TIMESTAMP),
        first_position,
        lamp.sort_values(LAMP_COLUMN_TIMESTAMP),
        mfs,
        runways.sort_values(RUNWAYS_COLUMN_TIMESTAMP),
        standtimes.sort_values(STANDTIMES_COLUMN_TIMESTAMP),
        tbfm,
        tfm,
    )
