import sys
import multiprocessing as mp
import time

import cudf
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from constants import FILE_NAME_RESULTS, AIRPORTS, TRAIN_PATH, COLUMN_NAME_TIMESTAMP, SUBMISSION_FORMAT_AIRPORT, \
    SUBMISSION_FORMAT_FLIGHT_ID, IS_DATA_CLEANED, FLIGHT_ID, \
    FILE_NAME_ETD, ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME, NUMBER_OF_PROCESSORS, FILE_NAME_RESULTS_2, \
    SUBMISSION_FORMAT_MINUTES_UNTIL_PUSHBACK, FILE_NAME_PREDICTION
from src.clean.Extractor import Extractor
from src.clean.TimestampSorter import sort_csv_files
from src.hyper_parameters import XGBOOST_PARAMETERS, XGBOOST_ESTIMATORS
from src.path_generator_utility import path_generator, labels_path_generator, model_path_generator, \
    open_arena_submission_format_path_generator, departure_model_path_generator, results_path_generator
from src.prediction.UnseenDataRunner import UnseenDataRunner
from src.prediction.implementors.FeatureLoader import FeatureLoader
from src.prediction.implementors.FeatureLoader2 import FeatureLoader2
from src.prediction.implementors.Predictor import Predictor


def add_departure(airport_name, data, departure_column_name):
    etd = pd.read_csv(
        path_generator(airport_name, FILE_NAME_ETD),
        parse_dates=[
            COLUMN_NAME_TIMESTAMP,
            ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME
        ]
    )
    etd = etd.groupby(FLIGHT_ID).last()
    etd = etd.reset_index()
    etd = etd[[FLIGHT_ID, ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME]]
    departure = data.merge(
        etd,
        how="left",
        on=FLIGHT_ID
    )
    departure[departure_column_name] = (departure[ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME] - departure[
        COLUMN_NAME_TIMESTAMP]).dt.seconds / 60
    departure.drop(ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME, axis=1, inplace=True)

    return departure


def data_loader_departure(airport_name):
    print("Loading data for airport: %s", airport_name)
    labeled_data = pd.read_csv(labels_path_generator(airport_name), parse_dates=[COLUMN_NAME_TIMESTAMP]) \
        .sort_values(COLUMN_NAME_TIMESTAMP)
    data = UnseenDataRunner(labeled_data, FeatureLoader()).run([airport_name])
    data.to_csv(path_generator(airport_name, FILE_NAME_RESULTS), index=False)


def data_loader_pushback(airport_name):
    print("Loading data for airport: %s", airport_name)
    data = pd.read_csv(labels_path_generator(airport_name), parse_dates=[COLUMN_NAME_TIMESTAMP]) \
        .sort_values(COLUMN_NAME_TIMESTAMP)

    departure = add_departure(airport_name, data, 'last_etd')

    data = UnseenDataRunner(departure, FeatureLoader2()).run([airport_name])
    data.to_csv(path_generator(airport_name, FILE_NAME_RESULTS_2), index=False)


def train(airport_name):
    departure_data = pd.read_csv(path_generator(airport_name, FILE_NAME_RESULTS), parse_dates=[COLUMN_NAME_TIMESTAMP]) \
        .sort_values(COLUMN_NAME_TIMESTAMP)

    departure_data = add_departure(airport_name, departure_data, SUBMISSION_FORMAT_MINUTES_UNTIL_PUSHBACK)

    departure_features = departure_data.iloc[:, 4:]
    departure_labels = departure_data.iloc[:, 3]

    model = xgb.XGBRegressor(n_estimators=XGBOOST_ESTIMATORS, **XGBOOST_PARAMETERS)
    print("Training departure model for airport: %s", airport_name)
    model.fit(departure_features, departure_labels)
    model.save_model(departure_model_path_generator(airport_name))

    pushback_data = pd.read_csv(path_generator(airport_name, FILE_NAME_RESULTS_2), parse_dates=[COLUMN_NAME_TIMESTAMP]) \
        .sort_values(COLUMN_NAME_TIMESTAMP)

    pushback_data = add_departure(airport_name, pushback_data, 'last_etd')

    pushback_features = pushback_data.iloc[:, 4:]
    pushback_labels = pushback_data.iloc[:, 3]

    model = xgb.XGBRegressor(n_estimators=XGBOOST_ESTIMATORS, **XGBOOST_PARAMETERS)
    print("Training model for airport: %s", airport_name)
    model.fit(pushback_features, pushback_labels)
    model.save_model(model_path_generator(airport_name))


def open_arena(airport_name):
    unlabeled_data = pd.read_csv(open_arena_submission_format_path_generator(), parse_dates=[COLUMN_NAME_TIMESTAMP]) \
        .sort_values(COLUMN_NAME_TIMESTAMP)
    predictions = UnseenDataRunner(unlabeled_data, Predictor()).run([airport_name])
    predictions.to_csv(path_generator(airport_name, FILE_NAME_PREDICTION), index=False)


def build_submission_format():
    predictions = []
    for airport_name in AIRPORTS:
        predictions.append(
            pd.read_csv(path_generator(airport_name, FILE_NAME_PREDICTION), parse_dates=[COLUMN_NAME_TIMESTAMP]))

    all_predictions = pd.concat(predictions, axis=0, ignore_index=True)

    airport_submission_format = pd.read_csv(open_arena_submission_format_path_generator(),
                                            parse_dates=[COLUMN_NAME_TIMESTAMP])

    all_predictions = (
        all_predictions.set_index([
            SUBMISSION_FORMAT_FLIGHT_ID,
            COLUMN_NAME_TIMESTAMP,
            SUBMISSION_FORMAT_AIRPORT
        ])
        .loc[
            airport_submission_format.set_index([
                SUBMISSION_FORMAT_FLIGHT_ID,
                COLUMN_NAME_TIMESTAMP,
                SUBMISSION_FORMAT_AIRPORT
            ]).index
        ]
        .reset_index()
    )
    all_predictions.to_csv(f"{TRAIN_PATH}submission_format_results.csv", index=False)


def add_departure_time():
    for airport_name in AIRPORTS:
        print("Loading data for airport: %s", airport_name)
        start_time = time.time()

        results = cudf.read_csv(
            path_generator(airport_name, FILE_NAME_RESULTS),
            parse_dates=[COLUMN_NAME_TIMESTAMP]
        )
        etd = cudf.read_csv(
            path_generator(airport_name, FILE_NAME_ETD),
            parse_dates=[
                COLUMN_NAME_TIMESTAMP,
                ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME
            ]
        )
        etd = etd.groupby(FLIGHT_ID).last()
        etd = etd.reset_index()
        etd = etd[[FLIGHT_ID, ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME]]
        departure = results.merge(
            etd,
            how="left",
            on=FLIGHT_ID
        )
        labels = (departure[ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME] - departure[
            COLUMN_NAME_TIMESTAMP]).dt.seconds / 60
        departure.drop(ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME, axis=1, inplace=True)

        labels = labels.to_pandas()
        departure = departure.to_pandas()
        labels = results.iloc[:, 3]
        features = results.iloc[:, 4:]

        params = {
            'objective': 'reg:squaredlogerror',
            'lambda': 0.8,
            'tree_method': 'hist',
            'max_bin': 24,
            'max_depth': 10,
            'eval_metric': 'mae'
        }

        model = xgb.XGBRegressor(n_estimators=100, **params)
        print("Training model for airport: %s", airport_name)
        model.fit(features, labels)
        model.save_model(departure_model_path_generator(airport_name))

        features['actual_departure'] = labels
        params = {
            'objective': 'reg:squaredlogerror',
            'lambda': 0.8,
            'tree_method': 'hist',
            'max_bin': 24,
            'max_depth': 10,
            'eval_metric': 'mae'
        }

        model = xgb.XGBRegressor(n_estimators=100, **params)
        print("Training model for airport: %s", airport_name)
        model.fit(features, labels)
        model.save_model(model_path_generator(airport_name))

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"train{airport_name}: {elapsed_time:.2f} seconds")


if __name__ == '__main__':
    start_time = time.time()
    if not IS_DATA_CLEANED:
        print("data cleaning is running:")
        for airport in AIRPORTS:
            Extractor(f"{TRAIN_PATH}", airport).extract()

        pool = mp.Pool(processes=5)
        pool.starmap(sort_csv_files,
                     [(airport,) for
                      airport in AIRPORTS])
        pool.close()
        pool.join()

    start_time = time.time()
    pool = mp.Pool(processes=NUMBER_OF_PROCESSORS)
    pool.starmap(data_loader_pushback, [(ts,) for ts in AIRPORTS])
    pool.close()
    pool.join()

    # open_arena()
    # add_departure_time()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
