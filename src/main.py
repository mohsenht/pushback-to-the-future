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
    FILE_NAME_ETD, ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME, NUMBER_OF_PROCESSORS
from src.path_generator_utility import path_generator, labels_path_generator, model_path_generator, \
    open_arena_submission_format_path_generator, departure_model_path_generator, results_path_generator
from src.prediction.UnseenDataRunner import UnseenDataRunner
from src.prediction.implementors.FeatureLoader import FeatureLoader
from src.prediction.implementors.Predictor import Predictor


def run_algorithm(airport_name):
    data = pd.read_csv(path_generator(airport_name, FILE_NAME_RESULTS))

    x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, 4:], data.iloc[:, 3], test_size=0.2)

    model = xgb.XGBRegressor()
    model.load_model(model_path_generator(airport_name))
    y_pred = model.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)
    print("MAE: %.2f" % mae)


def data_loader():
    data = pd.read_csv(open_arena_submission_format_path_generator(), parse_dates=[COLUMN_NAME_TIMESTAMP]) \
        .sort_values(COLUMN_NAME_TIMESTAMP)
    data_features = UnseenDataRunner(data, FeatureLoader()).run(AIRPORTS)
    data_features.to_csv(f"{TRAIN_PATH}result.csv", index=False)


def train(airport_name):
    print("Loading data for airport: %s", airport_name)
    labeled_data = pd.read_csv(labels_path_generator(airport_name), parse_dates=[COLUMN_NAME_TIMESTAMP]) \
        .sort_values(COLUMN_NAME_TIMESTAMP)
    data = UnseenDataRunner(labeled_data, FeatureLoader()).run([airport_name])
    data.to_csv(path_generator(airport_name, FILE_NAME_RESULTS), index=False)
    #
    # etd = pd.read_csv(
    #     path_generator(airport_name, FILE_NAME_ETD),
    #     parse_dates=[
    #         COLUMN_NAME_TIMESTAMP,
    #         ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME
    #     ]
    # )
    # etd = etd.groupby(FLIGHT_ID).last()
    # etd = etd.reset_index()
    # etd = etd[[FLIGHT_ID, ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME]]
    # departure = data.merge(
    #     etd,
    #     how="left",
    #     on=FLIGHT_ID
    # )
    # labels = (departure[ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME] - departure[
    #     COLUMN_NAME_TIMESTAMP]).dt.seconds / 60
    # departure.drop(ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME, axis=1, inplace=True)
    #
    # features = departure.iloc[:, 4:]
    #
    # params = {
    #     'objective': 'reg:squaredlogerror',
    #     'lambda': 0.8,
    #     'tree_method': 'hist',
    #     'max_bin': 24,
    #     'max_depth': 10,
    #     'eval_metric': 'mae'
    # }
    #
    # model = xgb.XGBRegressor(n_estimators=100, **params)
    # print("Training model for airport: %s", airport_name)
    # model.fit(features, labels)
    # model.save_model(departure_model_path_generator(airport_name))
    #
    # features['actual_departure'] = labels
    # labels = departure.iloc[:, 3]
    # params = {
    #     'objective': 'reg:squaredlogerror',
    #     'lambda': 0.8,
    #     'tree_method': 'hist',
    #     'max_bin': 24,
    #     'max_depth': 10,
    #     'eval_metric': 'mae'
    # }
    #
    # model = xgb.XGBRegressor(n_estimators=100, **params)
    # print("Training model for airport: %s", airport_name)
    # model.fit(features, labels)
    # model.save_model(model_path_generator(airport_name))
    #
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"train{airport_name}: {elapsed_time:.2f} seconds")


def open_arena():
    unlabeled_data = pd.read_csv(open_arena_submission_format_path_generator(), parse_dates=[COLUMN_NAME_TIMESTAMP]) \
        .sort_values(COLUMN_NAME_TIMESTAMP)
    predictions = UnseenDataRunner(unlabeled_data, Predictor()).run(AIRPORTS)

    airport_submission_format = pd.read_csv(open_arena_submission_format_path_generator(),
                                            parse_dates=[COLUMN_NAME_TIMESTAMP])
    predictions = (
        predictions.set_index([
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
    predictions.to_csv(f"{TRAIN_PATH}result.csv", index=False)


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
    if not IS_DATA_CLEANED:
        print("data cleaning is running:")
        start_time = time.time()

        # for airport in AIRPORTS:
        #     Extractor(f"{TRAIN_PATH}", airport).extract()

        # pool = mp.Pool(processes=5)
        # pool.starmap(sort_csv_files,
        #              [(airport,) for
        #               airport in AIRPORTS])
        # pool.close()
        # pool.join()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"data cleaned in Elapsed time: {elapsed_time:.2f} seconds")

    start_time = time.time()
    pool = mp.Pool(processes=NUMBER_OF_PROCESSORS)
    pool.starmap(train,[(ts, ) for ts in AIRPORTS])
    pool.close()
    pool.join()
    open_arena()
    # add_departure_time()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")