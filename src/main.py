import time

import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from constants import FILE_NAME_RESULTS, AIRPORTS, TRAIN_PATH, COLUMN_NAME_TIMESTAMP, SUBMISSION_FORMAT_AIRPORT, \
    SUBMISSION_FORMAT_FLIGHT_ID, IS_DATA_CLEANED, AIRPORTS_CLEAN
from src.clean.Extractor import Extractor
from src.clean.TimestampSorter import sort_csv_files
from src.path_generator_utility import path_generator, labels_path_generator, model_path_generator, \
    open_arena_submission_format_path_generator
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


def train():
    for airport_name in AIRPORTS:
        print("Loading data for airport: %s", airport_name)
        labeled_data = pd.read_csv(labels_path_generator(airport_name), parse_dates=[COLUMN_NAME_TIMESTAMP]) \
            .sort_values(COLUMN_NAME_TIMESTAMP)
        data = UnseenDataRunner(labeled_data, FeatureLoader()).run([airport_name])
        data.to_csv(path_generator(airport_name, FILE_NAME_RESULTS), index=False)
        labels = data.iloc[:, 3]
        features = data.iloc[:, 4:]
        params = {
            'objective': 'reg:pseudohubererror',
            'learning_rate': 0.1,
            'max_depth': 5
        }

        model = xgb.XGBRegressor(n_estimators=100, **params)
        print("Training model for airport: %s", airport_name)
        model.fit(features, labels)
        model.save_model(model_path_generator(airport_name))


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


if __name__ == '__main__':
    if not IS_DATA_CLEANED:
        print("data cleaning is running:")
        start_time = time.time()
        for airport in AIRPORTS:
            Extractor(f"{TRAIN_PATH}", airport).extract()
        for airport in AIRPORTS_CLEAN:
            sort_csv_files(airport)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"data cleaned in Elapsed time: {elapsed_time:.2f} seconds")

    start_time = time.time()
    train()
    open_arena()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

