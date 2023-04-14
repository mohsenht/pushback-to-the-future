import time

import cudf
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from constants import FILE_NAME_RESULTS, AIRPORTS, TRAIN_PATH, COLUMN_NAME_TIMESTAMP, SUBMISSION_FORMAT_AIRPORT, \
    SUBMISSION_FORMAT_FLIGHT_ID
from src.path_generator_utility import path_generator, labels_path_generator, model_path_generator, \
    open_arena_submission_format_path_generator
from src.prediction.UnseenDataRunner import UnseenDataRunner
from src.prediction.implementors.FeatureLoader import FeatureLoader
from src.prediction.implementors.Predictor import Predictor
from src.utility import rearrange_submission


def run_algorithm(airport_name):
    data = cudf.read_csv(path_generator(airport_name, FILE_NAME_RESULTS))

    x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, 4:], data.iloc[:, 3], test_size=0.2)

    model = xgb.XGBRegressor()
    model.load_model(model_path_generator(airport_name))
    y_pred = model.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)
    print("MAE: %.2f" % mae)


def data_loader():
    data = cudf.read_csv(open_arena_submission_format_path_generator(), parse_dates=[COLUMN_NAME_TIMESTAMP]) \
        .sort_values(COLUMN_NAME_TIMESTAMP)
    data_features = UnseenDataRunner(data, FeatureLoader()).run(AIRPORTS)
    data_features.to_csv(f"{TRAIN_PATH}result.csv", index=False)


def train():
    for airport_name in AIRPORTS:
        print(f"Loading data for airport: {airport_name}")
        labeled_data = cudf.read_csv(labels_path_generator(airport_name), parse_dates=[COLUMN_NAME_TIMESTAMP]) \
            .sort_values(COLUMN_NAME_TIMESTAMP)
        data = UnseenDataRunner(labeled_data, FeatureLoader()).run([airport_name])
        data.to_csv(path_generator(airport_name, FILE_NAME_RESULTS), index=False)
        # labels = data.iloc[:, 3]
        # features = data.iloc[:, 4:]
        # params = {
        #     'objective': 'reg:pseudohubererror',
        #     'learning_rate': 0.1,
        #     'max_depth': 5
        # }
        #
        # model = xgb.XGBRegressor(n_estimators=100, **params)
        # print(f"Training model for airport: {airport_name}")
        # model.fit(features, labels)
        # model.save_model(model_path_generator(airport_name))


def open_arena():
    unlabeled_data = cudf.read_csv(open_arena_submission_format_path_generator(), parse_dates=[COLUMN_NAME_TIMESTAMP]) \
        .sort_values(COLUMN_NAME_TIMESTAMP)
    predictions = UnseenDataRunner(unlabeled_data, Predictor()).run(AIRPORTS)

    airport_submission_format = cudf.read_csv(open_arena_submission_format_path_generator(),
                                              parse_dates=[COLUMN_NAME_TIMESTAMP])

    predictions = rearrange_submission(airport_submission_format, predictions)

    predictions.to_csv(f"{TRAIN_PATH}result.csv", index=False)


if __name__ == '__main__':
    start_time = time.time()

    # for airport in AIRPORTS:
    #     Extractor(f"{TRAIN_PATH}", airport).extract()
    train()
    # open_arena()
    # data_loader()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
