import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from clean.Extractor import Extractor
from constants import file_name_results, airports, train_path
from path_generator_utility import path_generator, labels_path_generator, model_path_generator, \
    open_arena_submission_format_path_generator
from prediction.UnseenDataRunner import UnseenDataRunner
from prediction.implementors.FeatureLoader import FeatureLoader
from prediction.implementors.Predictor import Predictor


def run_algorithm(airport_name):
    data = pd.read_csv(path_generator(airport_name, file_name_results))

    x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, 4:], data.iloc[:, 3], test_size=0.2)

    model = xgb.XGBRegressor()
    model.load_model(model_path_generator(airport_name))
    y_pred = model.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)
    print("MAE: %.2f" % mae)


def train():
    for airport_name in airports:
        labeled_data = pd.read_csv(labels_path_generator(airport_name), parse_dates=["timestamp"]) \
            .sort_values("timestamp")
        data = UnseenDataRunner(labeled_data, FeatureLoader()).run([airport_name])
        data.to_csv(path_generator(airport_name, file_name_results), index=False)
        labels = data.iloc[:, 3]
        features = data.iloc[:, 4:]
        params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.1,
            'max_depth': 5
        }

        model = xgb.XGBRegressor(n_estimators=100, **params)
        model.fit(features, labels)
        model.save_model(model_path_generator(airport_name))


def open_arena():
    unlabeled_data = pd.read_csv(open_arena_submission_format_path_generator(), parse_dates=["timestamp"]) \
        .sort_values("timestamp")
    predictions = UnseenDataRunner(unlabeled_data, Predictor()).run(airports)

    airport_submission_format = pd.read_csv(open_arena_submission_format_path_generator(), parse_dates=["timestamp"])
    predictions = (
        predictions.set_index(["gufi", "timestamp", "airport"])
        .loc[
            airport_submission_format.set_index(["gufi", "timestamp", "airport"]).index
        ]
        .reset_index()
    )
    predictions.to_csv(f"{train_path}result.csv", index=False)


if __name__ == '__main__':
    for airport in airports:
        Extractor(f"{train_path}", airport).extract()
    # train()
    # run_algorithm("KCLT")
    open_arena()
