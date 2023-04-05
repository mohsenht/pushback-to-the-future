import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from FeatureLoader import FeatureLoader
from Predictor import Predictor
from SubmissionFormatRunner import SubmissionFormatRunner
from clean.extract.Extractor import Extractor
from constants import file_name_results, number_of_processors, model_path, separator, airports, train_path
from path_generator import path_generator, labels_path_generator, model_path_generator, \
    open_arena_submission_format_path_generator


def run_algorithm(airport_name):
    data = pd.read_csv(path_generator(airport_name, file_name_results))

    x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, 4:], data.iloc[:, 3], test_size=0.2)

    model = xgb.XGBRegressor()
    model.load_model(model_path_generator(airport_name))
    y_pred = model.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)
    print("MAE: %.2f" % mae)


def train():
    for airport in airports:
        data = SubmissionFormatRunner(labels_path_generator(airport), FeatureLoader()).run([airport])
        data.to_csv(path_generator(airport, file_name_results), index=False)
        labels = data.iloc[:, 3]
        features = data.iloc[:, 4:]
        params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.1,
            'max_depth': 5
        }

        model = xgb.XGBRegressor(n_estimators=100, **params)
        model.fit(features, labels)
        model.save_model(model_path_generator(airport))


def open_arena():
    data = SubmissionFormatRunner(open_arena_submission_format_path_generator(), Predictor()).run(airports)
    data.to_csv(path_generator("KCLT", "asghar"), index=False)


if __name__ == '__main__':
    for airport in airports:
        Extractor(f"{train_path}", airport).extract()
    # train(airports)
    # run_algorithm("KCLT")
    open_arena()
