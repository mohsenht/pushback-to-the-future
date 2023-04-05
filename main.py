import multiprocessing as mp
import time

import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from clean.extract.Extractor import Extractor
from constants import file_name_results, number_of_processors, train_path
from loader.implementation.AircraftInfoLoader import AircraftInfoLoader
from loader.implementation.ArrivalToGateTimeWeightedMeanLoader import ArrivalToGateTimeWeightedMeanLoader
from loader.implementation.BusyETDLoader import BusyETDLoader
from loader.implementation.DataGatherer import DataGatherer
from loader.implementation.LastETDLoader import LastETDLoader
from loader.implementation.LastTwoETDLoader import LastTwoETDLoader
from loader.implementation.LastWeatherLoader import LastWeatherLoader
from loader.implementation.RunningRunwayInfoLoader import RunningRunwayInfoLoader
from loader.implementation.WeightedETDLoader import WeightedETDLoader
from path_generator import path_generator, labels_path_generator, types_path_generator


def gather_features(airport_name):
    data_gatherer = DataGatherer()
    data_gatherer.add_feature(LastETDLoader(airport_name))
    data_gatherer.add_feature(WeightedETDLoader(airport_name))
    data_gatherer.add_feature(LastTwoETDLoader(airport_name))
    data_gatherer.add_feature(BusyETDLoader(airport_name))
    data_gatherer.add_feature(ArrivalToGateTimeWeightedMeanLoader(airport_name))
    data_gatherer.add_feature(LastWeatherLoader(airport_name))
    data_gatherer.add_feature(RunningRunwayInfoLoader(airport_name))
    data_gatherer.add_feature(AircraftInfoLoader(airport_name))
    # data_gatherer.add_feature(RunwayETDMeanLoader(runways_path, etd_path))

    submission_format = pd.read_csv(labels_path_generator(airport_name), parse_dates=["timestamp"])\
        .sort_values("timestamp")
    airport_submission_format = submission_format.loc[
        submission_format.airport == airport_name
        ]
    timestamps = pd.to_datetime(airport_submission_format.timestamp.unique())

    pool = mp.Pool(processes=number_of_processors)
    results = pool.starmap(data_gatherer.load_data, [(ts, airport_submission_format) for ts in timestamps])
    pool.close()
    pool.join()

    results = pd.concat(results, ignore_index=True)
    results.to_csv(path_generator(airport_name, file_name_results), index=False)


def run_algorithm(airport_name):
    data = pd.read_csv(path_generator(airport_name, file_name_results))

    x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, 4:], data.iloc[:, 3], test_size=0.2)

    params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.1,
        'max_depth': 5
    }

    model = xgb.XGBRegressor(n_estimators=100, **params)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)
    print("MAE: %.2f" % mae)


if __name__ == '__main__':
    start_time = time.time()
    airport = "KCLT"
    gather_features(airport)
    run_algorithm(airport)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"time: {elapsed_time}")

# if __name__ == '__main__':
#     print("type extraction")
#     Extractor(f"{train_path}", "KCLT").extract()
