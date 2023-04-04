import os
import pandas as pd
import multiprocessing as mp
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from clean.extract.Extractor import Extractor
from constants import model_type_path
from loader.implementation.AircraftInfoLoader import AircraftInfoLoader
from loader.implementation.BusyETDLoader import BusyETDLoader
from loader.implementation.DataGatherer import DataGatherer
from loader.implementation.LastETDLoader import LastETDLoader
from loader.implementation.ArrivalToGateTimeWeightedMeanLoader import ArrivalToGateTimeWeightedMeanLoader
from loader.implementation.LastTwoETDLoader import LastTwoETDLoader
from loader.implementation.LastWeatherLoader import LastWeatherLoader
from loader.implementation.RunningRunwayInfoLoader import RunningRunwayInfoLoader
from loader.implementation.RunwayETDMeanLoader import RunwayETDMeanLoader
from loader.implementation.WeightedETDLoader import WeightedETDLoader

separator = os.path.sep
etd_path = f"data{separator}etd_2_weeks.csv"
runways_path = f"data{separator}runways_2_weeks.csv"
standtimes_path = f"data{separator}standtimes_2_weeks.csv"
lamp_path = f"data{separator}lamp_2_weeks.csv"
config_path = f"data{separator}config_2_weeks.csv"
mfs_path = f"data{separator}KCLT_mfs.csv"
labels_path = f"data{separator}labels_2_weeks.csv"
results_path = f"data{separator}results_2_weeks.csv"

# etd_path = f"data{separator}KCLT_etd.csv"
# runways_path = f"data{separator}KCLT_runways.csv"
# standtimes_path = f"data{separator}KCLT_standtimes.csv"
# lamp_path = f"data{separator}KCLT_lamp.csv"
# labels_path = f"data{separator}train_labels_KCLT.csv"
# results_path = f"data{separator}results_KCLT.csv"

number_of_processors = 10


def gather_features():
    data_gatherer = DataGatherer()
    data_gatherer.add_feature(LastETDLoader(etd_path))
    data_gatherer.add_feature(WeightedETDLoader(etd_path))
    data_gatherer.add_feature(LastTwoETDLoader(etd_path))
    data_gatherer.add_feature(BusyETDLoader(etd_path))
    data_gatherer.add_feature(ArrivalToGateTimeWeightedMeanLoader(runways_path, standtimes_path))
    data_gatherer.add_feature(LastWeatherLoader(lamp_path))
    data_gatherer.add_feature(RunningRunwayInfoLoader(config_path, model_type_path))
    data_gatherer.add_feature(AircraftInfoLoader(mfs_path, model_type_path))
    # data_gatherer.add_feature(RunwayETDMeanLoader(runways_path, etd_path))

    submission_format = pd.read_csv(labels_path, parse_dates=["timestamp"]).sort_values("timestamp")
    airport_submission_format = submission_format.loc[
        submission_format.airport == 'KCLT'
        ]
    timestamps = pd.to_datetime(airport_submission_format.timestamp.unique())

    pool = mp.Pool(processes=number_of_processors)
    results = pool.starmap(data_gatherer.load_data, [(ts, airport_submission_format) for ts in timestamps])
    pool.close()
    pool.join()

    results = pd.concat(results, ignore_index=True)
    results.to_csv(results_path, index=False)


def run_algorithm():
    data = pd.read_csv(results_path)

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
    gather_features()
    run_algorithm()

# if __name__ == '__main__':
#     print("type extraction")
#     Extractor("D:\\competetion\\", "KCLT").extract()
