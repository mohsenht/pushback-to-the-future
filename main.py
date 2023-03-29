import os
import pandas as pd
import multiprocessing as mp
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from loader.implementation.DataGatherer import DataGatherer
from loader.implementation.LastETDLoader import LastETDLoader
from loader.implementation.ArrivalToGateTimeWeightedMeanLoader import ArrivalToGateTimeWeightedMeanLoader
from loader.implementation.RunwayETDMeanLoader import RunwayETDMeanLoader
from loader.implementation.WeightedETDLoader import WeightedETDLoader

separator = os.path.sep
etd_path = f"data{separator}etd_2_weeks.csv"
runways_path = f"data{separator}runways_2_weeks.csv"
standtimes_path = f"data{separator}standtimes_2_weeks.csv"
labels_path = f"data{separator}labels_2_weeks.csv"
results_path = f"data{separator}results_2_weeks.csv"

number_of_processors = 10

def gather_features():
    dataGatherer = DataGatherer()
    dataGatherer.add_feature(WeightedETDLoader(etd_path))
    dataGatherer.add_feature(LastETDLoader(etd_path))
    dataGatherer.add_feature(ArrivalToGateTimeWeightedMeanLoader(runways_path, standtimes_path))
    #dataGatherer.add_feature(RunwayETDMeanLoader(runways_path, etd_path))

    submission_format = pd.read_csv(labels_path, parse_dates=["timestamp"])
    airport_submission_format = submission_format.loc[
        submission_format.airport == 'KCLT'
        ]
    timestamps = pd.to_datetime(airport_submission_format.timestamp.unique())

    pool = mp.Pool(processes=number_of_processors)
    results = pool.starmap(dataGatherer.load_data, [(ts, airport_submission_format) for ts in timestamps])
    pool.close()
    pool.join()

    results = pd.concat(results, ignore_index=True)
    results.to_csv(results_path, index=False)


def run_algorithm():
    data = pd.read_csv(results_path)

    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, 5:], data.iloc[:, 3], test_size=0.2)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.1,
        'max_depth': 5,
        'n_estimators': 100
    }
    model = xgb.train(params, dtrain)
    y_pred = model.predict(dtest)

    mae = mean_absolute_error(y_test, y_pred)
    print("MAE: %.2f" % mae)


if __name__ == '__main__':
    gather_features()
    run_algorithm()
