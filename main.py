import os
import pandas as pd
import multiprocessing as mp

from loader.implementation.DataGatherer import DataGatherer
from loader.implementation.ETDLoader import ETDLoader
from loader.implementation.PushbackMeanLoader import PushbackMeanLoader
from loader.implementation.RunwayETDMeanLoader import RunwayETDMeanLoader

separator = os.path.sep
etd_path = f"data{separator}etd_2_weeks.csv"
runways_path = f"data{separator}runways_2_weeks.csv"
standtimes_path = f"data{separator}standtimes_2_weeks.csv"
labels_path = f"data{separator}labels_2_weeks.csv"
results_path = f"data{separator}results_2_weeks.csv"

number_of_processors = 10

if __name__ == '__main__':
    dataGatherer = DataGatherer()
    dataGatherer.add_feature(ETDLoader(etd_path))
    dataGatherer.add_feature(PushbackMeanLoader(runways_path, standtimes_path))
#    dataGatherer.add_feature(RunwayETDMeanLoader(runways_path, etd_path))

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
