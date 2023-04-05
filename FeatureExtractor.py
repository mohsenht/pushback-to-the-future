from loader.implementation.AircraftInfoLoader import AircraftInfoLoader
from loader.implementation.ArrivalToGateTimeWeightedMeanLoader import ArrivalToGateTimeWeightedMeanLoader
from loader.implementation.BusyETDLoader import BusyETDLoader
from loader.implementation.DataGatherer import DataGatherer
from loader.implementation.LastETDLoader import LastETDLoader
from loader.implementation.LastTwoETDLoader import LastTwoETDLoader
from loader.implementation.LastWeatherLoader import LastWeatherLoader
from loader.implementation.RunningRunwayInfoLoader import RunningRunwayInfoLoader
from loader.implementation.WeightedETDLoader import WeightedETDLoader


class FeatureExtractor:

    def __init__(self):
        data_gatherer = DataGatherer()
        data_gatherer.add_feature(LastETDLoader())
        data_gatherer.add_feature(WeightedETDLoader())
        data_gatherer.add_feature(LastTwoETDLoader())
        data_gatherer.add_feature(BusyETDLoader())
        data_gatherer.add_feature(ArrivalToGateTimeWeightedMeanLoader())
        data_gatherer.add_feature(LastWeatherLoader())
        data_gatherer.add_feature(RunningRunwayInfoLoader())
        data_gatherer.add_feature(AircraftInfoLoader())
        # data_gatherer.add_feature(RunwayETDMeanLoader(runways_path, etd_path))
        self.data_gatherer = data_gatherer
