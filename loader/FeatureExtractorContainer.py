from loader.implementation.AircraftInfoExtractor import AircraftInfoExtractor
from loader.implementation.ArrivalToGateTimeWeightedMeanExtractor import ArrivalToGateTimeWeightedMeanExtractor
from loader.implementation.BusyETDExtractor import BusyETDExtractor
from loader.FeaturesExtractRunner import FeatureExtractRunner
from loader.implementation.LastETDExtractor import LastETDExtractor
from loader.implementation.LastTwoETDExtractor import LastTwoETDExtractor
from loader.implementation.LastWeatherExtractor import LastWeatherExtractor
from loader.implementation.RunningRunwayInfoExtractor import RunningRunwayInfoExtractor
from loader.implementation.WeightedETDLoader import WeightedETDLoader


class FeatureExtractorContainer:

    def __init__(self):
        data_gatherer = FeatureExtractRunner()
        data_gatherer.add_feature(LastETDExtractor())
        data_gatherer.add_feature(WeightedETDLoader())
        data_gatherer.add_feature(LastTwoETDExtractor())
        data_gatherer.add_feature(BusyETDExtractor())
        data_gatherer.add_feature(ArrivalToGateTimeWeightedMeanExtractor())
        data_gatherer.add_feature(LastWeatherExtractor())
        data_gatherer.add_feature(RunningRunwayInfoExtractor())
        data_gatherer.add_feature(AircraftInfoExtractor())
        # data_gatherer.add_feature(RunwayETDMeanLoader(runways_path, etd_path))
        self.data_gatherer = data_gatherer
