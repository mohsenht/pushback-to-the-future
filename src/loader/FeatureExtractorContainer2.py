from src.loader.FeaturesExtractRunner import FeatureExtractRunner
from src.loader.implementation.AircraftInfoExtractor import AircraftInfoExtractor
from src.loader.implementation.ArrivalToGateTimeWeightedMeanExtractor import ArrivalToGateTimeWeightedMeanExtractor
from src.loader.implementation.BusyETDExtractor import BusyETDExtractor
from src.loader.implementation.BusyTBFMExtractor import BusyTBFMExtractor
from src.loader.implementation.BusyTFMExtractor import BusyTFMExtractor
from src.loader.implementation.LastTwoETDExtractor import LastTwoETDExtractor
from src.loader.implementation.LastWeatherExtractor import LastWeatherExtractor
from src.loader.implementation.RunningRunwayInfoExtractor import RunningRunwayInfoExtractor
from src.loader.implementation.WeightedETDLoader import WeightedETDLoader


class FeatureExtractorContainer2:

    def __init__(self):
        data_gatherer = FeatureExtractRunner()
        data_gatherer.add_feature(BusyETDExtractor())
        data_gatherer.add_feature(BusyTFMExtractor())
        data_gatherer.add_feature(BusyTBFMExtractor())
        data_gatherer.add_feature(LastWeatherExtractor())
        data_gatherer.add_feature(WeightedETDLoader())
        data_gatherer.add_feature(LastTwoETDExtractor())
        data_gatherer.add_feature(ArrivalToGateTimeWeightedMeanExtractor())
        data_gatherer.add_feature(RunningRunwayInfoExtractor())
        data_gatherer.add_feature(AircraftInfoExtractor())
        self.data_gatherer = data_gatherer
