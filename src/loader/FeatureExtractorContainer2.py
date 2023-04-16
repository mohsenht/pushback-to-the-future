from src.loader.FeaturesExtractRunner import FeatureExtractRunner
from src.loader.implementation.BusyETDExtractor import BusyETDExtractor
from src.loader.implementation.BusyTBFMExtractor import BusyTBFMExtractor
from src.loader.implementation.BusyTFMExtractor import BusyTFMExtractor
from src.loader.implementation.LastWeatherExtractor import LastWeatherExtractor


class FeatureExtractorContainer2:

    def __init__(self):
        data_gatherer = FeatureExtractRunner()
        data_gatherer.add_feature(BusyETDExtractor())
        data_gatherer.add_feature(BusyTFMExtractor())
        data_gatherer.add_feature(BusyTBFMExtractor())
        data_gatherer.add_feature(LastWeatherExtractor())
        self.data_gatherer = data_gatherer
