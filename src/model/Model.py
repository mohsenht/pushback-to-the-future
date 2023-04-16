from src.loader.FeaturesExtractRunner import FeatureExtractRunner


class Model:

    def __init__(
            self,
            airport_dict,
            data_gatherer: FeatureExtractRunner,
            data_gatherer_2: FeatureExtractRunner,
    ):
        self.airport_dict = airport_dict
        self.data_gatherer = data_gatherer
        self.data_gatherer_2 = data_gatherer_2
