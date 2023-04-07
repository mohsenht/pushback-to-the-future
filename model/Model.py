from loader.FeaturesExtractRunner import FeatureExtractRunner


class Model:

    def __init__(self, airport_dict, data_gatherer: FeatureExtractRunner):
        self.airport_dict = airport_dict
        self.data_gatherer = data_gatherer
