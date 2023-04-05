from loader.implementation.DataGatherer import DataGatherer


class Model:

    def __init__(self, airport_dict, data_gatherer: DataGatherer):
        self.airport_dict = airport_dict
        self.data_gatherer = data_gatherer
