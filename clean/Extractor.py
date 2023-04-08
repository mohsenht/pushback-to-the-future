from clean.TypeContainer import TypeContainer
from clean.TypeExtractor import TypeExtractor
from constants import FILE_NAME_MFS, FILE_NAME_CONFIG, CONFIG_COLUMN_DEPARTURE_RUNWAYS, \
    MFS_COLUMN_AIRCRAFT_TYPE, MFS_COLUMN_AIRCRAFT_ENGINE_CLASS, \
    MFS_COLUMN_MAJOR_CARRIER, MFS_COLUMN_FLIGHT_TYPE, CONFIG_COLUMN_ARRIVAL_RUNWAYS
from path_generator_utility import types_path_generator


class Extractor:

    def __init__(self, file_path, airport_name):
        self.file_path = file_path
        self.airport_name = airport_name

    def extract_runways(self):
        departure_runways = TypeExtractor(self.airport_name, self.file_path, FILE_NAME_CONFIG,
                                          CONFIG_COLUMN_DEPARTURE_RUNWAYS).extract_types()
        arrival_runways = TypeExtractor(self.airport_name, self.file_path, FILE_NAME_CONFIG,
                                        CONFIG_COLUMN_ARRIVAL_RUNWAYS).extract_types()
        arrival_runways.extend(departure_runways)
        runways = set(arrival_runways)
        new_set = set()
        for string in runways:
            items = string.split(', ')
            for item in items:
                new_set.add(item.strip())
        return list(new_set)

    def extract(self):
        container = TypeContainer(
            TypeExtractor(
                self.airport_name, self.file_path, FILE_NAME_MFS, MFS_COLUMN_AIRCRAFT_TYPE).extract_types(),
            TypeExtractor(
                self.airport_name, self.file_path, FILE_NAME_MFS, MFS_COLUMN_AIRCRAFT_ENGINE_CLASS).extract_types(),
            TypeExtractor(
                self.airport_name, self.file_path, FILE_NAME_MFS, MFS_COLUMN_MAJOR_CARRIER).extract_types(),
            TypeExtractor(
                self.airport_name, self.file_path, FILE_NAME_MFS, MFS_COLUMN_FLIGHT_TYPE).extract_types(),
            self.extract_runways()
        )

        container.to_json_file(types_path_generator(self.airport_name))
