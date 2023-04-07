from clean.TypeContainer import TypeContainer
from clean.TypeExtractor import TypeExtractor
from constants import file_name_mfs, file_name_config, runways_column_departure_runways, \
    runways_column_arrival_runways, mfs_column_aircraft_type, mfs_column_aircraft_engine_class, \
    mfs_column_major_carrier, mfs_column_flight_type
from path_generator_utility import types_path_generator


class Extractor:

    def __init__(self, file_path, airport_name):
        self.file_path = file_path
        self.airport_name = airport_name

    def extract_runways(self):
        departure_runways = TypeExtractor(self.airport_name, self.file_path, file_name_config,
                                          runways_column_departure_runways).extract_types()
        arrival_runways = TypeExtractor(self.airport_name, self.file_path, file_name_config,
                                        runways_column_arrival_runways).extract_types()
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
                self.airport_name, self.file_path, file_name_mfs, mfs_column_aircraft_type).extract_types(),
            TypeExtractor(
                self.airport_name, self.file_path, file_name_mfs, mfs_column_aircraft_engine_class).extract_types(),
            TypeExtractor(
                self.airport_name, self.file_path, file_name_mfs, mfs_column_major_carrier).extract_types(),
            TypeExtractor(
                self.airport_name, self.file_path, file_name_mfs, mfs_column_flight_type).extract_types(),
            self.extract_runways()
        )

        container.to_json_file(types_path_generator(self.airport_name))
