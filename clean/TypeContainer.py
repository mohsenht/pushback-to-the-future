import json

from constants import MFS_COLUMN_AIRCRAFT_TYPE, MFS_COLUMN_AIRCRAFT_ENGINE_CLASS, MFS_COLUMN_MAJOR_CARRIER, \
    MFS_COLUMN_FLIGHT_TYPE


class TypeContainer:
    def __init__(self,
                 aircraft_type,
                 aircraft_engine_class,
                 major_carrier,
                 flight_type,
                 runways_names):
        self.aircraft_type = aircraft_type
        self.aircraft_engine_class = aircraft_engine_class
        self.major_carrier = major_carrier
        self.flight_type = flight_type
        self.runways_names = runways_names

    @classmethod
    def from_file(cls, file_path):
        with open(file_path, "r") as f:
            json_string = f.read()
        my_dict = json.loads(json_string)
        return cls(my_dict[MFS_COLUMN_AIRCRAFT_TYPE],
                   my_dict[MFS_COLUMN_AIRCRAFT_ENGINE_CLASS],
                   my_dict[MFS_COLUMN_MAJOR_CARRIER],
                   my_dict[MFS_COLUMN_FLIGHT_TYPE],
                   my_dict["runways_names"]
                   )

    def to_json_file(self, file_path):
        my_json_string = json.dumps(self.__dict__)
        with open(file_path, "w") as f:
            f.write(my_json_string)
