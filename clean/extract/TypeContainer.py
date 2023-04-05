import json

from constants import mfs_column_aircraft_type, mfs_column_aircraft_engine_class, mfs_column_major_carrier, \
    mfs_column_flight_type


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
        return cls(my_dict[mfs_column_aircraft_type],
                   my_dict[mfs_column_aircraft_engine_class],
                   my_dict[mfs_column_major_carrier],
                   my_dict[mfs_column_flight_type],
                   my_dict["runways_names"]
                   )

    def to_json_file(self, file_path):
        my_json_string = json.dumps(self.__dict__)
        with open(file_path, "w") as f:
            f.write(my_json_string)
