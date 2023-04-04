from pandas import DataFrame, Timestamp
import pandas as pd

from clean.extract.TypeContainer import TypeContainer
from constants import mfs_column_aircraft_type, mfs_column_major_carrier, mfs_column_aircraft_engine_class, \
    mfs_column_flight_type, file_name_mfs
from loader.DataLoader import DataLoader
from path_generator import path_generator


class AircraftInfoLoader(DataLoader):
    def __init__(self, airport, type_file_path):
        self.mfs = pd.read_csv(path_generator(airport, file_name_mfs),)
        self.container = TypeContainer.from_file(type_file_path)

    def load_data(self, now: Timestamp, data: DataFrame) -> DataFrame:
        boolean_feature_names = []
        boolean_feature_names.extend(['a_' + s for s in self.container.aircraft_type])
        boolean_feature_names.extend(['e_' + s for s in self.container.aircraft_engine_class])
        boolean_feature_names.extend(['f_' + s for s in self.container.flight_type])
        boolean_feature_names.extend(['m_' + s for s in self.container.major_carrier])
        new_data = pd.DataFrame(False, index=data.index, columns=boolean_feature_names)
        data = pd.concat([data, new_data], axis=1)
        filtered_mfs = self.mfs[self.mfs['gufi'].isin(data.gufi)]
        results = data.apply(self.fill_mfs_for_each_flight, args=(filtered_mfs,), axis=1)

        return results

    def fill_mfs_for_each_flight(self, x, filtered_mfs):
        mfs_dataframe = filtered_mfs.loc[filtered_mfs.gufi == x.gufi]
        if mfs_dataframe.empty:
            return
        mfs = mfs_dataframe.iloc[0]
        aircraft_type = f"a_{mfs[mfs_column_aircraft_type]}"
        aircraft_engine_class = f"e_{mfs[mfs_column_aircraft_engine_class]}"
        major_carrier = f"m_{mfs[mfs_column_major_carrier]}"
        flight_type = f"f_{mfs[mfs_column_flight_type]}"

        if hasattr(x, aircraft_type):
            x[aircraft_type] = True
        else:
            x["a_nan"] = True
        if hasattr(x, aircraft_engine_class):
            x[aircraft_engine_class] = True
        else:
            x["e_nan"] = True
        if hasattr(x, major_carrier):
            x[major_carrier] = True
        else:
            x["m_nan"] = True
        if hasattr(x, flight_type):
            x[flight_type] = True
        else:
            x["f_nan"] = True

        return x
