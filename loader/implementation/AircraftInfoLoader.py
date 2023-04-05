import pandas as pd

from Input import Input
from clean.extract.TypeContainer import TypeContainer
from constants import mfs_column_aircraft_type, mfs_column_major_carrier, mfs_column_aircraft_engine_class, \
    mfs_column_flight_type, flight_id
from loader.DataLoader import DataLoader


class AircraftInfoLoader(DataLoader):

    def load_data(self, now: pd.Timestamp, data: pd.DataFrame, input: Input,
                  type_container: TypeContainer) -> pd.DataFrame:
        boolean_feature_names = []
        boolean_feature_names.extend(['a_' + s for s in type_container.aircraft_type])
        boolean_feature_names.extend(['e_' + s for s in type_container.aircraft_engine_class])
        boolean_feature_names.extend(['f_' + s for s in type_container.flight_type])
        boolean_feature_names.extend(['m_' + s for s in type_container.major_carrier])
        new_data = pd.DataFrame(False, index=data.index, columns=boolean_feature_names)
        data = pd.concat([data, new_data], axis=1)
        filtered_mfs = input.mfs[input.mfs[flight_id].isin(data.gufi)]
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
