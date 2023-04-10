import pandas as pd

from src.model.Input import Input
from src.clean.TypeContainer import TypeContainer
from src.constants import MFS_COLUMN_AIRCRAFT_TYPE, MFS_COLUMN_MAJOR_CARRIER, MFS_COLUMN_AIRCRAFT_ENGINE_CLASS, \
    MFS_COLUMN_FLIGHT_TYPE, FLIGHT_ID
from src.loader.FeatureExtractor import FeatureExtractor


class AircraftInfoExtractor(FeatureExtractor):

    def load_data(self,
                  now: pd.Timestamp,
                  data: pd.DataFrame,
                  input_data: Input,
                  type_container: TypeContainer) -> pd.DataFrame:
        boolean_feature_names = []
        boolean_feature_names.extend(['a_' + s for s in type_container.aircraft_type])
        boolean_feature_names.extend(['e_' + s for s in type_container.aircraft_engine_class])
        boolean_feature_names.extend(['f_' + s for s in type_container.flight_type])
        boolean_feature_names.extend(['m_' + s for s in type_container.major_carrier])
        new_data = pd.DataFrame(False, index=data.index, columns=boolean_feature_names)
        data = pd.concat([data, new_data], axis=1)
        filtered_mfs = input_data.mfs[input_data.mfs[FLIGHT_ID].isin(data.gufi)]
        results = data.apply(self.fill_mfs_for_each_flight, args=(filtered_mfs,), axis=1)

        return results

    def fill_mfs_for_each_flight(self, x, filtered_mfs):
        mfs_dataframe = filtered_mfs.loc[filtered_mfs.gufi == x.gufi]
        if mfs_dataframe.empty:
            return x
        mfs = mfs_dataframe.iloc[0]
        aircraft_type = f"a_{mfs[MFS_COLUMN_AIRCRAFT_TYPE]}"
        aircraft_engine_class = f"e_{mfs[MFS_COLUMN_AIRCRAFT_ENGINE_CLASS]}"
        major_carrier = f"m_{mfs[MFS_COLUMN_MAJOR_CARRIER]}"
        flight_type = f"f_{mfs[MFS_COLUMN_FLIGHT_TYPE]}"

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
