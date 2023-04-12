import cudf
import pandas as pd

from src.clean.TypeContainer import TypeContainer
from src.constants import MFS_COLUMN_AIRCRAFT_TYPE, MFS_COLUMN_MAJOR_CARRIER, MFS_COLUMN_AIRCRAFT_ENGINE_CLASS, \
    MFS_COLUMN_FLIGHT_TYPE, FLIGHT_ID, MFS_COLUMN_IS_DEPARTURE
from src.loader.FeatureExtractor import FeatureExtractor
from src.model.Input import Input


class AircraftInfoExtractor(FeatureExtractor):

    def load_data(self,
                  now: pd.Timestamp,
                  data: cudf.DataFrame,
                  input_data: Input,
                  type_container: TypeContainer) -> cudf.DataFrame:
        filtered_mfs = input_data.mfs[input_data.mfs[FLIGHT_ID].isin(data.gufi)]
        filtered_mfs.drop(columns=[MFS_COLUMN_IS_DEPARTURE])
        data = data.merge(
            filtered_mfs,
            how="left",
            on=FLIGHT_ID
        )
        return data
