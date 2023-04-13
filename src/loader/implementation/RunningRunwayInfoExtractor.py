import cudf
import pandas as pd

from src.constants import CONFIG_COLUMN_ARRIVAL_RUNWAYS, CONFIG_COLUMN_DEPARTURE_RUNWAYS
from src.model.Input import Input
from src.clean.TypeContainer import TypeContainer
from src.loader.FeatureExtractor import FeatureExtractor


class RunningRunwayInfoExtractor(FeatureExtractor):

    def load_data(self,
                  now: pd.Timestamp,
                  data: cudf.DataFrame,
                  input_data: Input,
                  type_container: TypeContainer) -> cudf.DataFrame:
        boolean_feature_names = []
        boolean_feature_names.extend(['de_' + s for s in type_container.runways_names])
        boolean_feature_names.extend(['ar_' + s for s in type_container.runways_names])
        new_data = cudf.DataFrame(index=data.index, columns=boolean_feature_names).fillna(False)
        new_data = cudf.concat([data, new_data], axis=1)
        if input_data.config.empty:
            return new_data
        now_running_runway = input_data.config.iloc[-1]
        now_running_runway = pd.Series(now_running_runway.to_pandas().iloc[-1])
        departure_runways = str(now_running_runway[CONFIG_COLUMN_DEPARTURE_RUNWAYS]).split(', ')
        arrival_runways = str(now_running_runway[CONFIG_COLUMN_ARRIVAL_RUNWAYS]).split(', ')
        for departure_runway in departure_runways:
            new_data[f"de_{departure_runway}"] = True
        for arrival_runway in arrival_runways:
            new_data[f"ar_{arrival_runway}"] = True

        return new_data
