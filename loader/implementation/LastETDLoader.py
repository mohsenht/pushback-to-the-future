from pandas import DataFrame, Timestamp

from Input import Input
from clean.extract.TypeContainer import TypeContainer
from constants import flight_id
from loader.DataLoader import DataLoader


class LastETDLoader(DataLoader):

    def load_data(self, now: Timestamp, data: DataFrame, input: Input, type_container: TypeContainer) -> DataFrame:
        latest_now_etd = input.etd.groupby(flight_id).last().departure_runway_estimated_time
        departure_runway_estimated_time = data.merge(
            latest_now_etd, how="left", on=flight_id
        ).departure_runway_estimated_time
        data["last_etd"] = (departure_runway_estimated_time - now).dt.total_seconds() / 60
        data["last_etd"] = data.last_etd.clip(lower=0).astype(int)
        return data
