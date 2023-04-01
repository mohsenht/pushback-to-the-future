from pandas import DataFrame, Timestamp
import pandas as pd
from datetime import timedelta

from loader.DataLoader import DataLoader


class LastWeatherLoader(DataLoader):
    def __init__(self, file_path):
        self.weather = pd.read_csv(
            file_path,
            parse_dates=["forecast_timestamp", "timestamp"],
        ).sort_values("timestamp")

    def load_data(self, now: Timestamp, data: DataFrame) -> DataFrame:
        now_weather = self.weather.loc[
            (self.weather.timestamp > now - timedelta(hours=30)) & (self.weather.timestamp <= now)]
        data['departure_time'] = data.last_etd.apply(lambda x: timedelta(minutes=x)) + now
        data['departure_time'] = data['departure_time'].apply(
            lambda x: (x + timedelta(minutes=30)).replace(second=0, microsecond=0, minute=0, hour=(x + timedelta(minutes=30)).hour))
        now_weather = now_weather[now_weather['forecast_timestamp'].isin(data.departure_time.unique())].groupby('forecast_timestamp').last()
        now_weather = now_weather.iloc[:, 1:]
        weather = pd.merge(data, now_weather, how='left', left_on='departure_time', right_on='forecast_timestamp')
        data['temperature'] = weather['temperature']
        data['wind_direction'] = weather['wind_direction']
        data['wind_speed'] = weather['wind_speed']
        data['wind_gust'] = weather['wind_gust']
        data['cloud_ceiling'] = weather['cloud_ceiling']
        data['visibility'] = weather['visibility']
        #data['cloud'] = weather['cloud'] # todo this dropping is temporary
        #data['lightning_prob'] = weather['lightning_prob'] # todo this dropping is temporary

        data.drop('departure_time', axis=1, inplace=True)
        return data
