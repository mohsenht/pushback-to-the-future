from datetime import timedelta

import pandas as pd

from Input import Input
from clean.extract.TypeContainer import TypeContainer
from constants import cloud_category_BK, cloud_category_CL, \
    cloud_category_FEW, cloud_category_OV, cloud_category_SC, lightning_prob_N, lightning_prob_L, lightning_prob_M, \
    lightning_prob_H
from loader.DataLoader import DataLoader


class LastWeatherLoader(DataLoader):

    def load_data(self, now: pd.Timestamp, data: pd.DataFrame, input: Input, type_container: TypeContainer) -> pd.DataFrame:
        data['departure_time'] = data.last_etd.apply(lambda x: timedelta(minutes=x)) + now
        data['departure_time'] = data['departure_time'].apply(
            lambda x: (x + timedelta(minutes=30)).replace(second=0, microsecond=0, minute=0,
                                                          hour=(x + timedelta(minutes=30)).hour))
        now_weather = input.lamp[input.lamp['forecast_timestamp'].isin(data.departure_time.unique())].groupby(
            'forecast_timestamp').last()
        now_weather = now_weather.iloc[:, 1:]
        weather = pd.merge(data, now_weather, how='left', left_on='departure_time', right_on='forecast_timestamp')
        data['temperature'] = weather['temperature']
        data['wind_direction'] = weather['wind_direction']
        data['wind_speed'] = weather['wind_speed']
        data['wind_gust'] = weather['wind_gust']
        data['cloud_ceiling'] = weather['cloud_ceiling']
        data['visibility'] = weather['visibility']

        weather['cl_BK'] = weather['cloud'] == cloud_category_BK
        weather['cl_CL'] = weather['cloud'] == cloud_category_CL
        weather['cl_FEW'] = weather['cloud'] == cloud_category_FEW
        weather['cl_OV'] = weather['cloud'] == cloud_category_OV
        weather['cl_SC'] = weather['cloud'] == cloud_category_SC

        data['cl_BK'] = weather['cl_BK']
        data['cl_CL'] = weather['cl_CL']
        data['cl_FEW'] = weather['cl_FEW']
        data['cl_OV'] = weather['cl_OV']
        data['cl_SC'] = weather['cl_SC']

        weather['lp_N'] = weather['lightning_prob'] == lightning_prob_N
        weather['lp_L'] = weather['lightning_prob'] == lightning_prob_L
        weather['lp_M'] = weather['lightning_prob'] == lightning_prob_M
        weather['lp_H'] = weather['lightning_prob'] == lightning_prob_H

        data['lp_N'] = weather['lp_N']
        data['lp_L'] = weather['lp_L']
        data['lp_M'] = weather['lp_M']
        data['lp_H'] = weather['lp_H']

        data.drop('departure_time', axis=1, inplace=True)
        return data
