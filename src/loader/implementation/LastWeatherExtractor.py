from datetime import timedelta

import pandas as pd

from src.model.Input import Input
from src.clean.TypeContainer import TypeContainer
from src.constants import CLOUD_CATEGORY_BK, CLOUD_CATEGORY_CL, \
    CLOUD_CATEGORY_FEW, CLOUD_CATEGORY_OV, CLOUD_CATEGORY_SC, LIGHTNING_PROB_CATEGORY_N, LIGHTNING_PROB_CATEGORY_L, \
    LIGHTNING_PROB_CATEGORY_M, \
    LIGHTNING_PROB_CATEGORY_H, LAMP_COLUMN_TEMPERATURE, LAMP_COLUMN_WIND_DIRECTION, \
    LAMP_COLUMN_WIND_SPEED, \
    LAMP_COLUMN_WIND_GUST, LAMP_COLUMN_CLOUD_CEILING, LAMP_COLUMN_VISIBILITY, LAMP_COLUMN_FORECAST_TIMESTAMP, \
    LAMP_COLUMN_CLOUD, CLOUD_CATEGORY_PREFIX, LAMP_COLUMN_LIGHTNING_PROB, LIGHTNING_PROB_CATEGORY_PREFIX, \
    LAMP_COLUMN_PRECIP
from src.loader.FeatureExtractor import FeatureExtractor


class LastWeatherExtractor(FeatureExtractor):
    DEPARTURE_TIME = 'departure_time'

    def load_data(self,
                  now: pd.Timestamp,
                  data: pd.DataFrame,
                  input_data: Input,
                  type_container: TypeContainer) -> pd.DataFrame:
        data[LastWeatherExtractor.DEPARTURE_TIME] = data.last_etd.apply(lambda x: timedelta(minutes=x) + now)
        data[LastWeatherExtractor.DEPARTURE_TIME] = data[LastWeatherExtractor.DEPARTURE_TIME].apply(
            lambda x: (x + timedelta(minutes=30)).replace(
                second=0,
                microsecond=0,
                minute=0,
                hour=(x + timedelta(minutes=30)).hour)
        )
        now_weather = input_data.lamp[
            input_data.lamp[LAMP_COLUMN_FORECAST_TIMESTAMP].isin(data.departure_time.unique())
        ] \
            .groupby(
            LAMP_COLUMN_FORECAST_TIMESTAMP).last()
        now_weather = now_weather.iloc[:, 1:]
        now_weather = now_weather.reset_index()
        weather = pd.merge(
            data,
            now_weather,
            how='left',
            left_on=LastWeatherExtractor.DEPARTURE_TIME,
            right_on=LAMP_COLUMN_FORECAST_TIMESTAMP
        )

        data[LAMP_COLUMN_TEMPERATURE] = weather[LAMP_COLUMN_TEMPERATURE]
        data[LAMP_COLUMN_WIND_DIRECTION] = weather[LAMP_COLUMN_WIND_DIRECTION]
        data[LAMP_COLUMN_WIND_SPEED] = weather[LAMP_COLUMN_WIND_SPEED]
        data[LAMP_COLUMN_WIND_GUST] = weather[LAMP_COLUMN_WIND_GUST]
        data[LAMP_COLUMN_CLOUD_CEILING] = weather[LAMP_COLUMN_CLOUD_CEILING]
        data[LAMP_COLUMN_VISIBILITY] = weather[LAMP_COLUMN_VISIBILITY]

        weather[f"{CLOUD_CATEGORY_PREFIX}{CLOUD_CATEGORY_BK}"] = weather[LAMP_COLUMN_CLOUD] == CLOUD_CATEGORY_BK
        weather[f"{CLOUD_CATEGORY_PREFIX}{CLOUD_CATEGORY_CL}"] = weather[LAMP_COLUMN_CLOUD] == CLOUD_CATEGORY_CL
        weather[f"{CLOUD_CATEGORY_PREFIX}{CLOUD_CATEGORY_FEW}"] = weather[LAMP_COLUMN_CLOUD] == CLOUD_CATEGORY_FEW
        weather[f"{CLOUD_CATEGORY_PREFIX}{CLOUD_CATEGORY_OV}"] = weather[LAMP_COLUMN_CLOUD] == CLOUD_CATEGORY_OV
        weather[f"{CLOUD_CATEGORY_PREFIX}{CLOUD_CATEGORY_SC}"] = weather[LAMP_COLUMN_CLOUD] == CLOUD_CATEGORY_SC

        data[f"{CLOUD_CATEGORY_PREFIX}{CLOUD_CATEGORY_BK}"] = weather[f"{CLOUD_CATEGORY_PREFIX}{CLOUD_CATEGORY_BK}"]
        data[f"{CLOUD_CATEGORY_PREFIX}{CLOUD_CATEGORY_CL}"] = weather[f"{CLOUD_CATEGORY_PREFIX}{CLOUD_CATEGORY_CL}"]
        data[f"{CLOUD_CATEGORY_PREFIX}{CLOUD_CATEGORY_FEW}"] = weather[f"{CLOUD_CATEGORY_PREFIX}{CLOUD_CATEGORY_FEW}"]
        data[f"{CLOUD_CATEGORY_PREFIX}{CLOUD_CATEGORY_OV}"] = weather[f"{CLOUD_CATEGORY_PREFIX}{CLOUD_CATEGORY_OV}"]
        data[f"{CLOUD_CATEGORY_PREFIX}{CLOUD_CATEGORY_SC}"] = weather[f"{CLOUD_CATEGORY_PREFIX}{CLOUD_CATEGORY_SC}"]

        weather[f"{LIGHTNING_PROB_CATEGORY_PREFIX}{LIGHTNING_PROB_CATEGORY_N}"] = weather[
                                                                                      LAMP_COLUMN_LIGHTNING_PROB] == LIGHTNING_PROB_CATEGORY_N
        weather[f"{LIGHTNING_PROB_CATEGORY_PREFIX}{LIGHTNING_PROB_CATEGORY_L}"] = weather[
                                                                                      LAMP_COLUMN_LIGHTNING_PROB] == LIGHTNING_PROB_CATEGORY_L
        weather[f"{LIGHTNING_PROB_CATEGORY_PREFIX}{LIGHTNING_PROB_CATEGORY_M}"] = weather[
                                                                                      LAMP_COLUMN_LIGHTNING_PROB] == LIGHTNING_PROB_CATEGORY_M
        weather[f"{LIGHTNING_PROB_CATEGORY_PREFIX}{LIGHTNING_PROB_CATEGORY_H}"] = weather[
                                                                                      LAMP_COLUMN_LIGHTNING_PROB] == LIGHTNING_PROB_CATEGORY_H

        data[f"{LIGHTNING_PROB_CATEGORY_PREFIX}{LIGHTNING_PROB_CATEGORY_N}"] = weather[
            f"{LIGHTNING_PROB_CATEGORY_PREFIX}{LIGHTNING_PROB_CATEGORY_N}"]
        data[f"{LIGHTNING_PROB_CATEGORY_PREFIX}{LIGHTNING_PROB_CATEGORY_L}"] = weather[
            f"{LIGHTNING_PROB_CATEGORY_PREFIX}{LIGHTNING_PROB_CATEGORY_L}"]
        data[f"{LIGHTNING_PROB_CATEGORY_PREFIX}{LIGHTNING_PROB_CATEGORY_M}"] = weather[
            f"{LIGHTNING_PROB_CATEGORY_PREFIX}{LIGHTNING_PROB_CATEGORY_M}"]
        data[f"{LIGHTNING_PROB_CATEGORY_PREFIX}{LIGHTNING_PROB_CATEGORY_H}"] = weather[
            f"{LIGHTNING_PROB_CATEGORY_PREFIX}{LIGHTNING_PROB_CATEGORY_H}"]

        weather[LAMP_COLUMN_PRECIP].fillna(False, inplace=True)
        data[LAMP_COLUMN_PRECIP] = weather[LAMP_COLUMN_PRECIP]

        data.drop(LastWeatherExtractor.DEPARTURE_TIME, axis=1, inplace=True)
        return data
