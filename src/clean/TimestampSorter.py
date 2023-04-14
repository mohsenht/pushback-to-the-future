import gc

import pandas as pd

from src.constants import FILE_NAME_ETD, FILE_NAME_TBFM, FILE_NAME_TFM, FILE_NAME_RUNWAYS, FILE_NAME_STANDTIMES, \
    FILE_NAME_LAMP, FILE_NAME_CONFIG, CONFIG_COLUMN_TIMESTAMP, LAMP_COLUMN_TIMESTAMP, \
    LAMP_COLUMN_FORECAST_TIMESTAMP, STANDTIMES_COLUMN_TIMESTAMP, STANDTIMES_COLUMN_ARRIVAL_STAND_ACTUAL_TIME, \
    RUNWAYS_COLUMN_TIMESTAMP, RUNWAYS_COLUMN_ARRIVAL_RUNWAY_ACTUAL_TIME, ETD_COLUMN_TIMESTAMP, TBFM_COLUMN_FLIGHT_ID, \
    TFM_COLUMN_TIMESTAMP, TBFM_COLUMN_SCHEDULED_RUNWAY_ESTIMATED_TIME, TFM_COLUMN_ARRIVAL_RUNWAY_ESTIMATED_TIME, \
    ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME
from src.path_generator_utility import path_generator


def sort_csv_files(airport):
    etd = pd.read_csv(
        path_generator(airport, FILE_NAME_ETD),
        parse_dates=[
            ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME,
            ETD_COLUMN_TIMESTAMP
        ],
    ).sort_values(ETD_COLUMN_TIMESTAMP)

    pd.to_csv(etd, path_generator(airport, FILE_NAME_ETD))
    del etd
    gc.collect()

    tbfm = pd.read_csv(
        path_generator(airport, FILE_NAME_TBFM),
        parse_dates=[
            TBFM_COLUMN_SCHEDULED_RUNWAY_ESTIMATED_TIME,
            TBFM_COLUMN_FLIGHT_ID
        ],
    ).sort_values(TBFM_COLUMN_FLIGHT_ID)

    pd.to_csv(tbfm, path_generator(airport, FILE_NAME_TBFM))
    del tbfm
    gc.collect()

    tfm = pd.read_csv(
        path_generator(airport, FILE_NAME_TFM),
        parse_dates=[
            TFM_COLUMN_ARRIVAL_RUNWAY_ESTIMATED_TIME,
            TFM_COLUMN_TIMESTAMP
        ],
    ).sort_values(TFM_COLUMN_TIMESTAMP)

    pd.to_csv(tfm, path_generator(airport, FILE_NAME_TFM))
    del tfm
    gc.collect()

    runways = pd.read_csv(
        path_generator(airport, FILE_NAME_RUNWAYS),
        parse_dates=[
            RUNWAYS_COLUMN_ARRIVAL_RUNWAY_ACTUAL_TIME,
            RUNWAYS_COLUMN_TIMESTAMP
        ],
    ).sort_values(RUNWAYS_COLUMN_TIMESTAMP)

    pd.to_csv(runways, path_generator(airport, FILE_NAME_RUNWAYS))
    del runways
    gc.collect()

    standtimes = pd.read_csv(
        path_generator(airport, FILE_NAME_STANDTIMES),
        parse_dates=[
            STANDTIMES_COLUMN_ARRIVAL_STAND_ACTUAL_TIME,
            STANDTIMES_COLUMN_TIMESTAMP
        ],
    ).sort_values(STANDTIMES_COLUMN_TIMESTAMP)

    pd.to_csv(standtimes, path_generator(airport, FILE_NAME_STANDTIMES))
    del standtimes
    gc.collect()

    weather = pd.read_csv(
        path_generator(airport, FILE_NAME_LAMP),
        parse_dates=[
            LAMP_COLUMN_FORECAST_TIMESTAMP,
            LAMP_COLUMN_TIMESTAMP
        ],
    ).sort_values(LAMP_COLUMN_TIMESTAMP)

    pd.to_csv(weather, path_generator(airport, FILE_NAME_LAMP))
    del weather
    gc.collect()

    config = pd.read_csv(
        path_generator(airport, FILE_NAME_CONFIG),
        parse_dates=[
            CONFIG_COLUMN_TIMESTAMP
        ],
    ).sort_values(CONFIG_COLUMN_TIMESTAMP)

    pd.to_csv(config, path_generator(airport, FILE_NAME_CONFIG))
    del config
    gc.collect()
