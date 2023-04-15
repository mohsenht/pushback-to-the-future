import gc

import pandas as pd

from src.constants import FILE_NAME_ETD, FILE_NAME_TBFM, FILE_NAME_TFM, FILE_NAME_RUNWAYS, FILE_NAME_STANDTIMES, \
    FILE_NAME_LAMP, FILE_NAME_CONFIG, CONFIG_COLUMN_TIMESTAMP, LAMP_COLUMN_TIMESTAMP, \
    LAMP_COLUMN_FORECAST_TIMESTAMP, STANDTIMES_COLUMN_TIMESTAMP, STANDTIMES_COLUMN_ARRIVAL_STAND_ACTUAL_TIME, \
    RUNWAYS_COLUMN_TIMESTAMP, RUNWAYS_COLUMN_ARRIVAL_RUNWAY_ACTUAL_TIME, ETD_COLUMN_TIMESTAMP, COLUMN_NAME_TIMESTAMP, \
    TFM_COLUMN_TIMESTAMP, TBFM_COLUMN_SCHEDULED_RUNWAY_ESTIMATED_TIME, TFM_COLUMN_ARRIVAL_RUNWAY_ESTIMATED_TIME, \
    ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME
from src.path_generator_utility import path_generator, labels_path_generator


def sort_csv_files(airport):
    print(f"sort {airport} data")

    labels = pd.read_csv(
        labels_path_generator(airport),
        parse_dates=[
            COLUMN_NAME_TIMESTAMP
        ]
    ).sort_values(COLUMN_NAME_TIMESTAMP)

    labels.to_csv(labels_path_generator(airport), index=False)
    del labels
    gc.collect()

    etd = pd.read_csv(
        path_generator(airport, FILE_NAME_ETD),
        parse_dates=[
            ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME,
            ETD_COLUMN_TIMESTAMP
        ],
    ).sort_values(ETD_COLUMN_TIMESTAMP)

    etd.to_csv(path_generator(airport, FILE_NAME_ETD), index=False)
    del etd
    gc.collect()

    tbfm = pd.read_csv(
        path_generator(airport, FILE_NAME_TBFM),
        parse_dates=[
            TBFM_COLUMN_SCHEDULED_RUNWAY_ESTIMATED_TIME,
            COLUMN_NAME_TIMESTAMP
        ],
    ).sort_values(COLUMN_NAME_TIMESTAMP)

    tbfm.to_csv(path_generator(airport, FILE_NAME_TBFM), index=False)
    del tbfm
    gc.collect()

    tfm = pd.read_csv(
        path_generator(airport, FILE_NAME_TFM),
        parse_dates=[
            TFM_COLUMN_ARRIVAL_RUNWAY_ESTIMATED_TIME,
            TFM_COLUMN_TIMESTAMP
        ],
    ).sort_values(TFM_COLUMN_TIMESTAMP)

    tfm.to_csv(path_generator(airport, FILE_NAME_TFM), index=False)
    del tfm
    gc.collect()

    runways = pd.read_csv(
        path_generator(airport, FILE_NAME_RUNWAYS),
        parse_dates=[
            RUNWAYS_COLUMN_ARRIVAL_RUNWAY_ACTUAL_TIME,
            RUNWAYS_COLUMN_TIMESTAMP
        ],
    ).sort_values(RUNWAYS_COLUMN_TIMESTAMP)

    runways.to_csv(path_generator(airport, FILE_NAME_RUNWAYS), index=False)
    del runways
    gc.collect()

    standtimes = pd.read_csv(
        path_generator(airport, FILE_NAME_STANDTIMES),
        parse_dates=[
            STANDTIMES_COLUMN_ARRIVAL_STAND_ACTUAL_TIME,
            STANDTIMES_COLUMN_TIMESTAMP
        ],
    ).sort_values(STANDTIMES_COLUMN_TIMESTAMP)

    standtimes.to_csv(path_generator(airport, FILE_NAME_STANDTIMES), index=False)
    del standtimes
    gc.collect()

    weather = pd.read_csv(
        path_generator(airport, FILE_NAME_LAMP),
        parse_dates=[
            LAMP_COLUMN_FORECAST_TIMESTAMP,
            LAMP_COLUMN_TIMESTAMP
        ],
    ).sort_values(LAMP_COLUMN_TIMESTAMP)

    weather.to_csv(path_generator(airport, FILE_NAME_LAMP), index=False)
    del weather
    gc.collect()

    config = pd.read_csv(
        path_generator(airport, FILE_NAME_CONFIG),
        parse_dates=[
            CONFIG_COLUMN_TIMESTAMP
        ],
    ).sort_values(CONFIG_COLUMN_TIMESTAMP)

    config.to_csv(path_generator(airport, FILE_NAME_CONFIG), index=False)
    del config
    gc.collect()
