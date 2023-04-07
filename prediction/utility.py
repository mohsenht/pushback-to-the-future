from datetime import timedelta

import pandas as pd


def crop_data_in_30h(now: pd.Timestamp, data: pd.DataFrame):
    return data.loc[
        (data.timestamp > now - timedelta(hours=30)) & (data.timestamp <= now)
        ]
