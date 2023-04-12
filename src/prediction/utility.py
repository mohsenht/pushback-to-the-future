from datetime import timedelta

import cudf
import pandas as pd


def crop_data_in_30h(now: pd.Timestamp, data: cudf.DataFrame):
    return data.loc[
        (data.timestamp > now - timedelta(hours=30)) & (data.timestamp <= now)
        ]
