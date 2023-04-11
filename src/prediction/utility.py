from datetime import timedelta

import cudf


def crop_data_in_30h(now: Timestamp, data: cudf.DataFrame):
    return data.loc[
        (data.timestamp > now - timedelta(hours=30)) & (data.timestamp <= now)
        ]
