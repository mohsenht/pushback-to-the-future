import pandas as pd

from src.constants import COLUMN_NAME_TIMESTAMP


class CSVReader:

    def __init__(self, path, chunk_size, parse_dates):
        self.data = pd.DataFrame()
        self.path = path
        self.chunk_size = chunk_size
        self.parse_dates = parse_dates
        self.skip_rows = 0
        self.file = open(self.path, 'r', encoding='utf-8')
        self.filereader = pd.read_csv(
            self.file,
            chunksize=self.chunk_size,
            skiprows=self.skip_rows,
            parse_dates=self.parse_dates
        )

    def get_data(self, start_time: pd.Timestamp, end_time: pd.Timestamp) -> pd.DataFrame:
        if not self.data.empty and self.data.iloc[-1][COLUMN_NAME_TIMESTAMP] > end_time:
            return self.delete_redundant_first_rows(start_time, len(self.data))
        for chunk in self.filereader:
            if chunk.empty:
                return self.delete_redundant_first_rows(start_time, len(self.data))
            current_chunk_size = len(chunk)
            self.skip_rows = self.skip_rows + current_chunk_size
            self.data = pd.concat([self.data, chunk], axis=0, ignore_index=True)
            if chunk.iloc[-1][COLUMN_NAME_TIMESTAMP] > end_time:
                return self.delete_redundant_first_rows(start_time, len(self.data))
        return self.delete_redundant_first_rows(start_time, len(self.data))

    def delete_redundant_first_rows(self, start_time, length):
        if length < (self.chunk_size / 3):
            return self.data
        length = int(length / 2)
        if self.data.iloc[length][COLUMN_NAME_TIMESTAMP] >= start_time:
            return self.delete_redundant_first_rows(start_time, length)
        else:
            self.data = self.data.loc[length:]
            return self.data

    def close_file(self):
        self.filereader.close()
        self.file.close()
        del self.data
