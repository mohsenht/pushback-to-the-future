import math
from typing import List
import cudf

from src.constants import SEPARATOR


class TypeExtractor:
    def __init__(self, airport_name, file_path, file_name, column_name):
        self.airport_name = airport_name
        self.file_path = file_path
        self.file_name = file_name
        self.column_name = column_name

    def extract_types(self) -> List[str]:
        dataframe = cudf.read_csv(f"{self.file_path}{self.airport_name}{SEPARATOR}{self.airport_name}_{self.file_name}.csv")
        lst = dataframe[self.column_name].unique().tolist()
        lst = [str(x) if (isinstance(x, float) and math.isnan(x)) else str(x) if isinstance(x, float) else 'unknown' if x is None else str(x) for x in lst]
        if 'nan' not in lst:
            lst.append('nan')
        return lst
