from abc import ABC, abstractmethod

import cudf

from src.model.Input import Input
from src.clean.TypeContainer import TypeContainer


class FeatureExtractor(ABC):

    @abstractmethod
    def load_data(self,
                  now: Timestamp,
                  data: cudf.DataFrame,
                  input_data: Input,
                  type_container: TypeContainer) -> cudf.DataFrame:
        pass
