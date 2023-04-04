from unittest import TestCase

from clean.extract.TypeExtractor import TypeExtractor
from main import separator


class TestTypeExtractor(TestCase):
    def test_load_data(self):
        expected_aircrafts = ["F900", "A321", "B738", "nan"]
        airport_name = "KCLT"
        file_path = f"data{separator}"
        file_name = "mfs"
        type_name = "aircraft"
        column_name = "aircraft_type"
        type_extractor = TypeExtractor(airport_name, file_path, file_name, column_name)
        actual_aircrafts = type_extractor.extract_types()

        assert set(expected_aircrafts) == set(actual_aircrafts), "Expected and actual aircrafts are not the same."
