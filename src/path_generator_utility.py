from src.constants import TRAIN_PATH, FILE_NAME_LABELS, SEPARATOR, MODEL_PATH


def path_generator(airport, file_name):
    return f"{TRAIN_PATH}{airport}{SEPARATOR}{airport}_{file_name}.csv"


def labels_path_generator(airport):
    return f"{TRAIN_PATH}{airport}{SEPARATOR}{FILE_NAME_LABELS}{airport}.csv"


def results_path_generator(airport):
    return f"{TRAIN_PATH}{airport}{SEPARATOR}{airport}_results.csv"


def open_arena_submission_format_path_generator():
    return f"{TRAIN_PATH}submission_format.csv"


def types_path_generator(airport):
    return f"{MODEL_PATH}{airport}{SEPARATOR}types.json"


def model_path_generator(airport):
    return f"{MODEL_PATH}{airport}{SEPARATOR}xgboost-pushback.model"


def departure_model_path_generator(airport):
    return f"{MODEL_PATH}{airport}{SEPARATOR}xgboost-departure.model"
