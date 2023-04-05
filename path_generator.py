from constants import train_path, file_name_labels, separator, model_path


def path_generator(airport, file_name):
    return f"{train_path}{airport}{separator}{airport}_{file_name}.csv"


def labels_path_generator(airport):
    return f"{train_path}{airport}{separator}{file_name_labels}{airport}.csv"


def types_path_generator(airport):
    return f"{model_path}{airport}{separator}types.json"
