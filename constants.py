import os

separator = os.path.sep

number_of_processors = 10

# airports = [
#     "KATL",
#     "KCLT",
#     "KDEN",
#     "KDFW",
#     "KJFK",
#     "KMEM",
#     "KMIA",
#     "KORD",
#     "KPHX",
#     "KSEA"
# ]

airports = [
    "KCLT"
]

file_name_config = "config"
file_name_etd = "etd"
file_name_first_position = "first_position"
file_name_lamp = "lamp"
file_name_mfs = "mfs"
file_name_runways = "runways"
file_name_standtimes = "standtimes"
file_name_tbfm = "tbfm"
file_name_tfm = "tfm"
file_name_results = "results"
file_name_labels = "train_labels_"

mfs_column_aircraft_type = "aircraft_type"
mfs_column_aircraft_engine_class = "aircraft_engine_class"
mfs_column_major_carrier = "major_carrier"
mfs_column_flight_type = "flight_type"

runways_column_departure_runways = "departure_runways"
runways_column_arrival_runways = "arrival_runways"

model_path = f"data{separator}model{separator}"

# train_path = f"data{separator}train{separator}"
train_path = f"data{separator}small_data{separator}"

flight_id = "gufi"

cloud_category_BK = "BK"
cloud_category_CL = "CL"
cloud_category_FEW = "FEW"
cloud_category_OV = "OV"
cloud_category_SC = "SC"

lightning_prob_N = "N"
lightning_prob_L = "L"
lightning_prob_M = "M"
lightning_prob_H = "H"
