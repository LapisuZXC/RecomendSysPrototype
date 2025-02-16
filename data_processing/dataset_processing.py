import os
import pandas as pd


def convert_dat_to_csv(directory):
    if not os.path.exists(directory):
        print(f"directory {directory} is not found")
    for filename in os.listdir(directory):
        if filename.endswith(".dat"):
            filepath = os.path.join(directory, filename)
            csv_filename = filename.replace(".dat", ".csv")
            csv_path = os.path.join(directory, csv_filename)

            df = pd.read_csv(filepath, sep="\t", engine="python")
            df.to_csv(csv_path, index=False, sep=",")


directory = "anime"
convert_dat_to_csv(directory)
