import pandas as pd
import numpy as np
from ports import check_if_in_port
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['signaldate'] = pd.to_datetime(data['signaldate'])
    data = data.sort_values('signaldate').reset_index(drop=True)

    return data


#print(load_data('training_data/Ship_Operation_example_dataset_classified_2.csv').head())