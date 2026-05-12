import pandas as pd

df = pd.read_csv("data/processed/Ship_Operation_example_dataset_unclassified_source_GPS_1.csv")

da = df[['signaldate', 'LAT', 'LON']].copy()

da['Outlier GPS'] = df['is_outlier']
da['At Sea Adrift'] = df['is_drift']

da.to_csv('Ship_Operation_example_dataset_unclassified_source_GPS_1.csv', index=False)