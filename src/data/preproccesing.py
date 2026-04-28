from ports import check_if_in_port
import numpy as np
from load_data import load_data

print(check_if_in_port(17.9000001,-62.85000001))



def create_windows(data, window_size=10):
    X = []
    Y = data['label'].values[window_size:]  # Etykiety dla każdego okna (klasyfikujemy ostatni punkt)
    # Przesuwamy się po danych wiersz po wierszu
    for i in range(len(data) - window_size):
        # Wycinamy 10 wierszy i tylko potrzebne kolumny (cechy)
        window = data.iloc[i : i + window_size]
        X.append(window)
    return np.array(X),Y

# Użycie:
df=load_data('training_data/Ship_Operation_example_dataset_classified_2.csv')

X_windows, Y_windows = create_windows(df, window_size=10)
print(X_windows[0:2])  # Powinno być około 59990, bo mamy 60000 wierszy i okno 10
print(Y_windows[0:2])  # Powinno być około 59990, bo mamy 60000 wierszy i okno 10
# Wynik X_windows.shape to np. (59990, 10, 3)