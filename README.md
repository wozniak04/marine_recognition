# Marine Recognition

Automatyczna klasyfikacja stanów operacyjnych statku na podstawie danych GPS/AIS.
Projekt realizowany w ramach programu **Data Waves** pod patronatem Center of Marine Physics.

## Stany operacyjne

| ID | Stan | Opis |
|----|------|------|
| 1 | Port Stay | Statek zacumowany w porcie |
| 2 | Anchor | Statek na kotwicy |
| 3 | Adrift | Dryfowanie (niekontrolowany ruch) |
| 4 | Voyage | Podróż morska |

## Wymagania

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) (package manager)

## Instalacja

```bash
git clone <repo-url>
cd marine_recognition
uv sync --all-extras
```

## Dane

Umieść pliki CSV w odpowiednich katalogach:

```
data/
  raw/
    classified/          # CSV z etykietami (operation_id)
    unclassified/        # CSV bez etykiet
  ports/
    EU_Port_Codes.csv    # Baza portów UN LOCODE
```

Format danych wejściowych (GPS classified):
```csv
signaldate,LAT,LON,operation_id,Outlier GPS,In Port,At Sea Anchor,At Sea Adrift,At Sea Voyage
2024-09-09T08:00:00.000000,44.099955833333,28.658128833333,1,0,1,0,0,0
```

## Użycie

### Preprocessing

```bash
make preprocess CONFIG=configs/base.yaml
```

Przetwarza surowe dane GPS: oblicza kinematykę (SOG, COG, dCOG, ROT, acceleration), wykrywa gapy czasowe, matchuje porty, flaguje outliery.

### Trening modelu

```bash
# Rule-based (baseline)
make train CONFIG=configs/rule_based_gps.yaml

# Random Forest
make train CONFIG=configs/rf_gps.yaml

# XGBoost
make train CONFIG=configs/xgb_gps.yaml

# LightGBM
make train CONFIG=configs/lgbm_gps.yaml

# Hidden Markov Model
make train CONFIG=configs/hmm_gps.yaml

# LSTM
make train CONFIG=configs/lstm_gps.yaml

# Wariant B: GPS+AIS (extended states)
make train CONFIG=configs/extended_ais.yaml
```

> **macOS:** Modele XGBoost/LightGBM wymagają `OMP_NUM_THREADS=1` (ustawione w konfiguracji).

### Porównanie modeli

```bash
make compare
```

Wyświetla tabelę porównawczą accuracy i F1 per klasa dla wszystkich wytrenowanych modeli.

### Inference (pretrenowany model)

Pełny pipeline na nowych danych: raw CSV -> preprocessing -> outlier detection -> port matching -> LSTM predict -> output CSV.

```bash
# Minimalne wywołanie
uv run python -m src.cli.infer \
  --model models/lstm_gps_v1 \
  --input data/raw/unclassified/nowy_statek.csv

# Z własną ścieżką wyjściową i configiem preprocessingu
uv run python -m src.cli.infer \
  --model models/lstm_gps_v1 \
  --input data.csv \
  --output results.csv \
  --config configs/lstm_gps.yaml
```

Wynik zapisywany do `reports/<model_name>/inference_output.csv` (domyślnie) lub do `--output`.

### Testy

```bash
make test       # Testy jednostkowe
make lint       # Linting (ruff)
make format     # Formatowanie (ruff)
make typecheck  # Type checking (pyright)
```

### Czyszczenie

```bash
make clean      # Usuwa przetworzone dane, modele i raporty
```

## Konfiguracja

Wszystkie parametry eksperymentu w plikach YAML (`configs/`). Schemat walidowany Pydantic v2.

Przykład `configs/rf_gps.yaml`:
```yaml
experiment_name: rf_gps_v1
variant: A

paths:
  raw_dir: data/raw/classified
  ports_csv: data/ports/EU_Port_Codes.csv

data:
  input_files:
    - Ship_Operation_example_dataset_classified.csv
    - Ship_Operation_example_dataset_classified_2.csv
  labeled: true

model:
  name: random_forest
  params:
    n_estimators: 300
    max_depth: 20
    class_weight: balanced

training:
  test_size: 0.2
  validation_size: 0.15
  split_strategy: episode
```

## Zaimplementowane modele

| Model | Opis |
|-------|------|
| **Rule-based (FSM)** | Maszyna stanów z konfigurowalnymi progami prędkości i rozrzutu pozycji |
| **Random Forest** | Klasyfikator na cechach okna czasowego (22 cechy) |
| **XGBoost** | Gradient boosting na cechach okna czasowego |
| **LightGBM** | Gradient boosting (Microsoft) na cechach okna czasowego |
| **HMM** | Gaussian Hidden Markov Model — modeluje dynamikę Markowa stanów |
| **LSTM** | Bidirectional LSTM — sieć rekurencyjna na sekwencjach czasowych (3 cechy: SOG, delta_lat, delta_lon) |

## Wyniki (dataset: 30K punktów, 21 epizodów)

### Overall Accuracy

| Model | Train | Val | Test |
|-------|------:|----:|-----:|
| **LSTM** | 93.2% | **94.0%** | **85.4%** |
| Rule-based (FSM) | 83.2% | 89.3% | 98.3%* |
| Random Forest | 99.8% | 94.8% | 50.5% |
| LightGBM | 99.8% | 93.4% | 38.2% |
| XGBoost | 99.8% | 77.3% | 50.0% |
| HMM | 88.3% | 58.8% | 83.0% |

*Test set nie zawiera klasy adrift — rule-based korzysta z tego rozkładu.

### Per-class F1 (LSTM, val set)

| port_stay | anchor | adrift | voyage |
|----------:|-------:|-------:|-------:|
| 0.98 | 0.82 | 0.95 | 0.73 |

### Wnioski

- **LSTM osiąga 94% val accuracy** z zaledwie 3 cechami (`sog_knots`, `delta_lat`, `delta_lon`).
- **Rule-based** ma najlepszy test accuracy (98.3%), ale test set nie ma klasy adrift — łatwiejszy rozkład.
- **Klasyczne ML** (RF, XGB, LightGBM) silnie overfitują: ~100% train vs 38-50% test.
- **HMM** — interpretowalana macierz przejść, 83% test.
- Z pełnym datasetem (więcej epizodów) wyniki ML powinny się dalej poprawić.

## Wariant B: GPS + AIS

Rozszerzenie o dane AIS z transponderów (SOG, COG, HDG bezpośrednio z urządzenia).

### Dodatkowe stany

| Stan | Opis |
|------|------|
| Port Maneuver | Manewr w porcie (niska prędkość, w obrębie portu) |
| Turn | Zwrot (duża zmiana kursu > 30°) |
| Shifting | Przemieszczanie w obrębie portu |

### Porównanie GPS vs AIS

Na wspólnym przedziale czasowym (09-18.09.2024):
- **151 zmatchowanych punktów** (tolerancja 5 min)
- **98.0% zgodności** między klasyfikacją GPS i AIS
- 3 niezgodności: AIS wskazuje "adrift" (SOG 1.4-2.8 kn), GPS wskazuje "voyage" — różnica wynika z metody obliczania prędkości (transponder vs GPS)

### Rozkład stanów AIS (319 punktów)

| Stan | Punktów | % |
|------|--------:|--:|
| Voyage | 278 | 87.1% |
| Adrift | 30 | 9.4% |
| Turn | 10 | 3.1% |
| Anchor | 1 | 0.3% |

## Wyjściowy CSV

Generowany automatycznie po treningu (`reports/<experiment>/output.csv`):

```csv
signaldate,LAT,LON,Outlier GPS,In Port,At Sea Voyage,At Sea Anchor,At Sea Adrift GPS,Confidence Rate,Departure Port Locode,Destination Port Locode
```

## Struktura projektu

```
marine_recognition/
  configs/               # Konfiguracje eksperymentów (YAML)
  data/                  # Dane (gitignored)
  models/                # Zapisane modele (gitignored)
  reports/               # Metryki i wyniki (gitignored)
  src/
    cli/                 # Entry points (preprocess, train, compare, infer)
    data/                # Loading, preprocessing, feature engineering, ports, outliers, splitter
    models/              # Implementacje modeli (rule_engine, classical, hmm, lstm)
    experiments/         # Orkiestracja eksperymentów (BaseExperiment + registry)
    evaluation/          # Metryki (accuracy, confusion matrix, per-class)
    utils/               # Config (Pydantic), geo (haversine), logger, seeding
    visualization/       # Wykresy, mapy
  tests/                 # Testy jednostkowe (pytest)
```
