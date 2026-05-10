# Marine Recognition - Klasyfikacja manewrów statku

System klasyfikacji stanów operacyjnych statku na podstawie danych GPS.

## Stany operacyjne

| Stan | Opis |
|------|------|
| `port_stay` | Statek w porcie |
| `anchor` | Statek na kotwicy |
| `adrift` | Statek dryfujący |
| `voyage` | Statek w podróży |

## Modele

| Model | Config | Opis |
|-------|--------|------|
| `rule_based` | `configs/rule_based.yaml` | Reguly if/else, konfigurowalne w YAML |
| `decision_tree` | `configs/decision_tree.yaml` | Drzewo decyzyjne (sklearn), uczy sie z danych |

## Struktura projektu

```
marine_recognition_final/
├── configs/
│   ├── rule_based.yaml               # Konfiguracja z regulami klasyfikacji
│   └── decision_tree.yaml            # Konfiguracja drzewa decyzyjnego
├── data/
│   ├── raw/                           # Surowe dane GPS (CSV)
│   ├── ports/                         # Bazy portów (wszystkie CSV z folderu)
│   └── preprocessed/                  # Dane po preprocessingu
├── models/                            # Zapisane modele
├── reports/                           # Wyniki, metryki, wykresy
├── src/
│   ├── config.py                      # Loader konfiguracji YAML
│   ├── data/
│   │   ├── preprocessing.py           # Kinematyka, outliery, cechy rolling
│   │   ├── ports.py                   # Dopasowanie portów z bazy
│   │   ├── splitter.py                # Podział train/val/test
│   │   └── oversampling.py            # Oversampling klas mniejszościowych
│   ├── models/
│   │   ├── rule_based.py              # Klasyfikator rule-based
│   │   └── decision_tree.py           # Klasyfikator decision tree
│   └── evaluation/
│       ├── metrics.py                 # Metryki ewaluacji
│       └── plots.py                   # Wykresy confusion matrix
├── preprocess.py                      # CLI: preprocessing
├── train.py                           # CLI: trening + wykresy
├── classify.py                        # CLI: klasyfikacja nowych danych
├── Makefile
└── pyproject.toml
```

## Instalacja

```bash
uv sync
```

## Workflow

### 1. Preprocessing

```bash
make preprocess
```

### 2. Trening (wymaga danych z etykietami)

```bash
make train                                       # rule-based (domyslny)
make train CONFIG=configs/decision_tree.yaml     # decision tree
```

Generuje metryki + wykresy confusion matrix do `reports/`.

### 3. Klasyfikacja nowych danych (surowe CSV, bez etykiet)

```bash
make classify INPUT=data/raw/nowe_dane.csv
make classify INPUT=data/raw/nowe_dane.csv OUTPUT=results/     # output do katalogu
make classify CONFIG=configs/decision_tree.yaml INPUT=data/raw/nowe_dane.csv
```

Robi pelny pipeline: preprocessing → outliery → porty → klasyfikacja → output CSV.
`OUTPUT` moze byc sciezka do pliku CSV lub katalogu (wtedy tworzy `output.csv` w srodku).

Wszystkie komendy domyslnie uzywaja `configs/rule_based.yaml`. Zeby uzyc innego configa:

```bash
make train CONFIG=configs/moj_config.yaml
```

## Konfiguracja regul klasyfikacji

Reguly sa zdefiniowane w YAML jako lista. Kazda regula ma warunki na cechy i wynikowy stan. Sprawdzane po kolei — pierwsza pasujaca wygrywa.

```yaml
model:
  rules:
    - state: voyage
      confidence: 85
      conditions:
        rolling_sog_kn: {min: 3.0}

    - state: adrift
      confidence: 70
      conditions:
        rolling_sog_kn: {min: 0.3, max: 3.0}
        rolling_spread_m: {min: 200.0}
        rolling_net_displacement_m:
          min_ratio: {of: rolling_spread_m, value: 0.3}

    - state: anchor
      confidence: 70
      conditions:
        rolling_sog_kn: {max: 0.5}
        rolling_spread_m: {max: 200.0}
```

Reguly klasyfikuja tylko stany morskie (`voyage`, `anchor`, `adrift`). `port_stay` jest przypisywany automatycznie na podstawie bazy portow.

Zeby dodac nowa regule: dopisz wpis do listy `rules` w configu. Zeby dodac nowa ceche: dodaj ja w `preprocessing.py`, a potem uzyj w warunkach reguly.

### Dostepne warunki

- `min` — wartosc minimalna
- `max` — wartosc maksymalna
- `min_ratio` — wartosc minimalna jako procent innej cechy: `{of: kolumna, value: 0.3}`

## Dostepne cechy

Z preprocessingu dostepne sa nastepujace cechy (mozna uzyc w regulach):

| Cecha | Opis |
|-------|------|
| `rolling_sog_kn` | Srednia predkosc w oknie rolling |
| `rolling_sog_std` | Odchylenie std predkosci — stabilnosc ruchu |
| `rolling_spread_m` | Maks. odleglosc od srodka ciezkosci okna — czy statek stoi w miejscu |
| `rolling_net_displacement_m` | Odleglosc pierwszy-ostatni punkt — kierunkowosc ruchu |
| `rolling_accel_abs` | Srednie |przyspieszenie| — aktywnosc manewrowa |
| `rolling_rot_abs` | Sredni |rate of turn| — intensywnosc skretow |
| `rolling_cog_std` | Circular std kursu — stabilnosc kierunku |
| `sog_knots` | Chwilowa predkosc (per-punkt) |
| `acceleration` | Chwilowe przyspieszenie (per-punkt) |
| `rot` | Chwilowy rate of turn (per-punkt) |
| `cog` | Chwilowy kurs (per-punkt) |

## Format wyjsciowy

| Kolumna | Opis |
|---------|------|
| `signaldate` | Czas sygnalu GPS |
| `LAT`, `LON` | Wspolrzedne |
| `Outlier GPS` | 1 = punkt uznany za outlier |
| `In Port` | 1 = statek w porcie |
| `At Sea Voyage` | 1 = podroz |
| `At Sea Anchor` | 1 = kotwica |
| `At Sea Adrift GPS` | 1 = dryf |
| `Confidence Rate` | Pewnosc klasyfikacji (0-100) |
| `Departure Port Locode` | LOCODE portu wyjscia |
| `Destination Port Locode` | LOCODE portu docelowego |

## Porty

System laduje **wszystkie pliki CSV** z folderu `data/ports/`. Kazdy plik musi miec kolumny `LAT`, `LON`, `LOCODE`. Zeby dodac nowe porty, wystarczy wrzucic kolejny CSV do folderu.

Klasyfikacja portow odbywa sie wylacznie na podstawie bazy portow (nie przez model). Punkty w promieniu portu sa oznaczane jako `port_stay` z confidence 100%.

### Konwersja plikow portow

Jesli nowy plik z portami ma inne nazwy kolumn lub separator, uzyj:

```bash
# Domyslnie (kolumny LAT, LON, LOCODE, separator auto-wykrywany):
make convert-ports INPUT=data/ports/nowy.csv OUTPUT=data/ports/nowy-converted.csv

# Z mapowaniem kolumn:
make convert-ports INPUT=data/ports/nowy.csv OUTPUT=data/ports/nowy-converted.csv \
    LAT=latitude LON=longitude LOCODE=code

# Z wymuszonym separatorem:
make convert-ports INPUT=data/ports/nowy.csv OUTPUT=data/ports/nowy-converted.csv SEP=";"
```

Po konwersji usun oryginalny plik (jesli ma inny format niz CSV z przecinkami), zeby port matcher go nie ladowal.
