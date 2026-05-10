"""Convert a port dataset to the standard format (LOCODE, LAT, LON as CSV).

Usage:
    python convert_ports.py --input data/ports/raw.csv --output data/ports/converted.csv \
        --lat latitude --lon longitude --locode code
"""
import argparse
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert port file to standard format")
    parser.add_argument("--input", required=True, help="Input port file")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--lat", default="LAT", help="Name of latitude column")
    parser.add_argument("--lon", default="LON", help="Name of longitude column")
    parser.add_argument("--locode", default="LOCODE", help="Name of LOCODE column")
    parser.add_argument("--sep", default=None, help="Separator (auto-detected if not set)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sep = args.sep
    if sep is None:
        with open(args.input) as f:
            header = f.readline()
        if "\t" in header:
            sep = "\t"
        elif ";" in header:
            sep = ";"
        else:
            sep = ","

    df = pd.read_csv(args.input, sep=sep)

    for col, expected in [(args.lat, "LAT"), (args.lon, "LON"), (args.locode, "LOCODE")]:
        if col not in df.columns:
            print(f"ERROR: Column '{col}' not found. Available: {list(df.columns)}")
            return

    out = pd.DataFrame({
        "LOCODE": df[args.locode],
        "LAT": pd.to_numeric(df[args.lat], errors="coerce"),
        "LON": pd.to_numeric(df[args.lon], errors="coerce"),
    })
    out = out.dropna(subset=["LAT", "LON"])
    out = out.drop_duplicates(subset=["LOCODE"])

    out.to_csv(args.output, index=False)
    print(f"Converted {len(out)} ports -> {args.output}")


if __name__ == "__main__":
    main()
