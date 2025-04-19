import os
import json
import csv
from collections import defaultdict

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Fuente_Json_Consolidado', 'series_flat.csv')
JSON_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Fuente_Json_Consolidado', 'maker_series_model.json')
OUTPUT_PATH = JSON_PATH  # Overwrite the main JSON

# Read (maker, series) -> canonical_series mapping from CSV
def load_equivalencies(csv_path):
    equiv = {}
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            key = (row['maker'].strip(), row['series'].strip())
            equiv[key] = row['canonical_series'].strip()
    return equiv

def main():
    equiv = load_equivalencies(CSV_PATH)
    with open(JSON_PATH, encoding='utf-8') as f:
        data = json.load(f)

    # Build new structure with canonical series
    new_data = defaultdict(lambda: defaultdict(set))
    for maker, series_dict in data.items():
        for series, models in series_dict.items():
            canonical = equiv.get((maker, series), series)
            for model in models:
                new_data[maker][canonical].add(model)

    # Convert sets to sorted lists and sort
    out = {
        maker: {series: sorted(list(models)) for series, models in sorted(series_dict.items())}
        for maker, series_dict in sorted(new_data.items())
    }
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Updated maker_series_model.json with canonical series mapping.")

if __name__ == '__main__':
    main()
