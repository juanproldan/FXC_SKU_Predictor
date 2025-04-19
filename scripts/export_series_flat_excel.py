import os
import json
import pandas as pd

# Path to the hierarchical JSON
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Fuente_Json_Consolidado', 'maker_series_model.json')
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Fuente_Json_Consolidado', 'series_flat.xlsx')

def main():
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rows = []
    for maker, series_dict in data.items():
        for series in series_dict.keys():
            rows.append({
                'maker': maker,
                'series': series,
                'canonical_series': series  # default to series; user can edit
            })

    df = pd.DataFrame(rows)
    df.to_excel(OUTPUT_PATH, index=False)
    print(f"Exported flat Excel to {OUTPUT_PATH}")

if __name__ == '__main__':
    main()
