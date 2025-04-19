import os
import csv
import json
import sys
import traceback

def log_uncaught_exceptions(ex_cls, ex, tb):
    with open("crash_log.txt", "a", encoding="utf-8") as f:
        f.write(f"--- Uncaught Exception in {__file__} ---\n")
        traceback.print_exception(ex_cls, ex, tb, file=f)
    sys.__excepthook__(ex_cls, ex, tb)

sys.excepthook = log_uncaught_exceptions

with open("pipeline_log.txt", "a", encoding="utf-8") as f:
    f.write(f"Started {__file__}\n")

# Path to the hierarchical JSON
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Fuente_Json_Consolidado', 'maker_series_model.json')
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Fuente_Json_Consolidado', 'series_flat.csv')

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

    # Always write header as the first row, use UTF-8 (no BOM), semicolon delimiter for Excel compatibility
    with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['maker', 'series', 'canonical_series'], delimiter=';', quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported flat CSV to {OUTPUT_PATH}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        with open("crash_log.txt", "a", encoding="utf-8") as f:
            f.write(f"Error in {__file__}: {e}\n")
            traceback.print_exc(file=f)
        raise
