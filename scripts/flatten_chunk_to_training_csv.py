import json
import csv
import os
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

# Input/output paths
CHUNK_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Fuente_Json_Consolidado', 'items_chunk_1.json')
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Fuente_Json_Consolidado', 'item_training_data.csv')

REQUIRED_ITEM_FIELDS = ["referencia", "descripcion"]
REQUIRED_ENTRY_FIELDS = ["maker", "series", "model"]

def main():
    with open(CHUNK_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rows = []
    for entry in data:
        maker = entry.get("maker", "")
        series = entry.get("series", "")
        model = entry.get("model", entry.get("fabrication_year", ""))
        items = entry.get("items", [])
        for item in items:
            referencia = item.get("referencia", "")
            descripcion = item.get("descripcion", "")
            if all([referencia, descripcion, maker, series, model]):
                rows.append({
                    "maker": maker,
                    "series": series,
                    "model": model,
                    "descripcion": descripcion,
                    "referencia": referencia
                })
    with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["maker", "series", "model", "descripcion", "referencia"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} rows to {OUTPUT_PATH}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        with open("crash_log.txt", "a", encoding="utf-8") as f:
            f.write(f"Error in {__file__}: {e}\n")
            traceback.print_exc(file=f)
        raise
