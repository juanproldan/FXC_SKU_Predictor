import os
import json
from glob import glob
from collections import defaultdict
import sys
import traceback

# Global error logging for uncaught exceptions

def log_uncaught_exceptions(ex_cls, ex, tb):
    with open("crash_log.txt", "a", encoding="utf-8") as f:
        traceback.print_exception(ex_cls, ex, tb, file=f)
    sys.__excepthook__(ex_cls, ex, tb)

sys.excepthook = log_uncaught_exceptions

# Directory containing the chunked JSON files
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Fuente_Json_Consolidado')
CHUNK_PATTERN = os.path.join(DATA_DIR, 'items_chunk_*.json')
OUTPUT_PATH = os.path.join(DATA_DIR, 'maker_series_model.json')

def main():
    # Nested dict: maker -> series -> set of models
    hierarchy = defaultdict(lambda: defaultdict(set))

    for file_path in glob(CHUNK_PATTERN):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
            for entry in data:
                maker = entry.get('maker')
                series = entry.get('series')
                model = entry.get('model')
                if maker and series and model:
                    maker_norm = maker.title().strip()
                    series_norm = series.title().strip()
                    hierarchy[maker_norm][series_norm].add(str(model))

    # Convert sets to sorted lists and sort makers and series alphabetically
    out = {
        maker: {series: sorted(list(models)) for series, models in sorted(series_dict.items())}
        for maker, series_dict in sorted(hierarchy.items())
    }

    # Write to JSON
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Output written to {OUTPUT_PATH}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        with open("crash_log.txt", "a", encoding="utf-8") as f:
            f.write(f"Error in main: {e}\n")
            traceback.print_exc(file=f)
        raise
