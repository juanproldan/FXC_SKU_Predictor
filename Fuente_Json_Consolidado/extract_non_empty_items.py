import json
import os
import sys
import traceback

def log_uncaught_exceptions(ex_cls, ex, tb):
    with open("crash_log.txt", "a", encoding="utf-8") as f:
        f.write(f"--- Uncaught Exception in {__file__} ---\n")
        traceback.print_exception(ex_cls, ex, tb, file=f)
    sys.__excepthook__(ex_cls, ex, tb)

sys.excepthook = log_uncaught_exceptions

def main():
    with open("pipeline_log.txt", "a", encoding="utf-8") as f:
        f.write(f"Started {__file__}\n")

INPUT_FILE = "Consolidado.json"
OUTPUT_FILE = "items_not_empty.json"

# Get the script directory to ensure correct relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, INPUT_FILE)
output_path = os.path.join(script_dir, OUTPUT_FILE)

with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Only keep quotes where 'items' is a non-empty list
filtered = [entry for entry in data if entry.get('items') and len(entry['items']) > 0]

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(filtered, f, ensure_ascii=False, indent=2)

print(f"Filtered {len(filtered)} entries with non-empty items. Output written to {output_path}")
