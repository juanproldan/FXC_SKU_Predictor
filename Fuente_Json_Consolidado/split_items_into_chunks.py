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
    import math
import math

INPUT_FILE = "items_not_empty.json"
NUM_CHUNKS = 10

script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, INPUT_FILE)

with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

chunk_size = math.ceil(len(data) / NUM_CHUNKS)

for i in range(NUM_CHUNKS):
    start = i * chunk_size
    end = start + chunk_size
    chunk = data[start:end]
    chunk_file = os.path.join(script_dir, f"items_chunk_{i+1}.json")
    with open(chunk_file, 'w', encoding='utf-8') as cf:
        json.dump(chunk, cf, ensure_ascii=False, indent=2)
    print(f"Chunk {i+1}: {len(chunk)} entries written to {chunk_file}")
