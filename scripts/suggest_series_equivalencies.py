import os
import json
from collections import defaultdict
from rapidfuzz import fuzz

# Path to the hierarchical JSON
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Fuente_Json_Consolidado', 'maker_series_model.json')
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Fuente_Json_Consolidado', 'suggested_series_equivalencies.json')

# Similarity threshold (tune as needed)
SIM_THRESHOLD = 85

def group_equivalents(series_list, threshold=SIM_THRESHOLD):
    clusters = []
    used = set()
    for i, s1 in enumerate(series_list):
        if s1 in used:
            continue
        cluster = [s1]
        used.add(s1)
        for j, s2 in enumerate(series_list):
            if i != j and s2 not in used:
                if fuzz.token_sort_ratio(s1, s2) >= threshold:
                    cluster.append(s2)
                    used.add(s2)
        clusters.append(cluster)
    return clusters

def main():
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result = {}
    for maker, series_dict in data.items():
        series_list = list(series_dict.keys())
        clusters = group_equivalents(series_list)
        # Use the first in each cluster as the canonical
        result[maker] = {cluster[0]: cluster for cluster in clusters}

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Suggested equivalencies written to {OUTPUT_PATH}")

if __name__ == '__main__':
    main()
