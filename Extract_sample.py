import json

# filepath: Replace with the path to your JSON file
json_file_path = r"C:\Users\juanp\OneDrive\Documents\Python\0_Training\017_Fixacar\001_SKU_desde_descripcion\Fuente_Json_Consolidado\Consolidado.json"

# Load the JSON file
with open(json_file_path, "r") as file:
    data = json.load(file)

# Extract a sample (e.g., first 5 items if it's a list)
if isinstance(data, list):
    sample = data[:5]
elif isinstance(data, dict):
    # Extract first 5 key-value pairs if it's a dictionary
    sample = {k: data[k] for k in list(data.keys())[:5]}
else:
    sample = data  # If it's neither, just return the data as is

# Save the sample to a new file
sample_file_path = r"C:\Users\juanp\OneDrive\Documents\Python\0_Training\017_Fixacar\001_SKU_desde_descripcion\sample.json"
with open(sample_file_path, "w") as sample_file:
    json.dump(sample, sample_file, indent=4)

print(f"Sample saved to {sample_file_path}")
