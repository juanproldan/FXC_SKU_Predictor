import pandas as pd

# Load data
print("Loading data...")
df = pd.read_csv("Fuente_Json_Consolidado/item_training_data.csv")
print(f"Loaded {len(df)} rows")

# Filter to Renault (case-insensitive)
df_renault = df[df["maker"].str.lower() == "renault"]
print(f"Filtered to {len(df_renault)} Renault rows")

# Count SKUs
sku_counts = df_renault["referencia"].value_counts()
print(f"Total unique Renault SKUs: {len(sku_counts)}")

# Print top SKUs
print("\nTop 10 Renault SKUs:")
for sku, count in sku_counts.head(10).items():
    print(f"  {sku}: {count} occurrences")
    examples = df_renault[df_renault["referencia"]
                          == sku]["descripcion"].values[:2]
    print(f"    Examples: {examples}")

# Print distribution
print("\nSKU frequency distribution:")
print(
    f"  SKUs with 1 example: {(sku_counts == 1).sum()} ({(sku_counts == 1).sum()/len(sku_counts)*100:.1f}%)")
print(
    f"  SKUs with 2-5 examples: {((sku_counts >= 2) & (sku_counts <= 5)).sum()} ({((sku_counts >= 2) & (sku_counts <= 5)).sum()/len(sku_counts)*100:.1f}%)")
print(
    f"  SKUs with 6-10 examples: {((sku_counts >= 6) & (sku_counts <= 10)).sum()} ({((sku_counts >= 6) & (sku_counts <= 10)).sum()/len(sku_counts)*100:.1f}%)")
print(
    f"  SKUs with >10 examples: {(sku_counts > 10).sum()} ({(sku_counts > 10).sum()/len(sku_counts)*100:.1f}%)")
