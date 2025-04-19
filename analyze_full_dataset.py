import pandas as pd
import os
import sys

# Path to the data file
DATA_PATH = "Fuente_Json_Consolidado/full_training_data.csv"

def analyze_dataset():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows")
    
    # Basic dataset info
    print("\n--- Basic Dataset Info ---")
    print(f"Number of columns: {len(df.columns)}")
    print("Columns:", df.columns.tolist())
    
    # Analyze makers (brands)
    print("\n--- Maker Analysis ---")
    makers = df["maker"].str.lower()  # Case-insensitive analysis
    unique_makers = makers.nunique()
    print(f"Number of unique makers (case-insensitive): {unique_makers}")
    
    # Count by maker
    maker_counts = makers.value_counts()
    print("\nTop 10 makers by count:")
    for maker, count in maker_counts.head(10).items():
        percentage = count / len(df) * 100
        print(f"  {maker}: {count} rows ({percentage:.1f}%)")
    
    # Analyze SKUs
    print("\n--- SKU Analysis ---")
    sku_col = "referencia"
    unique_skus = df[sku_col].nunique()
    print(f"Total unique SKUs: {unique_skus}")
    
    # Distribution of SKUs
    sku_counts = df[sku_col].value_counts()
    print(f"SKU frequency statistics:")
    print(f"  Min occurrences: {sku_counts.min()}")
    print(f"  Max occurrences: {sku_counts.max()}")
    print(f"  Mean occurrences: {sku_counts.mean():.2f}")
    print(f"  Median occurrences: {sku_counts.median()}")
    
    # Count SKUs with few examples
    skus_with_1 = (sku_counts == 1).sum()
    skus_with_2_5 = ((sku_counts >= 2) & (sku_counts <= 5)).sum()
    skus_with_6_10 = ((sku_counts >= 6) & (sku_counts <= 10)).sum()
    skus_with_more_than_10 = (sku_counts > 10).sum()
    
    print(f"\nSKU occurrence distribution:")
    print(f"  SKUs with 1 example: {skus_with_1} ({skus_with_1/unique_skus*100:.1f}%)")
    print(f"  SKUs with 2-5 examples: {skus_with_2_5} ({skus_with_2_5/unique_skus*100:.1f}%)")
    print(f"  SKUs with 6-10 examples: {skus_with_6_10} ({skus_with_6_10/unique_skus*100:.1f}%)")
    print(f"  SKUs with >10 examples: {skus_with_more_than_10} ({skus_with_more_than_10/unique_skus*100:.1f}%)")
    
    # Show most common SKUs
    print(f"\nTop 10 most common SKUs:")
    for sku, count in sku_counts.head(10).items():
        # Get a sample row for this SKU to show description
        sample_row = df[df[sku_col] == sku].iloc[0]
        description = sample_row.get('descripcion', 'N/A')
        maker = sample_row.get('maker', 'N/A')
        print(f"  {sku}: {count} occurrences - {maker} - {description[:50]}...")
    
    # Analyze SKUs by maker
    print("\n--- SKUs by Maker ---")
    # Group by maker and count unique SKUs
    skus_by_maker = df.groupby(makers)[sku_col].nunique().sort_values(ascending=False)
    print("Number of unique SKUs by maker (top 10):")
    for maker, count in skus_by_maker.head(10).items():
        print(f"  {maker}: {count} unique SKUs")
    
    # Count SKUs with at least N examples by maker
    print("\nNumber of SKUs with at least 10 examples by maker (top 10):")
    skus_with_10_by_maker = {}
    for maker in maker_counts.index[:10]:  # Top 10 makers
        maker_df = df[makers == maker]
        maker_sku_counts = maker_df[sku_col].value_counts()
        skus_with_10 = (maker_sku_counts >= 10).sum()
        skus_with_10_by_maker[maker] = skus_with_10
    
    for maker, count in sorted(skus_with_10_by_maker.items(), key=lambda x: x[1], reverse=True):
        print(f"  {maker}: {count} SKUs with ≥10 examples")

if __name__ == "__main__":
    analyze_dataset()
