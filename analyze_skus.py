import pandas as pd
import os
import sys

# Path to the data file
DATA_PATH = "Fuente_Json_Consolidado/item_training_data.csv"


def analyze_skus():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows")

    # Basic dataset info
    print("\n--- Basic Dataset Info ---")
    print(f"Number of columns: {len(df.columns)}")
    print("Columns:", df.columns.tolist())

    # Focus on SKUs (references)
    sku_col = "referencia"
    if sku_col not in df.columns:
        print(f"Error: '{sku_col}' column not found in the dataset")
        return

    # Count unique SKUs
    unique_skus = df[sku_col].nunique()
    print(f"\n--- SKU Analysis ---")
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
    skus_with_2 = (sku_counts == 2).sum()
    skus_with_3_5 = ((sku_counts >= 3) & (sku_counts <= 5)).sum()
    skus_with_more_than_5 = (sku_counts > 5).sum()

    print(f"\nSKU occurrence distribution:")
    print(
        f"  SKUs with 1 example: {skus_with_1} ({skus_with_1/unique_skus*100:.1f}%)")
    print(
        f"  SKUs with 2 examples: {skus_with_2} ({skus_with_2/unique_skus*100:.1f}%)")
    print(
        f"  SKUs with 3-5 examples: {skus_with_3_5} ({skus_with_3_5/unique_skus*100:.1f}%)")
    print(
        f"  SKUs with >5 examples: {skus_with_more_than_5} ({skus_with_more_than_5/unique_skus*100:.1f}%)")

    # Show most common SKUs
    print(f"\nTop 10 most common SKUs:")
    for sku, count in sku_counts.head(10).items():
        # Get a sample row for this SKU to show description
        sample_row = df[df[sku_col] == sku].iloc[0]
        description = sample_row.get('descripcion', 'N/A')
        maker = sample_row.get('maker', 'N/A')
        print(
            f"  {sku}: {count} occurrences - {maker} - {description[:50]}...")

    # Show some random SKUs with only 1 occurrence
    print(f"\nRandom examples of SKUs with only 1 occurrence:")
    single_skus = sku_counts[sku_counts == 1].index.tolist()
    import random
    random.seed(42)  # For reproducibility
    sample_size = min(5, len(single_skus))
    for sku in random.sample(single_skus, sample_size):
        sample_row = df[df[sku_col] == sku].iloc[0]
        description = sample_row.get('descripcion', 'N/A')
        maker = sample_row.get('maker', 'N/A')
        print(f"  {sku}: {maker} - {description[:50]}...")

    # Print frequency distribution in text form
    print("\nSKU frequency distribution:")
    bins = [1, 2, 3, 5, 10, 20, 50, 100, float('inf')]
    bin_counts = [0] * (len(bins))

    for count in sku_counts.values:
        for i, threshold in enumerate(bins):
            if count <= threshold:
                bin_counts[i] += 1
                break

    for i in range(len(bins)-1):
        if i == len(bins)-2:
            print(f"  > {bins[i]}: {bin_counts[i]} SKUs")
        else:
            print(f"  {bins[i]}-{bins[i+1]}: {bin_counts[i]} SKUs")

    # Check for patterns in SKU codes
    print(f"\n--- SKU Pattern Analysis ---")
    sku_lengths = df[sku_col].astype(str).str.len()
    print(f"SKU length statistics:")
    print(f"  Min length: {sku_lengths.min()}")
    print(f"  Max length: {sku_lengths.max()}")
    print(
        f"  Most common lengths: {sku_lengths.value_counts().head(3).to_dict()}")

    # Check if SKUs follow patterns (e.g., alphanumeric patterns)
    has_letters = df[sku_col].astype(str).str.contains('[A-Za-z]')
    has_numbers = df[sku_col].astype(str).str.contains('[0-9]')
    has_special = df[sku_col].astype(str).str.contains('[^A-Za-z0-9]')

    print(f"SKU character composition:")
    print(
        f"  Contains letters: {has_letters.sum()} ({has_letters.mean()*100:.1f}%)")
    print(
        f"  Contains numbers: {has_numbers.sum()} ({has_numbers.mean()*100:.1f}%)")
    print(
        f"  Contains special chars: {has_special.sum()} ({has_special.mean()*100:.1f}%)")
    print(
        f"  Alphanumeric only: {(has_letters & has_numbers & ~has_special).sum()} ({(has_letters & has_numbers & ~has_special).mean()*100:.1f}%)")

    # Analyze relationship between SKUs and other fields
    print(f"\n--- SKU Relationships ---")
    for col in ['maker', 'series', 'model']:
        if col in df.columns:
            unique_values = df[col].nunique()
            print(f"Unique values in '{col}': {unique_values}")

            # Check how many SKUs per maker/series/model
            skus_per_value = df.groupby(col)[sku_col].nunique()
            print(f"  Min SKUs per {col}: {skus_per_value.min()}")
            print(f"  Max SKUs per {col}: {skus_per_value.max()}")
            print(f"  Avg SKUs per {col}: {skus_per_value.mean():.2f}")

            # Show top values
            print(f"  Top 5 {col} by number of SKUs:")
            for value, count in skus_per_value.sort_values(ascending=False).head(5).items():
                print(f"    {value}: {count} SKUs")


if __name__ == "__main__":
    analyze_skus()
