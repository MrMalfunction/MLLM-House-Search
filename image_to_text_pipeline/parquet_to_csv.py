"""
Simple script to convert Parquet files to CSV with headers.
Usage: python -m image_to_text_pipeline.parquet_to_csv <parquet_file> [--output output.csv]
"""

import argparse
import os

import pandas as pd


def convert_parquet_to_csv(parquet_file, csv_file=None):
    """
    Convert a Parquet file to CSV format with headers.

    Args:
        parquet_file (str): Path to the input Parquet file
        csv_file (str, optional): Path to the output CSV file.
                                  If None, uses the same name with .csv extension

    Returns:
        str: Path to the created CSV file
    """
    # Check if parquet file exists
    if not os.path.exists(parquet_file):
        raise FileNotFoundError(f"Parquet file not found: {parquet_file}")

    # Generate output CSV filename if not provided
    if csv_file is None:
        csv_file = os.path.splitext(parquet_file)[0] + ".csv"

    print(f"Reading Parquet file: {parquet_file}")

    # Read the Parquet file
    df = pd.read_parquet(parquet_file)

    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")

    # Write to CSV with headers
    print(f"Writing to CSV file: {csv_file}")
    df.to_csv(csv_file, index=False, encoding="utf-8")

    print("Successfully converted to CSV!")
    print(f"Output file: {csv_file}")

    return csv_file


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(description="Convert Parquet files to CSV format with headers")
    parser.add_argument("parquet_file", help="Path to the input Parquet file")
    parser.add_argument(
        "--output", "-o", default=None, help="Path to the output CSV file (optional)"
    )

    args = parser.parse_args()

    try:
        csv_path = convert_parquet_to_csv(args.parquet_file, args.output)
        print(f"\n✅ Conversion successful: {csv_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure to:")
        print("1. Provide a valid parquet file path")
        print("2. Have pandas and pyarrow installed: pip install pandas pyarrow")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
