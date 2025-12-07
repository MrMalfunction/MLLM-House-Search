"""
Simple script to convert Parquet files to CSV with headers.
Usage: Run in Python IDLE or from command line with the parquet file path.
"""

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


# For running in Python IDLE
if __name__ == "__main__":
    # EDIT THIS: Put your parquet file path here
    parquet_file_path = "your_file.parquet"

    # Optional: specify output CSV path (leave as None to auto-generate)
    output_csv_path = None

    try:
        convert_parquet_to_csv(parquet_file_path, output_csv_path)
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to:")
        print("1. Edit 'parquet_file_path' variable with your actual file path")
        print("2. Install pandas with: pip install pandas pyarrow")
