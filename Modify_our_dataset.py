"""
Add 'manipulation' column to CSV using majority voting rule
Rule: If ≥3 of the 4 annotation columns are 1 → manipulation=1, else 0
"""
import pandas as pd

# Configuration
CSV_PATH = "./agreement_annotations_with_labels.csv"
TARGET_COLUMN = 'manipulation'  # Column to add
ANNOTATION_COLUMNS = [
    'manipulation_davide',
    'manipulation_diletta',
    'manipulation_inga',
    'manipulation_matias'
]


def add_manipulation_column(csv_path, target_col, annotation_cols):
    """
    Add manipulation column based on majority voting of annotation columns

    Args:
        csv_path (str): Path to input CSV file
        target_col (str): Name of the new column to add
        annotation_cols (list): List of annotation columns for voting
    """
    # Read CSV file
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV file: {csv_path} (Total rows: {len(df)})")

    # Drop unnecessary columns if they exist
    df = df.drop(columns = ['Unnamed: 0', 'agreement', 'index'], errors = 'ignore')

    # Validate annotation columns exist
    missing_cols = [col for col in annotation_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing annotation columns: {missing_cols}")

    # Apply majority voting rule (≥3 votes = 1)
    df[target_col] = (df[annotation_cols].sum(axis = 1) >= 3).astype(int)

    # Save updated CSV (overwrite original file)
    df.to_csv(csv_path, index = False)
    print(f"Successfully added '{target_col}' column!")

    # Print summary statistics
    manipulation_counts = df[target_col].value_counts()
    print(f"\nManipulation column statistics:")
    print(f"- Non-manipulative (0): {manipulation_counts.get(0, 0)} rows")
    print(f"- Manipulative (1): {manipulation_counts.get(1, 0)} rows")


if __name__ == '__main__':
    add_manipulation_column(CSV_PATH, TARGET_COLUMN, ANNOTATION_COLUMNS)