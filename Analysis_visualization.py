"""
Analyze prediction labels and their correlation with manipulation status
- Count frequency of each prediction label
- Analyze label distribution by manipulation status (0/1)
- Generate visualizations (bar charts)
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
CSV_PATH = "./agreement_annotations.csv"
LABEL_COLUMN = 'prediction label'
MANIPULATION_COLUMN = 'manipulation'

# Set plot style (professional look)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 6)


def analyze_label_distribution(csv_path, label_col, manipulation_col):
    """
    Perform statistical analysis on prediction labels and their correlation with manipulation

    Args:
        csv_path (str): Path to processed CSV file
        label_col (str): Column name for prediction labels
        manipulation_col (str): Column name for manipulation status
    """
    # Read processed CSV (contains both prediction label and manipulation columns)
    df = pd.read_csv(csv_path)
    print(f"Loaded processed CSV: {csv_path} (Total rows: {len(df)})")

    # Validate required columns exist
    required_cols = [label_col, manipulation_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # ====================
    # 1. Analyze label frequency (most common labels)
    # ====================
    print("\n" + "=" * 50)
    print("1. Most Frequent Prediction Labels")
    print("=" * 50)
    label_counts = df[label_col].value_counts()
    print(label_counts)

    # Plot: Label frequency bar chart
    plt.figure(figsize = (10, 6))
    sns.barplot(x = label_counts.index, y = label_counts.values, palette = 'viridis')
    plt.title('Frequency of Prediction Labels', fontsize = 14, pad = 20)
    plt.xlabel('Prediction Label', fontsize = 12)
    plt.ylabel('Count', fontsize = 12)
    plt.xticks(rotation = 45, ha = 'right')
    plt.tight_layout()
    plt.savefig('label_frequency.png', dpi = 300, bbox_inches = 'tight')
    print("\nSaved label frequency plot: label_frequency.png")

    # ====================
    # 2. Analyze label vs manipulation correlation
    # ====================
    print("\n" + "=" * 50)
    print("2. Label Distribution by Manipulation Status")
    print("=" * 50)

    # Calculate cross-tabulation (percentage by manipulation status)
    cross_tab = pd.crosstab(
        df[label_col],
        df[manipulation_col],
        normalize = 'columns'  # Normalize to percentage of each manipulation group
    ) * 100

    print("Label distribution by manipulation status (percentage):")
    print(cross_tab.round(2))  # Round to 2 decimal places

    # Plot: Label distribution by manipulation status
    plt.figure(figsize = (12, 7))
    cross_tab.plot(kind = 'bar', color = ['#3498db', '#e74c3c'])
    plt.title('Label Distribution by Manipulation Status', fontsize = 14, pad = 20)
    plt.xlabel('Prediction Label', fontsize = 12)
    plt.ylabel('Percentage (%)', fontsize = 12)
    plt.xticks(rotation = 45, ha = 'right')
    plt.legend(
        title = 'Manipulation Status',
        labels = ['Non-Manipulative (0)', 'Manipulative (1)'],
        loc = 'upper right'
    )
    plt.tight_layout()
    plt.savefig('label_vs_manipulation.png', dpi = 300, bbox_inches = 'tight')
    print("\nSaved label vs manipulation plot: label_vs_manipulation.png")

    # ====================
    # 3. Additional key statistics
    # ====================
    print("\n" + "=" * 50)
    print("3. Key Statistics Summary")
    print("=" * 50)

    # Most frequent label
    most_common_label = label_counts.index[0]
    most_common_count = label_counts.iloc[0]
    print(f"- Most frequent label: '{most_common_label}' ({most_common_count} occurrences)")

    # Percentage of manipulative texts for most common label
    most_common_manipulative = df[df[label_col] == most_common_label][manipulation_col].mean() * 100
    print(f"- Percentage of '{most_common_label}' in manipulative texts: {most_common_manipulative:.2f}%")

    # Overall manipulation rate
    overall_manipulation_rate = df[manipulation_col].mean() * 100
    print(f"- Overall manipulative text rate: {overall_manipulation_rate:.2f}%")


if __name__ == '__main__':
    analyze_label_distribution(CSV_PATH, LABEL_COLUMN, MANIPULATION_COLUMN)