"""
Only show all possible labels the model can output (no training/testing involved)
This script only reads the intent dictionary file, no model loading or data processing
"""
import pickle
import os

# Configure the path to intent dictionary (update this path to match your project)
INTENT_DICT_PATH = "./data/e2e_dialogue/intent2id_multi_with_tokens.pkl"


def load_intent_labels(intent_dict_path):
    """
    Load intent dictionary and return all possible output labels

    Args:
        intent_dict_path (str): Path to the intent dictionary pickle file

    Returns:
        list: All intent labels the model can output
    """
    # Check if the dictionary file exists
    if not os.path.exists(intent_dict_path):
        raise FileNotFoundError(f"Intent dictionary file not found: {intent_dict_path}")

    # Load the pickle dictionary file
    with open(intent_dict_path, 'rb') as f:
        intent_dic = pickle.load(f)

    # Extract all intent labels (keys of the dictionary are the output labels)
    intent_labels = list(intent_dic.keys())

    return intent_labels


def main():
    print("=" * 50)
    print("All possible labels the model can output:")
    print("=" * 50)

    # Load and print all labels
    try:
        # Get all intent labels from dictionary
        labels = load_intent_labels(INTENT_DICT_PATH)

        # Sort labels alphabetically for better readability (optional)
        labels.sort()

        # Print each label with index
        for idx, label in enumerate(labels, 1):
            print(f"{idx:2d}. {label}")

        # Print summary statistics
        print("=" * 50)
        print(f"Total number of labels: {len(labels)}")
        print("=" * 50)

    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nTroubleshooting checklist:")
        print(f"1. Is the intent dictionary path correct? Current path: {INTENT_DICT_PATH}")
        print("2. Is the file a valid pickle (.pkl) file?")


if __name__ == '__main__':
    main()