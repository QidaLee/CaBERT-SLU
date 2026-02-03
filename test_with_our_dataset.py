"""
Use trained model to predict intent for each row in agreement_annotations.csv
Add prediction label directly to original CSV (Fixed All Errors)
"""
import os
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, BertConfig

# Import your project's core modules
from utils import load_data
from model import BertContextNLU
from config import opt

def load_dictionaries():
    """Load intent and slot dictionaries (ID ↔ Name)"""
    # Load intent dictionary (for intent prediction)
    with open(opt.dic_path_with_tokens, 'rb') as f:
        intent_dic = pickle.load(f)
    intent_id2name = {v[0]: k for k, v in intent_dic.items()}  # ID → Name

    # Load slot dictionary (get exact slot count from trained model)
    with open(opt.slot_path, 'rb') as f:
        slot_dic = pickle.load(f)
    slot_id2name = {v: k for k, v in slot_dic.items()}  # ID → Name

    return intent_dic, intent_id2name, slot_dic, slot_id2name

def load_and_preprocess_csv(csv_path, text_col="text"):
    """
    Load CSV file and preprocess text (match training data format)
    Keep original index to align predictions correctly
    """
    # Load CSV with original index preserved
    df = pd.read_csv(csv_path)
    print("Loaded original CSV: {} (total {} rows)".format(csv_path, len(df)))

    # Check if text column exists
    if text_col not in df.columns:
        raise ValueError("Text column '{}' not found in CSV. Available columns: {}".format(text_col, df.columns.tolist()))

    # Create copy to avoid modifying original data during preprocessing
    df_working = df.copy()

    # Preprocess text (same as training)
    df_working["processed_text"] = df_working[text_col].fillna("").str.strip().str.replace(r"\s+", " ", regex=True)

    # Create mask for valid text (non-empty)
    valid_mask = df_working["processed_text"].str.len() > 0
    valid_texts = df_working.loc[valid_mask, "processed_text"].tolist()
    valid_indices = df_working.loc[valid_mask].index.tolist()

    print("Found {} valid rows (non-empty text) out of {} total rows".format(len(valid_texts), len(df)))

    return df, df_working, valid_texts, valid_indices

def text_to_3d_model_input(texts, tokenizer, maxlen=60):
    """
    Convert raw text to 3D model input (batch, dialog_len=1, utterance_len)
    Your model expects 3D input (dialog level) - even for single utterances
    """
    # Tokenize text (BERT-base-uncased)
    tokenized = tokenizer(
        texts,
        max_length=maxlen,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        add_special_tokens=True
    )

    # Get 2D input (batch, seq_len)
    input_ids_2d = tokenized["input_ids"]  # Shape: [num_texts, maxlen]
    attention_masks_2d = tokenized["attention_mask"]  # Shape: [num_texts, maxlen]

    # Convert to 3D (add dialog length dimension = 1 for single utterance)
    input_ids_3d = input_ids_2d.unsqueeze(1)  # Shape: [num_texts, 1, maxlen]
    attention_masks_3d = attention_masks_2d.unsqueeze(1)  # Shape: [num_texts, 1, maxlen]

    # Create lengths (dialog length = 1 for single utterance)
    lengths = torch.ones(len(texts)).long()  # Shape: [num_texts]

    return {
        "input_ids": input_ids_3d,
        "attention_masks": attention_masks_3d,
        "lengths": lengths
    }

def predict_intents_batch(model, all_input_data, intent_id2name, device, batch_size=32):
    """
    Predict intent labels in small batches to avoid CUDA OOM
    Handles both 2D and 3D intent outputs (fixes IndexError)
    """
    model.eval()
    all_predictions = []

    # Get full data
    input_ids = all_input_data["input_ids"]
    attention_masks = all_input_data["attention_masks"]
    lengths = all_input_data["lengths"]
    total_samples = len(input_ids)

    # Generate intent tokens (required by model) - once for all batches
    intent_tokens = [intent for name, (tag, intent) in intent_dic.items()]
    intent_tok, mask_tok = load_data(intent_tokens, 10)
    intent_tokens_tensor = torch.zeros(len(intent_tok), 10).long().to(device)
    mask_tokens_tensor = torch.zeros(len(mask_tok), 10).long().to(device)
    for i in range(len(intent_tok)):
        intent_tokens_tensor[i] = torch.tensor(intent_tok[i])
    for i in range(len(mask_tok)):
        mask_tokens_tensor[i] = torch.tensor(mask_tok[i])

    # Process in batches
    with torch.no_grad():
        for start_idx in tqdm(range(0, total_samples, batch_size), desc="Processing batches"):
            # Get batch data
            end_idx = min(start_idx + batch_size, total_samples)
            batch_input_ids = input_ids[start_idx:end_idx].to(device)
            batch_attention_masks = attention_masks[start_idx:end_idx].to(device)
            batch_lengths = lengths[start_idx:end_idx].to(device)

            # Create batch dummy labels (3D compatible)
            batch_size_current = batch_input_ids.shape[0]
            dialog_len = batch_input_ids.shape[1]
            utterance_len = batch_input_ids.shape[2]
            num_intents = len(intent_id2name)
            num_slots = len(slot_id2name)

            # Dummy intent labels (batch, dialog_len, num_intents)
            dummy_intent_labels = torch.zeros((batch_size_current, dialog_len, num_intents)).long().to(device)
            # Dummy slot labels (batch, dialog_len, utterance_len)
            dummy_slot_labels = torch.zeros((batch_size_current, dialog_len, utterance_len)).long().to(device)

            # Forward pass (batch)
            intent_outputs, _, _ = model(
                batch_input_ids, batch_attention_masks, batch_attention_masks,
                batch_lengths, dummy_slot_labels, dummy_intent_labels,
                intent_tokens_tensor, mask_tokens_tensor
            )

            # Parse batch predictions (handle both 2D and 3D outputs)
            intent_probs = torch.sigmoid(intent_outputs)

            # Check output dimensions and adjust parsing
            if intent_probs.dim() == 3:
                # 3D output (batch, dialog_len, num_intents) - expected
                for idx in range(batch_size_current):
                    dialog_probs = intent_probs[idx, 0, :]  # Get first (only) dialog turn
                    pred_ids = (dialog_probs > 0.5).nonzero().squeeze().cpu().numpy()

                    if pred_ids.size == 0:
                        pred_label = "no_intent"
                    elif pred_ids.ndim == 0:
                        pred_label = intent_id2name.get(int(pred_ids), "unknown")
                    else:
                        pred_labels = [intent_id2name.get(int(pid), "unknown") for pid in pred_ids]
                        pred_label = ", ".join(pred_labels)

                    all_predictions.append(pred_label)
            elif intent_probs.dim() == 2:
                # 2D output (batch, num_intents) - model flattened dialog dimension
                for idx in range(batch_size_current):
                    dialog_probs = intent_probs[idx, :]  # Direct access (no dialog dimension)
                    pred_ids = (dialog_probs > 0.5).nonzero().squeeze().cpu().numpy()

                    if pred_ids.size == 0:
                        pred_label = "no_intent"
                    elif pred_ids.ndim == 0:
                        pred_label = intent_id2name.get(int(pred_ids), "unknown")
                    else:
                        pred_labels = [intent_id2name.get(int(pid), "unknown") for pid in pred_ids]
                        pred_label = ", ".join(pred_labels)

                    all_predictions.append(pred_label)
            else:
                # Unexpected dimension - raise informative error
                raise ValueError("Unexpected intent output dimension: {}".format(intent_probs.dim()))

            # Clear GPU memory after batch
            del batch_input_ids, batch_attention_masks, batch_lengths, intent_outputs, intent_probs
            torch.cuda.empty_cache()

    return all_predictions

def add_predictions_to_original_csv(df, valid_indices, predictions, csv_path):
    """
    Add prediction label column to original CSV
    Fill empty text rows with "empty_text"
    """
    # Initialize prediction column with default value
    df["prediction label"] = "empty_text"

    # Assign predictions to valid rows (match original indices)
    for idx, pred in zip(valid_indices, predictions):
        df.loc[idx, "prediction label"] = pred

    # Save back to original CSV file
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print("Successfully added 'prediction label' column to original CSV: {}".format(csv_path))

    # Print summary statistics
    pred_counts = df["prediction label"].value_counts()
    print("\nPrediction Summary:")
    for label, count in pred_counts.items():
        print("  {}: {} rows".format(label, count))

    # Show sample of predictions
    print("\nSample Predictions (First 10 Rows):")
    sample_df = df[["text", "prediction label"]].head(10)
    for idx, row in sample_df.iterrows():
        # Truncate long text for readability
        text_sample = row["text"][:50] + "..." if len(row["text"]) > 50 else row["text"]
        print("Row {}: Text='{}' → Prediction='{}'".format(idx+1, text_sample, row["prediction label"]))

def main(csv_path="./agreement_annotations.csv", batch_size=16):
    """Main function: Predict and add label to original CSV (batch processing)"""
    # 1. Initialize device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using device: {}".format(device))

    # 2. Load dictionaries (intent + slot for correct model initialization)
    global intent_dic, intent_id2name, slot_dic, slot_id2name
    intent_dic, intent_id2name, slot_dic, slot_id2name = load_dictionaries()
    print("Loaded intent dictionary: {} intent labels".format(len(intent_id2name)))
    print("Loaded slot dictionary: {} slot labels".format(len(slot_id2name)))

    # 3. Load and preprocess CSV (keep original data)
    df_original, df_working, valid_texts, valid_indices = load_and_preprocess_csv(csv_path)

    if len(valid_texts) == 0:
        print("No valid text rows found in CSV. Exiting.")
        return

    # 4. Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    print("Initialized BERT tokenizer")

    # 5. Convert text to 3D model input (on CPU to save GPU memory)
    input_data = text_to_3d_model_input(valid_texts, tokenizer, opt.maxlen)
    print("Converted {} texts to 3D model input (shape: {})".format(len(valid_texts), input_data['input_ids'].shape))

    # 6. Load trained model (use exact intent/slot counts from dictionaries)
    config = BertConfig(
        vocab_size_or_config_json_file=32000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072
    )

    # Use actual intent/slot counts (matches trained model)
    model = BertContextNLU(config, opt, len(intent_dic), len(slot_dic))

    # Load model weights (now dimensions match)
    if not os.path.exists(opt.model_path):
        raise ValueError("Model file not found: {}".format(opt.model_path))

    model.load_state_dict(torch.load(opt.model_path, map_location=device))
    model = model.to(device)
    print("Loaded trained model: {}".format(opt.model_path))

    # 7. Run prediction in smaller batches (avoid CUDA OOM + IndexError)
    print("\nStarting prediction with batch size {}...".format(batch_size))
    predictions = predict_intents_batch(model, input_data, intent_id2name, device, batch_size)

    # 8. Add predictions to original CSV
    add_predictions_to_original_csv(df_original, valid_indices, predictions, csv_path)

    print("\nAll done! Your original CSV now has a 'prediction label' column.")

if __name__ == '__main__':
    # Configure paths (adjust these to match your project)
    opt.model_path = "./checkpoints/best_e2e_multi.pth"  # Your trained model path
    opt.dic_path_with_tokens = "./data/e2e_dialogue/intent2id_multi_with_tokens.pkl"  # Intent dictionary
    opt.slot_path = "./data/e2e_dialogue/slot2id.pkl"  # Slot dictionary (matches trained model)
    opt.maxlen = 60  # Must match training max sequence length

    # Run prediction with small batch size (16) to avoid OOM and ensure stability
    main(
        csv_path="./agreement_annotations.csv",  # Path to YOUR CSV file
        batch_size=16  # Reduced batch size for maximum stability
    )