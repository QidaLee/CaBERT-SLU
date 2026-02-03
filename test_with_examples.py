"""
Independent test script: Load trained model and print specific prediction examples
(text + real labels + predicted labels)
"""
import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertConfig

# Import your project modules
from utils import f1_score_intents, evaluate_iob, prf, load_data
from model import BertContextNLU
from all_data_context import get_dataloader_context
from config import opt

def load_dictionaries():
    """Load intent/slot dictionaries and build reverse mapping (ID to name)"""
    with open(opt.dic_path_with_tokens, 'rb') as f:
        intent_dic = pickle.load(f)
    intent_id2name = {v[0]: k for k, v in intent_dic.items()}

    with open(opt.slot_path, 'rb') as f:
        slot_dic = pickle.load(f)
    slot_id2name = {v: k for k, v in slot_dic.items()}

    return intent_id2name, slot_id2name

def load_test_data():
    """Load and split test data (same logic as original code)"""
    with open(opt.train_path, 'rb') as f:
        train_data = pickle.load(f)

    np.random.seed(0)
    indices = np.random.permutation(len(train_data))
    test_data = np.array(train_data, dtype=object)[indices[int(len(train_data)*0.7):]][:100]
    return test_data

def init_model(intent_num, slot_num):
    """Initialize model and load pretrained weights"""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    config = BertConfig(
        vocab_size_or_config_json_file=32000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072
    )

    model = BertContextNLU(config, opt, intent_num, slot_num)

    if opt.model_path and os.path.exists(opt.model_path):
        state_dict = torch.load(opt.model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Successfully loaded pretrained model: {}".format(opt.model_path))
    else:
        raise ValueError("Model file does not exist: {}".format(opt.model_path))

    model = model.to(device)
    model.eval()
    return model, device

def get_intent_tokens(intent_dic, device):
    """Generate intent tokens (required input for original model)"""
    intent_tokens = [intent for name, (tag, intent) in intent_dic.items()]
    intent_tok, mask_tok = load_data(intent_tokens, 10)

    intent_tokens_tensor = torch.zeros(len(intent_tok), 10).long().to(device)
    mask_tokens_tensor = torch.zeros(len(mask_tok), 10).long().to(device)
    for i in range(len(intent_tok)):
        intent_tokens_tensor[i] = torch.tensor(intent_tok[i])
    for i in range(len(mask_tok)):
        mask_tokens_tensor[i] = torch.tensor(mask_tok[i])

    return intent_tokens_tensor, mask_tokens_tensor

def print_prediction_examples(model, test_loader, intent_id2name, slot_id2name, device, intent_tokens, mask_tokens):
    """Core function: Stable version with correct text extraction (2D result_ids)"""
    # Use the same tokenizer as training
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    all_examples = []

    # Debug: Print first batch dimension info (only once)
    first_batch = True

    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
        result_ids, result_token_masks, result_masks, lengths, result_slot_labels, result_labels = batch

        # Move to device
        result_ids = result_ids.to(device)
        lengths = lengths.to(device)
        result_labels = result_labels.to(device)

        # Debug info for first batch
        if first_batch:
            print(f"\n=== Debug Info ===")
            print(f"result_ids shape (2D): {result_ids.shape}")  # Should show [8, xxx] (2D)
            print(f"lengths values: {lengths.cpu().numpy()[:5]}")
            first_batch = False

        # Get actual batch size (from 2D tensor)
        actual_batch_size = result_ids.shape[0]

        with torch.no_grad():
            outputs, labels, predicted_slot_outputs = model(
                result_ids, result_token_masks.to(device), result_masks.to(device), lengths,
                result_slot_labels.to(device), result_labels, intent_tokens, mask_tokens
            )

        # Process each sample (2D tensor compatible)
        for i in range(actual_batch_size):
            try:
                # 1. Safe length handling (2D compatible)
                if i >= lengths.shape[0]:
                    continue
                sample_len = lengths[i].item() if lengths[i].item() > 0 else opt.maxlen

                # 2. Correct token extraction for 2D tensor [batch_size, seq_len]
                # Directly take the i-th sample (no third dimension!)
                text_ids = result_ids[i, :sample_len].cpu().numpy()

                # 3. Filter valid tokens (match your load_data function)
                valid_ids = text_ids[text_ids != 0]  # Remove padding (0)

                # 4. Decode text (guaranteed non-empty)
                if len(valid_ids) > 0:
                    text = tokenizer.decode(valid_ids, skip_special_tokens=True).strip()
                    # Fallback if decode returns empty
                    text = text if text else f"Valid_IDs: {valid_ids[:5]}"
                else:
                    text = f"No_valid_tokens (len={sample_len})"

                # 5. Parse intent labels (original working logic)
                real_intent_ids = torch.where(labels[i] == 1)[0].cpu().numpy()
                real_intents = [intent_id2name.get(id, f"Unknown_ID_{id}") for id in real_intent_ids]

                pred_intent_logits = torch.sigmoid(outputs[i])
                pred_intent_ids = torch.where(pred_intent_logits > 0.5)[0].cpu().numpy()
                pred_intents = [intent_id2name.get(id, f"Unknown_ID_{id}") for id in pred_intent_ids]

                is_correct = set(real_intents) == set(pred_intents)

                # 6. Save results
                all_examples.append({
                    "text": text,
                    "real_intent_labels": real_intents,
                    "predicted_intent_labels": pred_intents,
                    "is_intent_correct": is_correct
                })

            except Exception as e:
                # Minimal error handling (only print, don't skip all samples)
                print(f"\nWarning: Sample {i} in batch {batch_idx} - {str(e)[:80]}")
                all_examples.append({
                    "text": f"Error: {str(e)[:50]}",
                    "real_intent_labels": ["Unknown"],
                    "predicted_intent_labels": ["Unknown"],
                    "is_intent_correct": False
                })
                continue

    # Print results (original format)
    print("\n" + "="*100)
    print("Test sample prediction examples (first 10):")
    print("="*100)
    for idx, example in enumerate(all_examples[:10]):
        print(f"\n[Sample {idx+1}]")
        print(f"Text: {example['text']}")
        print(f"Real intent labels: {example['real_intent_labels']}")
        print(f"Predicted intent labels: {example['predicted_intent_labels']}")
        print(f"Is prediction correct: {example['is_intent_correct']}")

    # Print error examples
    error_examples = [e for e in all_examples if not e["is_intent_correct"]]
    if error_examples:
        print("\n" + "="*100)
        print(f"Incorrect prediction examples (first 5, total {len(error_examples)}):")
        print("="*100)
        for idx, example in enumerate(error_examples[:5]):
            print(f"\n[Error Sample {idx+1}]")
            print(f"Text: {example['text']}")
            print(f"Real intent labels: {example['real_intent_labels']}")
            print(f"Predicted intent labels: {example['predicted_intent_labels']}")

    # Overall statistics
    total = len(all_examples)
    correct = sum([1 for e in all_examples if e["is_intent_correct"]])
    accuracy = correct / total if total > 0 else 0
    print("\n" + "="*100)
    print("Overall statistics:")
    print(f"Total valid test samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Intent prediction accuracy: {accuracy:.4f}")
    print("="*100)

def main(**kwargs):
    # Override config parameters
    for k, v in kwargs.items():
        setattr(opt, k, v)

    # Load dictionaries
    intent_id2name, slot_id2name = load_dictionaries()
    with open(opt.dic_path_with_tokens, 'rb') as f:
        intent_dic = pickle.load(f)

    # Load test data
    test_data = load_test_data()

    # Create data loader
    test_loader = get_dataloader_context(test_data, intent_dic, slot_id2name, opt)

    # Initialize model
    model, device = init_model(len(intent_dic), len(slot_id2name))

    # Generate intent tokens
    intent_tokens, mask_tokens = get_intent_tokens(intent_dic, device)

    # Run test and print examples
    print_prediction_examples(model, test_loader, intent_id2name, slot_id2name, device, intent_tokens, mask_tokens)

if __name__ == '__main__':
    main(
        model_path="./checkpoints/best_e2e_multi.pth",
        train_path="./data/e2e_dialogue/dialogue_data_multi_with_slots.pkl",
        dic_path_with_tokens="./data/e2e_dialogue/intent2id_multi_with_tokens.pkl",
        slot_path="./data/e2e_dialogue/slot2id.pkl",
        datatype="e2e",
        data_mode="multi",
        test_mode="validation",
        batch_size=8,
        maxlen=60
    )