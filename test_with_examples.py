"""
Independent test script: Load trained model and print specific prediction examples
(text + real labels + predicted labels)
"""
import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
from transformers import BertTokenizer, BertConfig

# Import necessary functions from utils.py
from utils import f1_score_intents, evaluate_iob, prf, load_data

# Import core dependencies from original code
from model import BertContextNLU
from all_data_context import get_dataloader_context
from config import opt

def load_dictionaries():
    """Load intent/slot dictionaries and build reverse mapping (ID to name)"""
    # Load intent dictionary (intent2id_multi_with_tokens.pkl)
    with open(opt.dic_path_with_tokens, 'rb') as f:
        intent_dic = pickle.load(f)
    # Reverse mapping: intent ID -> intent name
    intent_id2name = {v[0]: k for k, v in intent_dic.items()}

    # Load slot dictionary (slot2id.pkl)
    with open(opt.slot_path, 'rb') as f:
        slot_dic = pickle.load(f)
    # Reverse mapping: slot ID -> slot name
    slot_id2name = {v: k for k, v in slot_dic.items()}

    return intent_id2name, slot_id2name

def load_test_data():
    """Load and split test data (same logic as original code)"""
    with open(opt.train_path, 'rb') as f:
        train_data = pickle.load(f)

    # Split train/test set by 7:3 with fixed random seed (match original code)
    np.random.seed(0)
    indices = np.random.permutation(len(train_data))
    test_data = np.array(train_data, dtype=object)[indices[int(len(train_data)*0.7):]][:100]
    return test_data

def init_model(intent_num, slot_num):
    """Initialize model and load pretrained weights"""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Initialize BERT config (match original code/config)
    config = BertConfig(
        vocab_size_or_config_json_file=32000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072
    )

    # Initialize model
    model = BertContextNLU(config, opt, intent_num, slot_num)

    # Load trained model weights with safe device mapping
    if opt.model_path and os.path.exists(opt.model_path):
        state_dict = torch.load(opt.model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Successfully loaded pretrained model: {}".format(opt.model_path))
    else:
        raise ValueError("Model file does not exist: {}".format(opt.model_path))

    model = model.to(device)
    model.eval()  # Switch to evaluation mode
    return model, device

def get_intent_tokens(intent_dic, device):
    """Generate intent tokens (required input for original model)"""
    intent_tokens = [intent for name, (tag, intent) in intent_dic.items()]
    intent_tok, mask_tok = load_data(intent_tokens, 10)

    # Convert to tensor (match original code)
    intent_tokens_tensor = torch.zeros(len(intent_tok), 10).long().to(device)
    mask_tokens_tensor = torch.zeros(len(mask_tok), 10).long().to(device)
    for i in range(len(intent_tok)):
        intent_tokens_tensor[i] = torch.tensor(intent_tok[i])
    for i in range(len(mask_tok)):
        mask_tokens_tensor[i] = torch.tensor(mask_tok[i])

    return intent_tokens_tensor, mask_tokens_tensor

def print_prediction_examples(model, test_loader, intent_id2name, slot_id2name, device, intent_tokens, mask_tokens):
    """Core function: Run test and print specific prediction examples (fixed dimension handling)"""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    all_examples = []  # Store all test examples results

    # Iterate through test data batches (batch size = 8 from config)
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
        # Unpack batch data (match original data loader output exactly)
        result_ids, result_token_masks, result_masks, lengths, result_slot_labels, result_labels = batch

        # Move all tensors to correct device (ensure consistent dimensions)
        result_ids = result_ids.to(device)
        result_token_masks = result_token_masks.to(device)
        result_masks = result_masks.to(device)
        lengths = lengths.to(device)
        result_slot_labels = result_slot_labels.to(device)
        result_labels = result_labels.to(device)

        # Critical fix: Get actual batch size from lengths tensor (config batch_size=8)
        actual_batch_size = lengths.shape[0]

        # Forward pass (no gradient computation)
        with torch.no_grad():
            outputs, labels, predicted_slot_outputs = model(
                result_ids, result_token_masks, result_masks, lengths,
                result_slot_labels, result_labels, intent_tokens, mask_tokens
            )

        # Parse results for EACH sample in the batch (0 to actual_batch_size-1 only)
        for i in range(actual_batch_size):
            try:
                # 1. Safe length retrieval (absolute protection against out-of-bounds)
                sample_length = lengths[i].item() if i < lengths.shape[0] else opt.maxlen
                seq_len = min(sample_length, result_ids.shape[1])

                # 2. Correct token ID extraction (handle all dimension cases)
                # Result_ids shape: [batch_size, maxlen, 1] (from original data loader)
                if len(result_ids.shape) == 3:
                    # Extract token IDs and remove padding dimension
                    text_ids = result_ids[i, :seq_len, 0].squeeze()
                else:
                    text_ids = result_ids[i, :seq_len]

                # 3. Safe text decoding (filter invalid tokens)
                text_ids = text_ids.cpu().numpy()
                text_ids = text_ids[text_ids != 0]  # Remove padding tokens (0)
                text = tokenizer.decode(text_ids, skip_special_tokens=True)

                # 4. Parse real intents (multi-intent handling)
                real_intent_ids = torch.where(labels[i] == 1)[0].cpu().numpy()
                real_intents = [intent_id2name.get(id, "Unknown_ID_{}".format(id)) for id in real_intent_ids]

                # 5. Parse predicted intents (sigmoid threshold = 0.5)
                pred_intent_logits = torch.sigmoid(outputs[i])
                pred_intent_ids = torch.where(pred_intent_logits > 0.5)[0].cpu().numpy()
                pred_intents = [intent_id2name.get(id, "Unknown_ID_{}".format(id)) for id in pred_intent_ids]

                # 6. Check if prediction is correct
                is_correct = set(real_intents) == set(pred_intents)

                # Store results
                all_examples.append({
                    "text": text,
                    "real_intent_labels": real_intents,
                    "predicted_intent_labels": pred_intents,
                    "is_intent_correct": is_correct
                })

            except IndexError as e:
                # Skip problematic sample instead of crashing
                print("\nWarning: Skipping sample {} in batch {} (IndexError: {})".format(i, batch_idx, e))
                continue

    # ========== Print formatted results ==========
    print("\n" + "="*100)
    print("Test sample prediction examples (first 10):")
    print("="*100)
    for idx, example in enumerate(all_examples[:10]):
        print("\n[Sample {}]".format(idx+1))
        print("Text: {}".format(example['text']))
        print("Real intent labels: {}".format(example['real_intent_labels']))
        print("Predicted intent labels: {}".format(example['predicted_intent_labels']))
        print("Is prediction correct: {}".format(example['is_intent_correct']))

    # Print error examples (for analysis)
    error_examples = [e for e in all_examples if not e["is_intent_correct"]]
    if error_examples:
        print("\n" + "="*100)
        print("Incorrect prediction examples (first 5, total {}):".format(len(error_examples)))
        print("="*100)
        for idx, example in enumerate(error_examples[:5]):
            print("\n[Error Sample {}]".format(idx+1))
            print("Text: {}".format(example['text']))
            print("Real intent labels: {}".format(example['real_intent_labels']))
            print("Predicted intent labels: {}".format(example['predicted_intent_labels']))

    # Print overall statistics
    total = len(all_examples)
    correct = sum([1 for e in all_examples if e["is_intent_correct"]])
    accuracy = correct / total if total > 0 else 0
    print("\n" + "="*100)
    print("Overall statistics:")
    print("Total valid test samples: {}".format(total))
    print("Correct predictions: {}".format(correct))
    print("Intent prediction accuracy: {:.4f}".format(accuracy))
    print("="*100)

def main(**kwargs):
    # 1. Override config parameters with input arguments
    for k, v in kwargs.items():
        setattr(opt, k, v)

    # 2. Load dictionaries (intent/slot ID to name mapping)
    intent_id2name, slot_id2name = load_dictionaries()
    with open(opt.dic_path_with_tokens, 'rb') as f:
        intent_dic = pickle.load(f)

    # 3. Load test data (7:3 split from train data)
    test_data = load_test_data()

    # 4. Create data loader (use batch size from config)
    test_loader = get_dataloader_context(test_data, intent_dic, slot_id2name, opt)

    # 5. Initialize model (match intent/slot count from dictionaries)
    model, device = init_model(len(intent_dic), len(slot_id2name))

    # 6. Generate intent tokens (required for model input)
    intent_tokens, mask_tokens = get_intent_tokens(intent_dic, device)

    # 7. Run test and print prediction examples
    print_prediction_examples(model, test_loader, intent_id2name, slot_id2name, device, intent_tokens, mask_tokens)

if __name__ == '__main__':
    # Run with config matching your environment (update paths as needed)
    main(
        model_path="./checkpoints/best_e2e_multi.pth",
        train_path="./data/e2e_dialogue/dialogue_data_multi_with_slots.pkl",
        dic_path_with_tokens="./data/e2e_dialogue/intent2id_multi_with_tokens.pkl",
        slot_path="./data/e2e_dialogue/slot2id.pkl",
        datatype="e2e",
        data_mode="multi",
        test_mode="validation",
        batch_size=8,  # Match config batch size
        maxlen=60      # Match config max sequence length
    )