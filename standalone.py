import argparse
import pandas as pd
import os
import torch
import json
import yaml
import joblib
from huggingface_hub import hf_hub_download
import importlib.util


def validate_file(filepath, separator):
    """Validates if the file exists and can be read."""
    try:
        return pd.read_csv(filepath, sep=separator)
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
    except pd.errors.EmptyDataError:
        print(f"Error: File is empty: {filepath}")
    except pd.errors.ParserError:
        print(f"Error: Parsing error with separator '{separator}': {filepath}")
    return None



def load_create_model(repo_id):
    """Load create_model from HF repo (network.py)."""
    network_file = hf_hub_download(repo_id, "network.py")

    spec = importlib.util.spec_from_file_location("network", network_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.create_model


def predict_probabilities(df, model_dir, ordered_cols):
    """Predict probabilities using per-gene LR models."""
    results = []

    for col in ordered_cols:
        model_path = os.path.join(model_dir, f"lr_models/{col}.pkl")

        try:
            model = joblib.load(model_path)
            probs = model.predict_proba(df[[col]])[:, 1]
            results.append(pd.Series(probs, name=col))

        except FileNotFoundError:
            print(f"Error: Missing model for {col}: {model_path}")
            return None

        except Exception as e:
            print(f"Error in {col}: {e}")
            return None

    return pd.concat(results, axis=1)


def probability_to_amino_acid(p):
    """Map probability to amino acid."""
    bins = [
        (0.00, 0.05, 'A'), (0.05, 0.10, 'C'), (0.10, 0.15, 'D'),
        (0.15, 0.20, 'E'), (0.20, 0.25, 'F'), (0.25, 0.30, 'G'),
        (0.30, 0.35, 'H'), (0.35, 0.40, 'I'), (0.40, 0.45, 'K'),
        (0.45, 0.50, 'L'), (0.50, 0.55, 'M'), (0.55, 0.60, 'N'),
        (0.60, 0.65, 'P'), (0.65, 0.70, 'Q'), (0.70, 0.75, 'R'),
        (0.75, 0.80, 'S'), (0.80, 0.85, 'T'), (0.85, 0.90, 'V'),
        (0.90, 0.95, 'W'), (0.95, 1.00, 'Y')
    ]
    for low, high, aa in bins:
        if low <= p < high or (p == 1.0 and aa == 'Y'):
            return aa
    return 'X'


def load_hf_model():
    repo_id = "shubhamc-iiitd/pdac_pred_llm"

    model_path = hf_hub_download(repo_id, "model.pt")
    config_path = hf_hub_download(repo_id, "config.yaml")
    mapping_path = hf_hub_download(repo_id, "tokenizer_mapping.json")

    config = yaml.safe_load(open(config_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["device"] = device

    # ✅ load model definition dynamically
    create_model = load_create_model(repo_id)

    model = create_model(config)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=False)

    model.to(device)
    model.eval()

    mapping = json.load(open(mapping_path))

    return model, mapping, device


def predict_sequence(model, mapping, device, sequence):
    """Predict PDAC probability from peptide."""
    tokens = [mapping.get(c, mapping.get("[UNK]", 0)) for c in sequence]
    input_ids = torch.tensor([tokens]).to(device)

    attention_mask = (input_ids != 0).float()

    with torch.no_grad():
        output = model(input_ids, attention_mask)
        prob = output[0].item()

    return prob


def main():
    parser = argparse.ArgumentParser(description="PDAC prediction pipeline")
    parser.add_argument("filepath", help="Input CSV file")
    parser.add_argument("--separator", "-s", default=",")
    parser.add_argument("--output", "-o", default="output.csv")
    args = parser.parse_args()

    model_dir = os.path.dirname(os.path.abspath(__file__))

    df = validate_file(args.filepath, args.separator)
    if df is None:
        exit(1)

    if df.index.name is None:
        df.index = [f"sample_{i+1}" for i in range(len(df))]

    # ✅ Updated gene set
    required_columns = [
        'ENSG00000171345',
        'ENSG00000163347',
        'ENSG00000168685',
        'ENSG00000151655',
        'ENSG00000152601'
    ]

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        print(f"Error: Missing columns: {missing}")
        exit(1)

    # Step 1: LR predictions
    prob_df = predict_probabilities(df, model_dir, required_columns)
    if prob_df is None:
        exit(1)
    print(prob_df.shape)
    # Step 2: Convert to peptide sequences
    aa_df = prob_df.map(probability_to_amino_acid)
    sequences = aa_df.apply(lambda row: ''.join(row), axis=1).tolist()
    print(sequences)
    # Step 3: Load HF model
    model, mapping, device = load_hf_model()

    # Step 4: Predict
    probs = []
    labels = []

    for seq in sequences:
        p = predict_sequence(model, mapping, device, seq)
        probs.append(p)
        labels.append('PDAC' if p > 0.5 else 'non-PDAC')

    # Step 5: Output
    result = pd.DataFrame({
        "Sample_ID": df.index,
        "PDAC Probability": probs,
        "Prediction": labels
    })

    result.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
