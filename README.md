# Classification of Pancreatic Ductal Carcinoma Patients using a Large Language Model
 
### This Python script provides a command-line interface (CLI) for predicting PDAC patients based on the expression of a 5-gene biomarker using a fine-tuned LLM model hosted on Hugging Face.
 
## Prerequisites
* Python (>=3.9)
* scikit-learn == 1.6.1
* pandas
* torch == 2.5.1
* PyYAML
* huggingface_hub == 0.29.3
 
You can install the required packages using pip:
```bash
pip install pandas torch==2.5.1 huggingface_hub==0.29.3 scikit-learn==1.6.1 pyyaml
```
 
## Required Input Columns
The input CSV must contain the following 5 Ensembl gene IDs as columns:
```
ENSG00000171345
ENSG00000163347
ENSG00000168685
ENSG00000151655
ENSG00000152601
```
 
## Usage
```bash
python standalone.py <filepath> [--separator <separator>] [--output <output filepath>]
```
 
* `<filepath>`: Path to the input gene expression CSV file (values in TPM).
* `--separator` or `-s`: Separator used in the input file (default: `,`).
* `--output` or `-o`: Path to the output file (default: `output.csv`).
 
## Description
The script performs the following steps:
 
1. **Input Data Validation**: Reads the input gene expression (TPM) file using pandas, with a user-specified separator. Checks that all 5 required gene columns are present.
 
2. **Probability Prediction**: Uses pre-trained Logistic Regression (LR) models (pickled `.pkl` files named `<ENSG_ID>.pkl`) located under a `lr_models/` subdirectory relative to the script. Probabilities are predicted per gene for each sample.
 
3. **Amino Acid Conversion**: Maps each predicted probability to one of 20 amino acid characters based on fixed 0.05-width bins spanning [0.0, 1.0].
 
4. **Peptide Sequence Generation**: Concatenates the 5 amino acids per sample row into a 5-character peptide sequence.
 
5. **LLM Classification**:
   * Dynamically downloads the model definition (`network.py`), weights (`model.pt`), config (`config.yaml`), and tokenizer mapping (`tokenizer_mapping.json`) from the Hugging Face repository [`shubhamc-iiitd/pdac_pred_llm`](https://huggingface.co/shubhamc-iiitd/pdac_pred_llm).
   * Tokenizes each peptide sequence using the downloaded mapping.
   * Runs inference on CPU or CUDA (auto-detected).
   * Classifies each sample as `PDAC` (probability > 0.5) or `non-PDAC`.
 
6. **Output**: Saves a CSV with columns `Sample_ID`, `PDAC Probability`, and `Prediction`.
 
## Model Directory Structure
```
<script_dir>/
├── standalone.py
└── lr_models/
    ├── ENSG00000171345.pkl
    ├── ENSG00000163347.pkl
    ├── ENSG00000168685.pkl
    ├── ENSG00000151655.pkl
    └── ENSG00000152601.pkl
```
 
## Example
```bash
python standalone.py data.csv --separator "," --output results/output.csv
```
 
This reads `data.csv` with comma separation and writes predictions to `results/output.csv`.
 
### Example Output (`output.csv`)
| Sample_ID | PDAC Probability | Prediction |
|-----------|-----------------|------------|
| sample_1  | 0.82            | PDAC       |
| sample_2  | 0.31            | non-PDAC   |
 
## Code Structure
 
| Function | Description |
|---|---|
| `validate_file(filepath, separator)` | Validates and loads the input CSV file. |
| `load_create_model(repo_id)` | Dynamically downloads and loads the `create_model` function from `network.py` on Hugging Face. |
| `predict_probabilities(df, model_dir, ordered_cols)` | Predicts per-gene probabilities using pre-trained LR models. |
| `probability_to_amino_acid(p)` | Maps a probability value to a single-letter amino acid code. |
| `load_hf_model()` | Downloads and initialises the LLM from Hugging Face Hub. |
| `predict_sequence(model, mapping, device, sequence)` | Runs inference on a tokenised peptide sequence and returns PDAC probability. |
| `main()` | Orchestrates the full prediction pipeline. |
