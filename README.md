# Shakespeare Dataset Training Examples

This repository provides code and resources to replicate experiments using the Shakespeare dataset for language modeling under three different architectures:

- **Classic Reservoir Network** (`train_reservoir.py`)
- **Attention-Enhanced Reservoir Network** (`train_att_reservoir.py`)
- **Character-Level Transformer** (`train_transformer.py`)

Each script demonstrates:

- Data loading & preprocessing
- Model definition
- Mixed-precision training (AMP)
- Logging train/test losses to CSV
- Saving model checkpoints

## Directory Structure

```text
.
├── .gitignore
├── LICENSE
├── README.md
├── CONTRIBUTING.md
├── requirements.txt
│
├── data/
│   └── shakespeare.txt
│
├── models/
│   ├── reservoir.py
│   ├── att_reservoir.py
│   └── transformer.py
│
├── utils/
│   ├── data_utils.py
│   └── training_utils.py
│
├── train_reservoir.py
├── train_att_reservoir.py
└── train_transformer.py

