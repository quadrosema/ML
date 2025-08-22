# Stroke Prediction ML Project

This repository contains a simple machine learning pipeline for predicting stroke occurrences from health records. The code loads a dataset, performs preprocessing, balances and splits the data and evaluates several classification models.

## Project Structure

```
ML/
├── data/                # Sample dataset
└── src/                 # Source code for the ML pipeline
    ├── main.py          # Entry point that runs the full workflow
    ├── preprocess.py    # Data cleaning and encoding utilities
    ├── read.py          # Dataset loading and exploratory helpers
    ├── prepare.py       # Feature selection and train/test split
    └── models.py        # Model training and evaluation
```

## Installation

Ensure Python 3.8+ is installed. Install required packages:

```bash
pip install pandas seaborn matplotlib scikit-learn imbalanced-learn xgboost lightgbm
```

## Usage

From the repository root, run the pipeline:

```bash
python -m src.main
```

The script will output evaluation metrics for each model.

## Data

The `data/dataset.csv` file is a small sample dataset used for demonstrations. Replace it with your own data as needed.

## License

This project is provided for educational purposes.
