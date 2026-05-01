# Yash-Dattani-TEAM-24-CS439-Professor-Wang-Final-Project-
Group 24: Yash Dattani
# NBA Game Outcome Prediction

## Overview
This project predicts NBA game outcomes using machine learning models based on rolling team performance statistics.
The program starts in main.py, which acts as the entry point and runs the full pipeline in order: it first calls load_nba_games() in src/data_loader.py, which downloads NBA data (via the NBA API or cached file) and saves it into data/raw/games_raw.csv; next, create_features() in src/features.py reads that raw file, performs cleaning and feature engineering (including rolling averages and shifts to avoid data leakage), and writes the processed dataset to data/processed/games_processed.csv; then train_models() in src/train.py loads the processed data, splits it chronologically into train/test sets, trains models like Logistic Regression and XGBoost, and returns trained models plus test data; after that, evaluate_models() in src/evaluate.py computes metrics like accuracy, precision, recall, F1, and generates visual outputs such as confusion matrices; finally, plot_results() saves figures like model comparison charts into an automatically created outputs/ folder (e.g., outputs/model_accuracy.png), making the entire workflow fully reproducible from raw data to saved results just by running python main.py.
## Setup
```bash
pip install -r requirements.txt
python main.py
