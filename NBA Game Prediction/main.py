from src.data_loader import load_nba_games
from src.features import create_features
from src.train import train_models
from src.evaluate import evaluate_models, plot_results

def main():
    print("Loading data...")
    df = load_nba_games()

    print("Creating features...")
    df = create_features(df)

    print("Training models...")
    models, X_test, y_test = train_models(df)

    print("Evaluating models...")
    results = evaluate_models(models, X_test, y_test)

    print("Plotting results...")
    plot_results(results)

    print("Done. Outputs saved in /outputs")

if __name__ == "__main__":
    main()
