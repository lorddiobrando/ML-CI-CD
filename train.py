import argparse
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay

# Changed to force DVC to rerun the training stage in CI
mlflow.set_experiment("DSAI_406_A3_AmrKhalid_Sklearn_DVC")

def parse_args():
    """Parse CLI options controlling training and model hyperparameters."""
    parser = argparse.ArgumentParser(description="Train an MLP classifier on DVC-tracked digits dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--hidden-layer-sizes", type=str, default="100", help="Hidden layer sizes (comma-separated).")
    parser.add_argument("--max-iter", type=int, default=200, help="Maximum number of iterations.")
    parser.add_argument("--no-show", action="store_true", help="Skip matplotlib plots.")
    parser.add_argument("--data-path", type=str, default="data/digits.csv", help="Path to the DVC-tracked CSV.")
    return parser.parse_args()


def set_reproducibility(seed):
    """Seed all RNG sources for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def main():
    """Train MLP model and log results to MLflow."""
    args = parse_args()
    set_reproducibility(args.seed)

    print(f"Running with seed={args.seed}, hidden_layer_sizes={args.hidden_layer_sizes}, max_iter={args.max_iter}")

    # 1) Load Dataset from DVC-tracked CSV
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found at {args.data_path}. Did you run save_data.py and dvc pull?")

    print(f"Loading dataset from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    X = df.drop(columns=['target']).values
    y = df['target'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed
    )

    with mlflow.start_run() as run:
        mlflow.set_tag("student_name", "Amr Khalid")
        mlflow.set_tag("student_id", "21102490")
        mlflow.set_tag("model_type", "MLP_DVC")

        # Log parameters
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("hidden_layer_sizes", args.hidden_layer_sizes)
        mlflow.log_param("max_iter", args.max_iter)
        mlflow.log_param("data_path", args.data_path)

        # 2) Initialize and Train Model
        # Convert string like "100,50" to tuple (100, 50)
        hidden_layers = tuple(int(x) for x in args.hidden_layer_sizes.split(','))
        
        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            max_iter=args.max_iter,
            random_state=args.seed
        )
        clf.fit(X_train, y_train)

        # 3) Evaluate Model
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:\n", report)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)

        # 4) Log Model
        mlflow.sklearn.log_model(
            clf, "mlp_model_dvc",
            registered_model_name="mlp_digits_dvc",
            input_example=X_train[:1]
        )

        # 5) Visualizations (Confusion Matrix)
        if not args.no_show:
            fig, ax = plt.subplots(figsize=(8, 8))
            ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, ax=ax, cmap="Blues")
            plt.title("Confusion Matrix - Digits (MLP)")
            
            # Save plot to a temporary file and log as artifact
            plot_path = "confusion_matrix_dvc_mlp.png"
            plt.savefig(plot_path)
            mlflow.log_artifact(plot_path)
            plt.show()
            
            # Clean up plot file
            if os.path.exists(plot_path):
                os.remove(plot_path)

if __name__ == "__main__":
    main()
