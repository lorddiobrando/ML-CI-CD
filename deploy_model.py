import argparse
import os
import sys
import mlflow

def deploy_model(run_id):
    """Register or update the model in the MLflow Model Registry."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    model_name = "mlp_digits_dvc"
    model_uri = f"runs:/{run_id}/mlp_model_dvc"

    print(f"Registering model from Run ID: {run_id}")
    print(f"Model URI: {model_uri}")
    print(f"Model Name: {model_name}")

    try:
        result = mlflow.register_model(model_uri=model_uri, name=model_name)
        print(f"Model registered successfully!")
        print(f"  Name: {result.name}")
        print(f"  Version: {result.version}")
        print(f"  Status: {result.status}")
    except Exception as e:
        print(f"Model registration note: {e}")
        print("Model may already be registered from the training step. Continuing.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="MLflow Run ID")
    args = parser.parse_args()
    deploy_model(args.run_id)
