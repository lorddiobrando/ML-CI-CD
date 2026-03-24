import os
import sys
import mlflow

def download_model():
    run_id = os.environ.get("RUN_ID")
    if not run_id:
        print("Error: RUN_ID environment variable not set.")
        sys.exit(1)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"Using MLflow tracking URI: {tracking_uri}")
    else:
        print("Warning: MLFLOW_TRACKING_URI not set. Downloading from local mlruns.")

    print(f"Downloading model for Run ID: {run_id}")
    
    # The artifact path used in train.py when logging the model
    model_uri = f"runs:/{run_id}/mlp_model_dvc"
    
    try:
        local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path="model")
        print(f"Model successfully downloaded to {local_path}")
    except Exception as e:
        print(f"Failed to download model. Ensure you have network access and valid credentials: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_model()
