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
    
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        # FIX: boto3 hangs for exactly 4 minutes in GitHub Actions trying to fetch EC2 metadata 
        # when downloading S3-based MLflow artifacts from DagsHub over a proxied connection.
        os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
        # Provide fallback AWS credentials mapped to DagsHub
        os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get("MLFLOW_TRACKING_USERNAME", "dummy")
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get("MLFLOW_TRACKING_PASSWORD", "dummy")
        
        print("Using MlflowClient to fetch the artifact...")
        # Downloads the directory into the current working directory under ./mlp_model_dvc
        local_dir = client.download_artifacts(run_id, "mlp_model_dvc", ".")
        
        # Rename it to 'model' so the Dockerfile COPY command works as expected
        if os.path.exists("model"):
            import shutil
            shutil.rmtree("model")
        os.rename(local_dir, "model")
        
        print("Model successfully downloaded and saved to ./model")
    except Exception as e:
        print(f"Failed to download model. Ensure you have network access and valid credentials: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_model()
