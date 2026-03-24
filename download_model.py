import os
import sys

def download_model():
    run_id = os.environ.get("RUN_ID")
    if not run_id:
        print("Error: RUN_ID environment variable not set.")
        sys.exit(1)

    # Use dagshub.init() to properly configure MLflow tracking AND artifact storage
    # This is the officially recommended way per DagsHub documentation
    import dagshub
    dagshub.init(repo_owner="lorddiobrando", repo_name="ML-CI-CD", mlflow=True)

    import mlflow
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Downloading model for Run ID: {run_id}")

    artifact_path = "mlp_model_dvc"
    model_uri = f"runs:/{run_id}/{artifact_path}"

    try:
        local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path="model")
        print(f"Model successfully downloaded to {local_path}")
    except Exception as e:
        print(f"Failed to download model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_model()
