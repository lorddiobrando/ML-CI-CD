import os
import sys

# CRITICAL: Must be set BEFORE importing boto3/mlflow/dagshub
# Prevents boto3 from hanging for 4 minutes trying to reach EC2 metadata
os.environ["AWS_EC2_METADATA_DISABLED"] = "true"

def download_model():
    run_id = os.environ.get("RUN_ID")
    if not run_id:
        print("Error: RUN_ID environment variable not set.")
        sys.exit(1)

    username = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
    password = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")

    # Configure DagsHub S3 storage credentials explicitly
    # DagsHub uses username/token as AWS credentials for its S3 proxy
    os.environ["AWS_ACCESS_KEY_ID"] = username
    os.environ["AWS_SECRET_ACCESS_KEY"] = password
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://dagshub.com/lorddiobrando/ML-CI-CD.s3"

    import dagshub
    dagshub.init(repo_owner="lorddiobrando", repo_name="ML-CI-CD", mlflow=True)

    import mlflow
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Downloading model for Run ID: {run_id}")
    print(f"S3 endpoint: {os.environ.get('MLFLOW_S3_ENDPOINT_URL')}")

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
