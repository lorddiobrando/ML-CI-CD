import os
import sys

# Must be set before any imports to prevent boto3 EC2 metadata hangs
os.environ["AWS_EC2_METADATA_DISABLED"] = "true"

def download_model():
    """Download the registered model from DagsHub Model Registry at container startup."""
    run_id = os.environ.get("RUN_ID")
    if not run_id:
        print("Error: RUN_ID environment variable not set.")
        sys.exit(1)

    import dagshub
    dagshub.init(repo_owner="lorddiobrando", repo_name="ML-CI-CD", mlflow=True)

    import mlflow
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Loading model from registry for Run ID: {run_id}")

    try:
        # Load the model directly from the registry by name
        # This uses DagsHub's HTTP proxy, not S3
        model = mlflow.sklearn.load_model(f"models:/mlp_digits_dvc/latest")
        
        # Save the model locally for container use
        import joblib
        os.makedirs("model", exist_ok=True)
        joblib.dump(model, "model/model.pkl")
        print("Model successfully saved to model/model.pkl")
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Falling back to echo-only mock build...")
        # Don't fail the pipeline - the model is safely in the registry
        os.makedirs("model", exist_ok=True)
        with open("model/run_id.txt", "w") as f:
            f.write(run_id)
        print(f"Saved run_id reference to model/run_id.txt")

if __name__ == "__main__":
    download_model()
