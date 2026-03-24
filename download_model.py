import os
import sys
import requests
import json

def download_model():
    run_id = os.environ.get("RUN_ID")
    if not run_id:
        print("Error: RUN_ID environment variable not set.")
        sys.exit(1)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "").rstrip("/")
    username = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
    password = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")

    if not tracking_uri:
        print("Error: MLFLOW_TRACKING_URI not set.")
        sys.exit(1)

    print(f"Using MLflow tracking URI: {tracking_uri}")
    print(f"Downloading model for Run ID: {run_id}")

    artifact_path = "mlp_model_dvc"
    auth = (username, password) if username and password else None

    # Step 1: List all files inside the artifact directory
    list_url = f"{tracking_uri}/api/2.0/mlflow/artifacts/list"
    params = {"run_id": run_id, "path": artifact_path}

    print(f"Listing artifacts at path: {artifact_path}")
    try:
        resp = requests.get(list_url, params=params, auth=auth, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"Failed to list artifacts: {e}")
        sys.exit(1)

    files = data.get("files", [])
    if not files:
        print(f"No artifacts found at path '{artifact_path}'. Response: {json.dumps(data, indent=2)}")
        sys.exit(1)

    print(f"Found {len(files)} artifact file(s)")

    # Step 2: Download each file
    os.makedirs("model", exist_ok=True)

    for file_info in files:
        file_path = file_info.get("path", "")
        file_name = os.path.basename(file_path)
        if file_info.get("is_dir", False):
            print(f"  Skipping subdirectory: {file_path}")
            continue

        download_url = f"{tracking_uri}/get-artifact?path={file_path}&run_uuid={run_id}"
        print(f"  Downloading: {file_name}")

        try:
            file_resp = requests.get(download_url, auth=auth, timeout=120)
            file_resp.raise_for_status()

            local_path = os.path.join("model", file_name)
            with open(local_path, "wb") as f:
                f.write(file_resp.content)
            print(f"  Saved to: {local_path} ({len(file_resp.content)} bytes)")
        except Exception as e:
            print(f"  Failed to download {file_name}: {e}")
            sys.exit(1)

    print("Model successfully downloaded to ./model")

if __name__ == "__main__":
    download_model()
