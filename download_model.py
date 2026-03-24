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

    auth = (username, password) if username and password else None
    list_url = f"{tracking_uri}/api/2.0/mlflow/artifacts/list"

    # Step 1: List ALL artifacts at root to see what exists
    print("Listing all artifacts at root...")
    try:
        resp = requests.get(list_url, params={"run_id": run_id}, auth=auth, timeout=30)
        resp.raise_for_status()
        root_data = resp.json()
        print(f"Root listing response: {json.dumps(root_data, indent=2)}")
    except Exception as e:
        print(f"Failed to list root artifacts: {e}")
        sys.exit(1)

    root_files = root_data.get("files", [])
    if not root_files:
        print("No artifacts found at root level. The model may not have been logged in this run.")
        sys.exit(1)

    # Find the model directory (could be mlp_model_dvc or something else)
    model_dir = None
    for f in root_files:
        print(f"  Found: {f.get('path')} (is_dir={f.get('is_dir', False)})")
        if f.get("is_dir", False):
            model_dir = f.get("path")

    if not model_dir:
        # If no directories, download all root files directly
        model_dir = None
        files_to_download = root_files
        print("No subdirectories found, downloading root files...")
    else:
        print(f"Found model directory: {model_dir}")
        # Step 2: List files inside the model directory
        try:
            resp = requests.get(list_url, params={"run_id": run_id, "path": model_dir}, auth=auth, timeout=30)
            resp.raise_for_status()
            dir_data = resp.json()
            files_to_download = dir_data.get("files", [])
            print(f"Found {len(files_to_download)} file(s) inside '{model_dir}'")
        except Exception as e:
            print(f"Failed to list directory contents: {e}")
            sys.exit(1)

    if not files_to_download:
        print("No files found to download.")
        sys.exit(1)

    # Step 3: Download each file
    os.makedirs("model", exist_ok=True)

    for file_info in files_to_download:
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
