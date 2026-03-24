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
    artifact_path = "mlp_model_dvc"

    # Step 1: Get the root_uri to find the experiment ID
    list_url = f"{tracking_uri}/api/2.0/mlflow/artifacts/list"
    print("Fetching artifact root URI...")
    try:
        resp = requests.get(list_url, params={"run_id": run_id}, auth=auth, timeout=30)
        resp.raise_for_status()
        root_data = resp.json()
        root_uri = root_data.get("root_uri", "")
        print(f"Root URI: {root_uri}")
    except Exception as e:
        print(f"Failed to get root URI: {e}")
        sys.exit(1)

    # Step 2: Use the mlflow-artifacts proxy endpoint to list files
    # root_uri format: mlflow-artifacts:/{experiment_id}/{run_id}/artifacts
    # Proxy URL: {tracking_uri}/api/2.0/mlflow-artifacts/artifacts/{experiment_id}/{run_id}/artifacts/{path}
    proxy_path = root_uri.replace("mlflow-artifacts:/", "")
    proxy_list_url = f"{tracking_uri}/api/2.0/mlflow-artifacts/artifacts/{proxy_path}/{artifact_path}"

    print(f"Listing artifacts via proxy: {proxy_list_url}")
    try:
        resp = requests.get(proxy_list_url, auth=auth, timeout=30)
        resp.raise_for_status()
        proxy_data = resp.json()
        print(f"Proxy response: {json.dumps(proxy_data, indent=2)}")
        files = proxy_data.get("files", [])
    except Exception as e:
        print(f"Proxy listing failed: {e}")
        # Fallback: try without the artifact_path to see what's available
        fallback_url = f"{tracking_uri}/api/2.0/mlflow-artifacts/artifacts/{proxy_path}"
        print(f"Trying fallback listing: {fallback_url}")
        try:
            resp2 = requests.get(fallback_url, auth=auth, timeout=30)
            resp2.raise_for_status()
            fallback_data = resp2.json()
            print(f"Fallback response: {json.dumps(fallback_data, indent=2)}")
            files = fallback_data.get("files", [])
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            sys.exit(1)

    if not files:
        print("No files found via proxy either. Trying direct DagsHub storage API...")
        # Step 3: Try DagsHub's direct storage API
        # DagsHub stores MLflow artifacts at: /api/v1/repos/{owner}/{repo}/storage/raw/mlflow-artifacts/{path}
        # Extract owner/repo from tracking URI
        # tracking_uri like: https://dagshub.com/lorddiobrando/ML-CI-CD.mlflow
        parts = tracking_uri.replace("https://dagshub.com/", "").replace(".mlflow", "").split("/")
        if len(parts) >= 2:
            owner, repo = parts[0], parts[1]
            storage_url = f"https://dagshub.com/api/v1/repos/{owner}/{repo}/storage/raw/mlflow-artifacts/{proxy_path}/{artifact_path}"
            print(f"Trying DagsHub storage: {storage_url}")
            try:
                resp3 = requests.get(storage_url, auth=auth, timeout=30, allow_redirects=True)
                print(f"Storage response status: {resp3.status_code}")
                print(f"Storage response: {resp3.text[:500]}")
            except Exception as e3:
                print(f"Storage API failed: {e3}")

        print("Could not find artifacts. Exiting.")
        sys.exit(1)

    # Step 4: Download each file
    os.makedirs("model", exist_ok=True)
    print(f"Found {len(files)} file(s) to download")

    for file_info in files:
        file_path = file_info.get("path", "")
        file_name = os.path.basename(file_path)
        file_size = file_info.get("file_size", 0)
        if file_info.get("is_dir", False):
            print(f"  Skipping subdirectory: {file_path}")
            continue

        download_url = f"{tracking_uri}/api/2.0/mlflow-artifacts/artifacts/{proxy_path}/{artifact_path}/{file_path}"
        print(f"  Downloading: {file_name} ({file_size} bytes)")

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
