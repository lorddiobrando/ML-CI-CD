import argparse
import mlflow
import sys
import os

def check_accuracy(run_id, threshold=0.85):
    """Fetch the run from MLflow and verify its accuracy."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        print("Warning: MLFLOW_TRACKING_URI not set, using default local tracking.")
        
    print(f"Checking accuracy for Run ID: {run_id}")
    
    try:
        run = mlflow.get_run(run_id)
        accuracy = run.data.metrics.get("accuracy")
        
        if accuracy is None:
            print("Error: 'accuracy' metric not found in this run.")
            sys.exit(1)
            
        print(f"Accuracy found: {accuracy:.4f}")
        
        if accuracy < threshold:
            print(f"Test failed: Accuracy {accuracy:.4f} is below threshold {threshold}.")
            sys.exit(1)
        else:
            print(f"Test passed: Accuracy {accuracy:.4f} meets or exceeds threshold {threshold}.")
            sys.exit(0)
            
    except Exception as e:
        print(f"An error occurred while fetching run {run_id}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check model accuracy against a threshold.")
    parser.add_argument("--run-id", required=True, help="MLflow Run ID to check")
    parser.add_argument("--threshold", type=float, default=0.85, help="Minimum accuracy threshold (default: 0.85)")
    args = parser.parse_args()
    
    check_accuracy(args.run_id, args.threshold)