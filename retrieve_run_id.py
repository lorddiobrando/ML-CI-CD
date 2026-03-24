import mlflow
import os
import sys

def retrieve_latest_run_id():
    """Fetch the latest run ID for the specified experiment and save it to a file."""
    # Set tracking URI from environment variable (provided in CI)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        print("Warning: MLFLOW_TRACKING_URI not set, using default local tracking.")

    experiment_name = "DSAI_406_A3_AmrKhalid_Sklearn_DVC"
    print(f"Searching for the latest run in experiment: {experiment_name}")

    try:
        # Get experiment ID by name
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
            
            # Search for the most recent FINISHED run in this experiment
            runs = mlflow.search_runs(
                experiment_ids=[experiment_id],
                filter_string="attributes.status = 'FINISHED'",
                order_by=["attribute.start_time DESC"],
                max_results=1
            )
            
            if not runs.empty:
                latest_run_id = runs.iloc[0].run_id
                print(f"Latest Run ID found: {latest_run_id}")
                
                # Save run ID to a file for subsequent CI steps
                with open("model_info.txt", "w") as f:
                    f.write(latest_run_id)
                print("Run ID saved to model_info.txt")
            else:
                print(f"No runs found for experiment: {experiment_name}")
                sys.exit(1)
        else:
            print(f"Experiment '{experiment_name}' not found.")
            sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    retrieve_latest_run_id()