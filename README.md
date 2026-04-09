# ML-CI-CD

A reproducible **MLP Classifier** training project on the scikit-learn Digits dataset. The project integrates **DVC** for data and pipeline versioning, **MLflow** (hosted on [DagsHub](https://dagshub.com)) for experiment tracking and model registry, **Docker** for containerization, and **GitHub Actions** for CI/CD automation.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [DVC Data Versioning](#dvc-data-versioning)
- [Usage](#usage)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [Docker](#docker)
- [CI/CD Pipeline](#cicd-pipeline)
- [Reproducibility](#reproducibility)
- [License](#license)

---

## Project Overview

This project trains a scikit-learn **MLP Classifier** on the [Digits dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html). The dataset is version-controlled with **DVC** and optionally synced via Google Drive. Key highlights:

- **Reproducible training**: strict seed control across Python, NumPy, and scikit-learn
- **Data versioning**: dataset tracked and reproduced with DVC pipelines
- **Experiment tracking**: all hyperparameters, metrics, and trained models are logged to MLflow on DagsHub
- **Containerized**: Docker image built automatically in CI, incorporating the trained model
- **CI/CD**: automated lint, DVC pipeline execution, accuracy gate, and deployment via GitHub Actions

---

## Architecture

The model is a **Multi-Layer Perceptron (MLP) Classifier** from scikit-learn:

```
Input (64 features) → Hidden Layer(s) → Output (10 digit classes)
```

- Configurable hidden layer sizes (default: `100`)
- Trained with scikit-learn's `MLPClassifier` using `random_state` for reproducibility
- Evaluated with accuracy score and classification report

---

## Requirements

- Python 3.10
- [MLflow](https://mlflow.org/)
- Matplotlib
- scikit-learn
- pandas
- DVC + dvc-gdrive
- boto3
- dagshub

All dependencies with pinned versions are listed in [`requirements.txt`](requirements.txt).

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/lorddiobrando/ML-CI-CD.git
   cd ML-CI-CD
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Project Structure

| File | Description |
|---|---|
| `train.py` | Main training script: loads DVC-tracked data, trains MLP, logs to MLflow |
| `save_data.py` | Prepares the Digits dataset and saves it as `data/digits.csv` |
| `check_accuracy.py` | Fetches a run from MLflow and validates accuracy against a threshold |
| `deploy_model.py` | Registers the trained model in the MLflow Model Registry |
| `download_model.py` | Downloads the registered model from DagsHub at container startup |
| `retrieve_run_id.py` | Retrieves the latest finished MLflow run ID and saves it to `model_info.txt` |
| `gan_train.py` | Legacy standalone GAN training script (Fashion-MNIST); not part of the main DVC pipeline |
| `dvc.yaml` | DVC pipeline definition (`prepare` → `train`) |
| `Dockerfile` | Container definition: installs deps, copies app, runs `train.py` |

---

## DVC Data Versioning

The project uses **DVC** to version the dataset and define a reproducible training pipeline.

### Pipeline Stages (`dvc.yaml`)

1. **prepare** – runs `save_data.py` to produce `data/digits.csv`
2. **train** – runs `train.py --no-show` using the prepared CSV

### Reproducing the Pipeline

```bash
# Pull tracked data (requires Google Drive credentials)
dvc pull

# Re-run any outdated pipeline stages
dvc repro
```

The `GDRIVE_CREDENTIALS_DATA` secret is used in CI to authenticate DVC with Google Drive.

---

## Usage

Run the training script directly:

```bash
python train.py
```

### Available Arguments

| Argument | Default | Description |
|---|---|---|
| `--seed` | `42` | Random seed for reproducibility |
| `--hidden-layer-sizes` | `100` | Hidden layer sizes (comma-separated, e.g. `100,50`) |
| `--max-iter` | `200` | Maximum number of training iterations |
| `--no-show` | *(flag)* | Skip matplotlib plots |
| `--data-path` | `data/digits.csv` | Path to the DVC-tracked CSV |

### Example

```bash
python train.py --hidden-layer-sizes 100,50 --max-iter 300 --no-show
```

---

## MLflow Experiment Tracking

Training automatically logs to MLflow under the experiment **`DSAI_406_A3_AmrKhalid_Sklearn_DVC`**, hosted on [DagsHub](https://dagshub.com/lorddiobrando/ML-CI-CD).

**Logged parameters:** `seed`, `hidden_layer_sizes`, `max_iter`, `data_path`

**Logged metrics:** `accuracy`

**Registered model:** `mlp_digits_dvc`

Set the following environment variables to point to DagsHub:

```bash
export MLFLOW_TRACKING_URI=<dagshub_tracking_uri>
export MLFLOW_TRACKING_USERNAME=<username>
export MLFLOW_TRACKING_PASSWORD=<token>
```

To launch a local MLflow UI instead:

```bash
mlflow ui
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Docker

The Docker image accepts the `RUN_ID` build argument to embed the trained model's run reference:

```bash
# Build the image (with a specific MLflow run ID)
docker build --build-arg RUN_ID=<run_id> -t ml-ci-cd .

# Run the container
docker run --rm ml-ci-cd
```

The default command runs `python train.py`. The image is based on `python:3.10-slim` and installs all dependencies from `requirements.txt`.

---

## CI/CD Pipeline

The GitHub Actions workflow ([`.github/workflows/ml-pipeline.yml`](.github/workflows/ml-pipeline.yml)) runs automatically on:

- **Pushes** to any branch except `main`
- **Pull requests**

### Job 1: `validate`

1. **Checkout** – fetches the latest code
2. **Set up Python 3.10**
3. **Install dependencies** – `pip install -r requirements.txt`
4. **DVC Pull** – pulls tracked data from Google Drive (skipped if secret is not set)
5. **Lint with Flake8** – fails on critical syntax/name errors (`E9`, `F63`, `F7`, `F82`)
6. **Run DVC Pipeline** – executes `dvc repro` to train the model, logged to MLflow on DagsHub
7. **Save Model Run** – runs `retrieve_run_id.py` to capture the latest run ID into `model_info.txt`
8. **Upload artifact** – archives `model_info.txt` for the deploy job

### Job 2: `deploy` (requires `validate`)

1. **Checkout** – fetches the latest code
2. **Set up Python 3.10**
3. **Install dependencies**
4. **Download artifact** – retrieves `model_info.txt` from the validate job
5. **Read Run ID** – exports `RUN_ID` from `model_info.txt`
6. **Check Accuracy** – runs `check_accuracy.py` to verify accuracy ≥ 0.85 (fails pipeline if not)
7. **Containerize** – downloads model via `download_model.py` and builds the Docker image
8. **Deploy** – registers the model in the MLflow Model Registry via `deploy_model.py`

---

## Reproducibility

The project enforces deterministic training:

- `PYTHONHASHSEED` environment variable is set at startup
- Python `random`, NumPy, and scikit-learn all receive the same seed via `--seed`
- DVC pipeline locks dependency hashes in `dvc.lock` to ensure data consistency across runs

---

## License

This project is licensed under the [MIT License](LICENSE).