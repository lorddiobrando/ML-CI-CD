# ML-CI-CD

A reproducible **Generative Adversarial Network (GAN)** training project that generates synthetic fashion items using the Fashion-MNIST dataset. The project integrates **MLflow** for experiment tracking and model registry, **Docker** for containerization, and **GitHub Actions** for CI/CD automation.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [Docker](#docker)
- [CI/CD Pipeline](#cicd-pipeline)
- [Reproducibility](#reproducibility)
- [License](#license)

---

## Project Overview

This project trains a simple MLP-based GAN on the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset. The model learns to generate realistic 28×28 grayscale images of clothing items. Key highlights:

- **Reproducible training**: strict seed control and deterministic PyTorch algorithms
- **Experiment tracking**: all hyperparameters, metrics, and trained models are logged to MLflow
- **Containerized**: ready-to-run Docker image for consistent environments
- **CI/CD**: automated linting and validation via GitHub Actions on every push and pull request

---

## Architecture

### Generator
Converts a latent vector into a 28×28 image:

```
Linear(latent_dim → 256) → ReLU → Linear(256 → 784) → Tanh → reshape [1, 28, 28]
```

### Discriminator
Classifies images as real or fake:

```
Linear(784 → 256) → LeakyReLU(0.2) → Dropout(0.3) → Linear(256 → 1) → Sigmoid
```

Both models are trained with **Binary Cross-Entropy (BCELoss)** and the **Adam** optimizer.

---

## Requirements

- Python 3.10
- [PyTorch](https://pytorch.org/) 2.5.1 (CPU)
- TorchVision 0.20.1
- MLflow 3.10.1
- Matplotlib 3.10.8

All dependencies are listed in [`requirements.txt`](requirements.txt).

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

## Usage

Run the training script with default hyperparameters:

```bash
python gan_train.py
```

### Available Arguments

| Argument | Default | Description |
|---|---|---|
| `--seed` | `42` | Random seed for reproducibility |
| `--epochs` | `20` | Number of training epochs |
| `--batch-size` | `128` | Mini-batch size |
| `--latent-dim` | `64` | Size of the generator's latent vector |
| `--lr` | `0.0002` | Learning rate for both optimizers |
| `--dataset-root` | `dataset/` | Path to store/load the Fashion-MNIST data |
| `--num-workers` | `0` | Number of DataLoader worker processes |
| `--max-batches` | *(none)* | Limit batches per epoch (useful for quick tests) |
| `--no-show` | *(flag)* | Skip matplotlib image preview |
| `--allow-nondeterministic` | *(flag)* | Disable deterministic algorithms for speed |

### Example: Quick Test Run

```bash
python gan_train.py --epochs 2 --max-batches 10 --no-show
```

---

## MLflow Experiment Tracking

Training automatically logs to MLflow under the experiment **`DSAI_406_A3_AmrKhalid`**.

**Logged parameters:** `seed`, `device`, `batch_size`, `latent_dim`, `lr`, `epochs`

**Logged metrics (per epoch):** `lossD` (discriminator loss), `lossG` (generator loss)

**Registered models:** `generator`, `discriminator`

To launch the MLflow UI and inspect results:

```bash
mlflow ui
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Docker

Build and run the training container:

```bash
# Build the image
docker build -t ml-ci-cd .

# Run training
docker run --rm ml-ci-cd
```

Pass custom arguments:

```bash
docker run --rm ml-ci-cd python gan_train.py --epochs 5 --no-show
```

The image is based on `python:3.10-slim` and installs all dependencies from `requirements.txt`.

---

## CI/CD Pipeline

The GitHub Actions workflow ([`.github/workflows/ml-pipeline.yml`](.github/workflows/ml-pipeline.yml)) runs automatically on:

- **Pushes** to any branch except `main`
- **Pull requests**

### Pipeline Steps

1. **Checkout** – fetches the latest code
2. **Set up Python 3.10**
3. **Install dependencies** – `pip install -r requirements.txt`
4. **Lint with Flake8**
   - Fails on critical syntax/name errors (`E9`, `F63`, `F7`, `F82`)
   - Warns on style issues: complexity ≤ 10, max line length 127 (non-blocking)
5. **Model dry test** – verifies PyTorch is importable and the environment is healthy
6. **Upload README artifact** – archives `README.md` as a build artifact

---

## Reproducibility

The project enforces fully deterministic training:

- `PYTHONHASHSEED` and `CUBLAS_WORKSPACE_CONFIG` environment variables are set at startup
- PyTorch and CUDA RNG states are seeded via `--seed`
- NumPy RNG is seeded consistently
- `torch.use_deterministic_algorithms(True)` is enabled by default
- DataLoader workers receive unique, deterministic seeds

Use `--allow-nondeterministic` to trade reproducibility for speed.

---

## License

This project is licensed under the [MIT License](LICENSE).