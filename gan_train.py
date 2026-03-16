import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import mlflow

mlflow.set_experiment("DSAI_406_A3_AmrKhalid")

def parse_args():
    """Parse CLI options controlling training, reproducibility, and debug shortcuts."""
    parser = argparse.ArgumentParser(description="Train a simple reproducible GAN on Fashion-MNIST.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--dataset-root", type=str, default="dataset/")
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 for strongest reproducibility.")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional cap for quick tests.")
    parser.add_argument("--no-show", action="store_true", help="Skip matplotlib window.")
    parser.add_argument(
        "--allow-nondeterministic",
        action="store_true",
        help="Disable deterministic algorithm enforcement (can be faster).",
    )
    return parser.parse_args()


def set_reproducibility(seed, deterministic=True):
    """Seed all RNG sources and optionally enforce deterministic PyTorch kernels."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Needed by some CUDA kernels when deterministic mode is enforced.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def seed_worker(worker_id):
    """Initialize each DataLoader worker with a reproducible, unique RNG state."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Discriminator(nn.Module):
    """Binary classifier that scores real-vs-fake images."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Flatten image batches and return real/fake probabilities."""
        return self.model(x.view(x.size(0), -1))


class Generator(nn.Module):
    """MLP generator that maps latent vectors to 28x28 grayscale images."""

    def __init__(self, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Tanh(),
        )

    def forward(self, z):
        """Decode latent noise into image-shaped tensors."""
        return self.model(z).view(-1, 1, 28, 28)


def build_loader(args, seed):
    """Create a deterministically shuffled Fashion-MNIST DataLoader."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = datasets.FashionMNIST(
        root=args.dataset_root,
        train=True,
        transform=transform,
        download=True,
    )

    data_gen = torch.Generator()
    data_gen.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=data_gen,
        persistent_workers=args.num_workers > 0,
    )


def show_fake_images(gen, latent_dim, device, seed, num_images=16, show=True):
    """Generate a fixed-noise preview grid so visual checks are comparable across runs."""
    gen.eval()
    with torch.no_grad():
        eval_gen = torch.Generator()
        eval_gen.manual_seed(seed + 1)
        noise = torch.randn(num_images, latent_dim, generator=eval_gen).to(device)
        fakes = gen(noise).cpu()
        grid = torchvision.utils.make_grid(fakes, nrow=4, normalize=True)

    if show:
        plt.imshow(grid.permute(1, 2, 0))
        plt.title("Generated Fashion Items")
        plt.show()


def main():
    """Train GAN models and optionally display deterministic sample images."""
    args = parse_args()
    deterministic = not args.allow_nondeterministic
    set_reproducibility(args.seed, deterministic=deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Running with seed={args.seed}, deterministic={deterministic}, "
        f"device={device}, workers={args.num_workers}"
    )

    loader = build_loader(args, args.seed)

    disc = Discriminator().to(device)
    gen = Generator(args.latent_dim).to(device)
    opt_disc = optim.Adam(disc.parameters(), lr=args.lr)
    opt_gen = optim.Adam(gen.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    train_noise_gen = torch.Generator()
    train_noise_gen.manual_seed(args.seed + 123)
    with mlflow.start_run() as run:
        mlflow.set_tag("student_name", "Amr Khalid")
        mlflow.set_tag("student_id", "21102490")
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("deterministic", deterministic)
        mlflow.log_param("device", device)
        mlflow.log_param("workers", args.num_workers)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("latent_dim", args.latent_dim)
        mlflow.log_param("lr", args.lr)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("max_batches", args.max_batches)
        mlflow.log_param("no_show", args.no_show)
        mlflow.log_param("allow_nondeterministic", args.allow_nondeterministic)

        for epoch in range(args.epochs):
            lossD = torch.tensor(float("nan"), device=device)
            lossG = torch.tensor(float("nan"), device=device)

            for batch_idx, batch in enumerate(loader):
                real = batch[0]
                if not isinstance(real, torch.Tensor):
                    raise TypeError("Expected tensor images in dataloader batch")
                real = real.to(device)
                batch_size_curr = real.size(0)

                noise = torch.randn(batch_size_curr, args.latent_dim, generator=train_noise_gen).to(device)
                fake = gen(noise)

                # 1) Update discriminator on real and detached fake samples.
                disc_real = disc(real).view(-1)
                lossD_real = criterion(disc_real, torch.ones_like(disc_real))
                disc_fake = disc(fake.detach()).view(-1)
                lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
                lossD = (lossD_real + lossD_fake) / 2

                disc.zero_grad()
                lossD.backward()
                opt_disc.step()

                # 2) Update generator so discriminator labels fakes as real.
                output = disc(fake).view(-1)
                lossG = criterion(output, torch.ones_like(output))

                gen.zero_grad()
                lossG.backward()
                opt_gen.step()

                if args.max_batches is not None and (batch_idx + 1) >= args.max_batches:
                    break

            print(f"Epoch [{epoch + 1}/{args.epochs}] Loss D: {lossD:.4f}, Loss G: {lossG:.4f}")
            mlflow.log_metric("lossD", lossD.item(), step=epoch)
            mlflow.log_metric("lossG", lossG.item(), step=epoch)

        mlflow.pytorch.log_model(gen, "generator",
                                registered_model_name="generator",
                                pip_requirements=["torch", "torchvision", "numpy", "matplotlib", "mlflow"])
        mlflow.pytorch.log_model(disc, "discriminator",
                                registered_model_name="discriminator",
                                pip_requirements=["torch", "torchvision", "numpy", "matplotlib", "mlflow"])
    show_fake_images(gen, args.latent_dim, device, args.seed, show=not args.no_show)


if __name__ == "__main__":
    main()
