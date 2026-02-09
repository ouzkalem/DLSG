"""
Assignment1.2: Learning Physics from Data
=========================================

Main entry point for the project.

The main.py file serves as the entry point for the project and walks you through
the steps of data loading, model initialization, training, and testing. The comments
provided in the code will guide you in filling out necessary parts where values or
function implementations are missing. There are additional utility files that handle
the loading and processing of data as well as visualization tools to help you monitor
model performance during training.

How to run:
    python main.py

Files to complete:
    1. model.py     — Define the MLP architectures
    2. train.py     — Implement training and validation loops
    3. evaluate.py  — Complete the experiment runner functions
"""

import torch
import numpy as np

from config import SEED, TRAIN_CONFIG, DATA_CONFIG, MODEL_CONFIG
from data import prepare_data
from model import create_model
from train import train_model, get_optimizer
from evaluate import (
    experiment_optimizers,
    experiment_regularization,
    experiment_dataset_size,
)
from visualize import plot_loss_curves, plot_dataset_exploration, plot_trajectories
from data import generate_dataset, simulate_projectile_trajectory


def main():

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Random Seed: {SEED}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")


    X_raw, y_raw = generate_dataset(
        n_samples=DATA_CONFIG["n_samples_medium"], seed=SEED
    )
    plot_dataset_exploration(X_raw, y_raw)
    plot_trajectories(simulate_projectile_trajectory)


    # Prepare a small dataset for quick testing 
    # In practice, you should also do sanity checks. 
    # See if model converges with a small dataset first, before running the full experiments)
    train_loader, test_loader, stats = prepare_data(
        n_samples=DATA_CONFIG["n_samples_small"],
        batch_size=TRAIN_CONFIG["batch_size"],
    )

    # Create a standard model and optimizer, then train it.
    print("\n" + "=" * 60)
    print("  SANITY CHECK: Quick training on small dataset")
    print("=" * 60)
    
    # 1. Create model
    model = create_model("standard")
    
    # 2. Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG["learning_rate"])
    
    # 3. Create loss criterion
    criterion = torch.nn.MSELoss()
    
    # 4. Train the model
    from train import train_model
    history = train_model(model, train_loader, test_loader, criterion, optimizer, 
                          epochs=TRAIN_CONFIG["epochs"], verbose=True)
    
    # 5. Plot loss curves
    plot_loss_curves(history, title="Sanity Check")
    
    # Debugging assertions
    assert model is not None, "Model was not created"
    assert history is not None, "Training did not return history"
    print(f"\n  Sanity check passed! Final train loss: {history['train_loss'][-1]:.6f}, test loss: {history['test_loss'][-1]:.6f}")

    # Run all experiments
    results_optimizers = experiment_optimizers()
    results_regularization = experiment_regularization()
    results_dataset_size = experiment_dataset_size()

    print("\nCheck the generated .png plots and the summary tables above.")


if __name__ == "__main__":
    main()
