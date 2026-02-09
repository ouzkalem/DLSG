"""
Evaluation & experiment runner module for Assignment 2.
Students must complete the experiment functions.
"""

import torch
import torch.nn as nn
import numpy as np

from config import (SEED, TRAIN_CONFIG, MODEL_CONFIG,
                    OPTIMIZER_EXPERIMENT, REGULARIZATION_EXPERIMENT,
                    DATASET_SIZE_EXPERIMENT, DATA_CONFIG)
from data import prepare_data
from model import create_model
from train import train_model, get_optimizer
from visualize import (plot_loss_curves, plot_experiment_comparison,
                       plot_dataset_size_comparison, print_results_table)


def set_seed(seed=SEED):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_single_experiment(model_type, optimizer_name, train_loader, test_loader,
                          lr, epochs, weight_decay=0.0, momentum=0.9, seed=SEED):
    """
    Train a single model configuration and return the training history.

    Args:
        model_type: "standard" or "dropout"
        optimizer_name: "adam" or "sgd"
        train_loader: Training DataLoader
        test_loader: Test DataLoader
        lr: Learning rate
        epochs: Number of epochs
        weight_decay: L2 regularization strength
        momentum: SGD momentum
        seed: Random seed

    Returns:
        history: dict with 'train_loss' and 'test_loss' lists
        model: The trained model
    """
    set_seed(seed)

    # 1. Create the model
    model = create_model(model_type)
    
    # 2. Create the optimizer
    optimizer = get_optimizer(model, optimizer_name, lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    # 3. Define the loss criterion (MSELoss for regression)
    criterion = nn.MSELoss()
    
    # 4. Train the model
    history = train_model(model, train_loader, test_loader, criterion, optimizer, epochs=epochs, verbose=True)
    
    # Debugging assertions
    assert model is not None, "Model was not created"
    assert isinstance(criterion, nn.MSELoss), "Use nn.MSELoss for regression"
    assert "train_loss" in history and "test_loss" in history, "History should contain 'train_loss' and 'test_loss'"
    assert len(history["train_loss"]) == epochs, f"Expected {epochs} epochs in history, got {len(history['train_loss'])}"
    
    # 5. Return history and model
    return history, model



def experiment_optimizers():
    """
    Compare Adam vs SGD+Momentum on the same dataset.

    Steps:
        1. Prepare a medium-sized dataset
        2. Train with Adam (lr from config)
        3. Train with SGD+Momentum (lr from config)
        4. Plot and compare results
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT 1: Optimizer Comparison (Adam vs SGD+Momentum)")
    print("=" * 60)

    epochs = OPTIMIZER_EXPERIMENT["epochs"]

    # 1. Prepare a medium-sized dataset
    train_loader, test_loader, stats = prepare_data(
        n_samples=DATA_CONFIG["n_samples_medium"],
        batch_size=TRAIN_CONFIG["batch_size"]
    )
    
    # 2. Run experiment with Adam
    print("\n  Training with Adam...")
    adam_history, adam_model = run_single_experiment(
        model_type="standard",
        optimizer_name="adam",
        train_loader=train_loader,
        test_loader=test_loader,
        lr=TRAIN_CONFIG["learning_rate"],
        epochs=epochs
    )
    
    # 3. Run experiment with SGD
    print("\n  Training with SGD+Momentum...")
    sgd_history, sgd_model = run_single_experiment(
        model_type="standard",
        optimizer_name="sgd",
        train_loader=train_loader,
        test_loader=test_loader,
        lr=TRAIN_CONFIG["sgd_learning_rate"],
        momentum=TRAIN_CONFIG["sgd_momentum"],
        epochs=epochs
    )
    
    # 4. Store results
    results = {
        "Adam": adam_history,
        "SGD+Momentum": sgd_history,
    }
    
    # 5. Plot experiment comparisons
    plot_experiment_comparison(results, title="Optimizer Comparison")
    
    # 6. Print results table
    print_results_table(results)
    
    # 7. Return results
    return results


def experiment_regularization():
    """
    Compare No Regularization vs Dropout vs Weight Decay.

    Steps:
        1. Prepare a medium-sized dataset
        2. Train with no regularization (standard model, weight_decay=0)
        3. Train with Dropout (dropout model, weight_decay=0)
        4. Train with Weight Decay (standard model, weight_decay > 0)
        5. Plot and compare results
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT 2: Regularization Comparison")
    print("=" * 60)

    epochs = REGULARIZATION_EXPERIMENT["epochs"]

    # 1. Prepare a medium-sized dataset
    train_loader, test_loader, stats = prepare_data(
        n_samples=DATA_CONFIG["n_samples_medium"],
        batch_size=TRAIN_CONFIG["batch_size"]
    )
    
    # 2a. Train with No Regularization
    print("\n  Training with No Regularization...")
    history_none, model_none = run_single_experiment(
        model_type="standard",
        optimizer_name="adam",
        train_loader=train_loader,
        test_loader=test_loader,
        lr=TRAIN_CONFIG["learning_rate"],
        weight_decay=0.0,
        epochs=epochs
    )
    
    # 2b. Train with Dropout
    print("\n  Training with Dropout...")
    history_dropout, model_dropout = run_single_experiment(
        model_type="dropout",
        optimizer_name="adam",
        train_loader=train_loader,
        test_loader=test_loader,
        lr=TRAIN_CONFIG["learning_rate"],
        weight_decay=0.0,
        epochs=epochs
    )
    
    # 2c. Train with Weight Decay
    print("\n  Training with Weight Decay...")
    history_wd, model_wd = run_single_experiment(
        model_type="standard",
        optimizer_name="adam",
        train_loader=train_loader,
        test_loader=test_loader,
        lr=TRAIN_CONFIG["learning_rate"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
        epochs=epochs
    )
    
    # 3. Store results
    results = {
        "No Regularization": history_none,
        "Dropout": history_dropout,
        "Weight Decay": history_wd,
    }
    
    # 5. Plot experiment comparisons
    plot_experiment_comparison(results, title="Regularization Comparison")
    
    # 6. Print results table
    print_results_table(results)
    
    # 7. Return results
    return results



def experiment_dataset_size():
    """
    Compare model performance with Small vs Medium vs Large datasets.

    Steps:
        1. Generate 3 datasets of different sizes
        2. Train the same model architecture on each
        3. Plot and compare results
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT 3: Dataset Size Comparison")
    print("=" * 60)

    epochs = DATASET_SIZE_EXPERIMENT["epochs"]

    # 1. Define size mappings
    sizes = {
        "Small (500)": DATA_CONFIG["n_samples_small"],
        "Medium (2000)": DATA_CONFIG["n_samples_medium"],
        "Large (10000)": DATA_CONFIG["n_samples_large"],
    }
    
    results = {}
    
    # 2. Train on each dataset size
    for size_name, n_samples in sizes.items():
        print(f"\n  Training with {size_name} samples...")
        
        # Prepare data for this size
        train_loader, test_loader, stats = prepare_data(
            n_samples=n_samples,
            batch_size=TRAIN_CONFIG["batch_size"]
        )
        
        # Train model
        history, model = run_single_experiment(
            model_type="standard",
            optimizer_name="adam",
            train_loader=train_loader,
            test_loader=test_loader,
            lr=TRAIN_CONFIG["learning_rate"],
            epochs=epochs
        )
        
        results[size_name] = history
    
    # 5. Plot experiment comparisons
    plot_dataset_size_comparison(results, title="Dataset Size Comparison")
    
    # 6. Print results table
    print_results_table(results)
    
    # 7. Return results
    return results
