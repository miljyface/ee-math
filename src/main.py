import torch
import torch.nn as nn
import json
import numpy as np
import time
from pathlib import Path

from methods.trainingutils.landscaper import LossLandscapeAnalyzer
from neuralnetwork.architecture import MNISTNet
from neuralnetwork.trainer import train_model
from torch.utils.data import DataLoader
from methods.trainingutils.tensorprep import prep_tensors
from methods.trainingutils.hessian import compute_hessian_eigenvalues


BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("mps")
OUTPUT_DIR = Path("/Users/guanrong/Desktop/ee_shit/ee-math/output")
JSON_DIR = OUTPUT_DIR / "json"
WEIGHTS_DIR = OUTPUT_DIR / "weights"
JSON_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


def find_convergence_epoch(val_acc_list, threshold=98.0):
    """Find the first epoch where validation accuracy reaches threshold."""
    for epoch_idx, acc in enumerate(val_acc_list):
        if acc >= threshold:
            return epoch_idx + 1
    return -1  # Not converged


def train_phase(norm_type: str) -> dict:
    """
    Training phase: train model and save weights + basic training history.
    
    Args:
        norm_type: Type of normalization to use
        
    Returns:
        dict: Training results including history and metadata
    """
    train_dataset, test_dataset = prep_tensors()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = MNISTNet(norm_type=norm_type)
    num_params = sum(p.numel() for p in model.parameters())
    
    print(f"[{norm_type.upper()}] Training started")
    print(f" Parameters: {num_params:,}")
    print(f" Epochs: {NUM_EPOCHS}")
    print(f" Batch size: {BATCH_SIZE}")
    print(f" Learning rate: {LEARNING_RATE}")
    
    # Train
    history = train_model(
        model, train_loader, test_loader,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        device=DEVICE
    )
    
    # Calculate convergence epoch
    convergence_epoch = find_convergence_epoch(history['val_acc'], threshold=98.0)
    if convergence_epoch == -1:
        convergence_epoch_str = "Not converged"
        convergence_epoch_num = NUM_EPOCHS
    else:
        convergence_epoch_str = str(convergence_epoch)
        convergence_epoch_num = convergence_epoch
    
    # Compile training results
    results = {
        'metadata': {
            'normalization': norm_type,
            'num_parameters': num_params,
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'device': str(DEVICE)
        },
        'training_history': {
            'train_loss': history['train_loss'],
            'train_acc': history['train_acc'],
            'val_loss': history['val_loss'],
            'val_acc': history['val_acc'],
            'learning_rate': history['lr']
        },
        'final_metrics': {
            'train_loss': history['train_loss'][-1],
            'train_acc': history['train_acc'][-1],
            'val_loss': history['val_loss'][-1],
            'val_acc': history['val_acc'][-1],
            'convergence_epoch': convergence_epoch_num,
            'convergence_status': convergence_epoch_str,
            'max_val_acc': max(history['val_acc']),
            'min_val_loss': min(history['val_loss'])
        }
    }
    
    # Save model weights
    weight_path = WEIGHTS_DIR / f"model_{norm_type}.pth"
    torch.save(model.state_dict(), weight_path)
    
    # Save training results JSON
    json_path = JSON_DIR / f"training_{norm_type}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[{norm_type.upper()}] Training complete")
    print(f" Final val accuracy: {results['final_metrics']['val_acc']:.2f}%")
    print(f" Max val accuracy: {results['final_metrics']['max_val_acc']:.2f}%")
    print(f" Convergence epoch (98% acc): {convergence_epoch_str}")
    print(f" Weights saved: {weight_path}")
    print(f" Training results saved: {json_path}\n")
    
    return results


def analyze_loss_landscape(norm_type: str, alpha_range=(-0.5, 0.5), beta_range=(-0.5, 0.5), num_points=20) -> dict:
    """
    Analyze loss landscape for a trained model.
    
    Args:
        norm_type: Type of normalization used in model
        alpha_range: Range for first direction
        beta_range: Range for second direction
        num_points: Number of grid points per dimension
        
    Returns:
        dict: Loss landscape analysis results
    """
    # Load model
    model = MNISTNet(norm_type=norm_type)
    weight_path = WEIGHTS_DIR / f"model_{norm_type}.pth"
    
    if not weight_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weight_path}")
    
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.to(DEVICE)
    
    # Load test data
    _, test_dataset = prep_tensors()
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"[{norm_type.upper()}] Computing loss landscape")
    
    # Compute loss landscape
    analyzer = LossLandscapeAnalyzer(model, nn.CrossEntropyLoss(), test_loader, DEVICE)
    directions = analyzer.get_filter_normalized_directions(num_directions=2)
    alpha_grid, beta_grid, loss_grid = analyzer.compute_loss_2d(
        directions[0], directions[1],
        alpha_range=alpha_range,
        beta_range=beta_range,
        num_points=num_points
    )
    
    # Compile landscape results
    landscape_results = {
        'loss_landscape': {
            'alpha_grid': alpha_grid.tolist(),
            'beta_grid': beta_grid.tolist(),
            'loss_grid': loss_grid.tolist(),
            'grid_size': alpha_grid.shape[0],
            'alpha_range': [float(alpha_grid.min()), float(alpha_grid.max())],
            'beta_range': [float(beta_grid.min()), float(beta_grid.max())],
            'loss_min': float(loss_grid.min()),
            'loss_max': float(loss_grid.max()),
            'loss_mean': float(loss_grid.mean()),
            'loss_std': float(loss_grid.std()),
            'loss_variance': float(loss_grid.var())
        }
    }
    
    # Save landscape results
    json_path = JSON_DIR / f"landscape_{norm_type}.json"
    with open(json_path, 'w') as f:
        json.dump(landscape_results, f, indent=2)
    
    print(f" Loss landscape grid: {alpha_grid.shape[0]}x{alpha_grid.shape[0]}")
    print(f" Loss range: [{loss_grid.min():.4f}, {loss_grid.max():.4f}]")
    print(f" Landscape results saved: {json_path}\n")
    
    return landscape_results


def analyze_hessian(norm_type: str, num_eigenvalues=5) -> dict:
    """
    Analyze Hessian eigenvalues for a trained model.
    
    Args:
        norm_type: Type of normalization used in model
        num_eigenvalues: Number of top eigenvalues to compute
        
    Returns:
        dict: Hessian analysis results
    """
    # Load model
    model = MNISTNet(norm_type=norm_type)
    weight_path = WEIGHTS_DIR / f"model_{norm_type}.pth"
    
    if not weight_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weight_path}")
    
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.to(DEVICE)
    
    # Load test data
    _, test_dataset = prep_tensors()
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"[{norm_type.upper()}] Computing Hessian eigenvalues")

    start_time = time.time()
    hessian_results = compute_hessian_eigenvalues(
        model, nn.CrossEntropyLoss(), test_loader, device=torch.device("cpu"), k=5
    )

    elapsed = time.time() - start_time
    top_eigenvalues = hessian_results['top_eigenvalues']
    max_eigenvalue = float(np.max(top_eigenvalues))
    min_eigenvalue = float(np.min(top_eigenvalues))
    condition_number = float(abs(max_eigenvalue / (min_eigenvalue if min_eigenvalue != 0 else 1e-12)))

    # Compile hessian results
    hessian_analysis = {
        'hessian_analysis': {
            'top_eigenvalues': top_eigenvalues.tolist(),
            'max_eigenvalue': max_eigenvalue,
            'min_eigenvalue': min_eigenvalue,
            'condition_number': condition_number,
            'computation_time': elapsed
        }
    }
    
    # Save hessian results
    json_path = JSON_DIR / f"hessian_{norm_type}.json"
    with open(json_path, 'w') as f:
        json.dump(hessian_analysis, f, indent=2)
    
    print(f" Max eigenvalue: {hessian_analysis['hessian_analysis']['max_eigenvalue']:.6f}")
    print(f" Min eigenvalue: {hessian_analysis['hessian_analysis']['min_eigenvalue']:.6f}")
    print(f" Condition number: {hessian_analysis['hessian_analysis']['condition_number']:.2f}")
    print(f" Hessian results saved: {json_path}\n")
    
    return hessian_analysis


def combine_results(norm_type: str) -> dict:
    # Load all individual result files
    training_path = JSON_DIR / f"training_{norm_type}.json"
    landscape_path = JSON_DIR / f"landscape_{norm_type}.json"
    hessian_path = JSON_DIR / f"hessian_{norm_type}.json"
    
    combined = {}
    
    if training_path.exists():
        with open(training_path, 'r') as f:
            combined.update(json.load(f))
    
    if landscape_path.exists():
        with open(landscape_path, 'r') as f:
            combined.update(json.load(f))
    
    if hessian_path.exists():
        with open(hessian_path, 'r') as f:
            combined.update(json.load(f))
    
    # Save combined results
    combined_path = JSON_DIR / f"results_{norm_type}.json"
    with open(combined_path, 'w') as f:
        json.dump(combined, f, indent=2)
    
    print(f"[{norm_type.upper()}] Combined results saved: {combined_path}")
    
    return combined

if __name__ == "__main__":
    train_phase('none')
    analyze_loss_landscape('none')
    analyze_hessian('none')
    
    # Option 4: Run specific analysis
    # analyze_loss_landscape('layer')
    # analyze_hessian('layer')
    
    # Option 5: Batch processing
    # norm_types = ['dyt', 'none', 'batch', 'layer', 'group', 'rms', 'piecewise']
    # for norm_type in norm_types:
    #     run_full_pipeline(norm_type)
