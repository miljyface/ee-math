import torch
import torch.nn as nn
import json
from pathlib import Path
from methods.trainingutils.landscaper import LossLandscapeAnalyzer
from neuralnetwork.architecture import MNISTNet
from neuralnetwork.trainer import train_model
from torch.utils.data import DataLoader
from methods.trainingutils.tensorprep import prep_tensors

BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda")
OUTPUT_DIR = Path("/Users/guanrong/Desktop/EE_analyse/output")
JSON_DIR = OUTPUT_DIR / "json"
WEIGHTS_DIR = OUTPUT_DIR / "weights"
JSON_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

def find_convergence_epoch(val_acc_list, threshold=98.0):
    for epoch_idx, acc in enumerate(val_acc_list):
        if acc >= threshold:
            return epoch_idx + 1
    return -1  # Not converged

def main(norm_type: str) -> dict:
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
    
    print(f"[{norm_type.upper()}] Computing loss landscape")
    
    # Compute loss landscape
    analyzer = LossLandscapeAnalyzer(model, nn.CrossEntropyLoss(), test_loader, DEVICE)
    directions = analyzer.get_filter_normalized_directions(num_directions=2)
    alpha_grid, beta_grid, loss_grid = analyzer.compute_loss_2d(
        directions[0], directions[1],
        alpha_range=(-0.5, 0.5),
        beta_range=(-0.5, 0.5),
        num_points=20
    )
    
    print(f"[{norm_type.upper()}] Computing Hessian eigenvalues")
    
    # Compute Hessian eigenvalues and related metrics
    hessian_results = analyzer.compute_hessian_eigenvalues(num_eigenvalues=5)
    
    # Calculate convergence epoch
    convergence_epoch = find_convergence_epoch(history['val_acc'], threshold=98.0)
    if convergence_epoch == -1:
        convergence_epoch_str = "Not converged"
        convergence_epoch_num = NUM_EPOCHS
    else:
        convergence_epoch_str = str(convergence_epoch)
        convergence_epoch_num = convergence_epoch
    
    # Compile results
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
        },
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
        },
        'hessian_analysis': {
            'top_eigenvalues': hessian_results['top_eigenvalues'].tolist(),
            'max_eigenvalue': float(hessian_results['max_eigenvalue'][0]),
            'min_eigenvalue': float(hessian_results['min_eigenvalue'][0]),
            'trace': float(hessian_results['trace'][0]),
            'condition_number': float(hessian_results.get('condition_number', 1.0)),
            'density_eigen': hessian_results['density_eigen'].tolist() if 'density_eigen' in hessian_results else [],
            'density_weight': hessian_results['density_weight'].tolist() if 'density_weight' in hessian_results else []
        }
    }
    
    # Save model weights
    weight_path = WEIGHTS_DIR / f"model_{norm_type}.pth"
    torch.save(model.state_dict(), weight_path)
    
    # Save results JSON
    json_path = JSON_DIR / f"results_{norm_type}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print(f"\n[{norm_type.upper()}] Training complete")
    print(f" Final val accuracy: {results['final_metrics']['val_acc']:.2f}%")
    print(f" Max val accuracy: {results['final_metrics']['max_val_acc']:.2f}%")
    print(f" Final val loss: {results['final_metrics']['val_loss']:.4f}")
    print(f" Min val loss: {results['final_metrics']['min_val_loss']:.4f}")
    print(f" Convergence epoch (98% acc): {convergence_epoch_str}")
    print(f" Loss landscape grid: {alpha_grid.shape[0]}x{alpha_grid.shape[0]}")
    print(f" Max eigenvalue: {results['hessian_analysis']['max_eigenvalue']:.6f}")
    print(f" Condition number: {results['hessian_analysis']['condition_number']:.2f}")
    print(f" Weights saved: {weight_path}")
    print(f" Results saved: {json_path}\n")
    
    return results

# test = ['dyt', 'none', 'batch', 'layer', 'group', 'rms', 'piecewise']
main('layer')