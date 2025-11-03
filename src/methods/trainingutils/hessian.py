import torch
import numpy as np

def compute_hessian_eigenvalues(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device = torch.device("cpu"),
    k: int = 5,
    max_iter: int = 80,
    tol: float = 1e-6,
    verbose: bool = True,
) -> dict:
    """
    Computes the top-k eigenvalues of the Hessian of the loss w.r.t. model parameters
    using power iteration. Uses a single batch from dataloader.
    """
    model.eval()
    model.to(device)
    criterion.to(device)
    
    # Use a single batch for Hessian calculation
    inputs, targets = next(iter(dataloader))
    inputs, targets = inputs.to(device), targets.to(device)
    
    # Closure to evaluate loss
    def loss_fn():
        outputs = model(inputs)
        return criterion(outputs, targets)
    
    # Collect all parameters
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    if verbose:
        print(f"[Hessian] Model has {n_params:,} parameters")
    
    # Helper: Flatten gradients
    def _flatten_grads(grad_tensors):
        return torch.cat([g.contiguous().view(-1) for g in grad_tensors if g is not None])
    
    # Helper: Hessian-vector product (autograd, Pearlmutter)
    def hvp(vector):
        loss = loss_fn()
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
        grad_vec = _flatten_grads(grads)
        grad_dot = torch.dot(grad_vec, vector)
        hv = torch.autograd.grad(grad_dot, params, retain_graph=True)
        return _flatten_grads(hv).detach()
    
    eigvals = []
    eigvecs = []
    n = n_params
    residual_vec = torch.randn(n, device=device)
    residual_vec = residual_vec / residual_vec.norm()
    
    # Lanczos/power iteration for top-k
    for eig_idx in range(k):
        vec = torch.randn(n, device=device)
        vec = vec / vec.norm()
        prev_eigenvalue = None
        
        eigenvalue = []
        for it in range(max_iter):
            # Orthogonalize to previously found eigenvectors
            for prev_vec in eigvecs:
                proj = torch.dot(vec, prev_vec)
                vec = vec - proj * prev_vec
            vec = vec / (vec.norm() + 1e-8)
            
            # Hessian-vector product
            hv = hvp(vec)
            eigenvalue = torch.dot(vec, hv).item()
            
            if verbose:
                print(f"[Hessian] Eigen {eig_idx+1}, iter {it+1}: eigenvalue={eigenvalue:.6f}")
            
            # Check for convergence
            if prev_eigenvalue is not None and abs(eigenvalue - prev_eigenvalue) < tol:
                break
            prev_eigenvalue = eigenvalue
            
            # Power iteration step
            vec = hv / (hv.norm() + 1e-8)
        try:
            eigvals.append(eigenvalue)
        except Exception as e:
            print("fucking unbound lmao")
        eigvecs.append(vec.detach().clone())

    return {
        'top_eigenvalues': np.array(eigvals),
        'top_eigenvectors': [v.cpu().numpy() for v in eigvecs],
    }
