import torch
import numpy as np
from typing import Dict


def compute_hessian_eigenvalues(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device = torch.device("mps"),
    k: int = 5,
    max_iter: int = 100,
    tol: float = 1e-8,
    num_batches: int = 10,
    verbose: bool = True,
) -> Dict:

    model.eval()
    model.to(device)
    criterion.to(device)
    
    # Collect multiple batches for more stable Hessian estimation
    batch_data = []
    for i, (inputs, targets) in enumerate(dataloader):
        if i >= num_batches:
            break
        batch_data.append((inputs.to(device), targets.to(device)))
    
    if verbose:
        print(f"[Hessian] Using {len(batch_data)} batches for estimation")
    
    def loss_fn():
        total_loss = 0.0
        for inputs, targets in batch_data:
            outputs = model(inputs)
            total_loss += criterion(outputs, targets)
        return total_loss / len(batch_data)
    
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    
    if verbose:
        print(f"[Hessian] Model has {n_params:,} parameters")
    
    def _flatten_grads(grad_tensors):
        return torch.cat([
            g.contiguous().view(-1) if g is not None else torch.zeros(p.numel(), device=device)
            for g, p in zip(grad_tensors, params)
        ])
    
    def hvp(vector):
        # hessian vector product using double backward pass

        model.zero_grad()
        
        # First backward pass: compute gradients
        loss = loss_fn()
        grads: tuple = torch.autograd.grad(
            loss, # type: ignore
            params, 
            create_graph=True, 
            retain_graph=True,
            allow_unused=True
        )
        grad_vec = _flatten_grads(grads)
        
        # Compute dot product with input vector
        grad_dot = torch.dot(grad_vec, vector)
        
        # Second backward pass: compute Hessian-vector product
        model.zero_grad()
        hv = torch.autograd.grad(
            grad_dot, 
            params, 
            retain_graph=True,
            allow_unused=True
        )
        
        return _flatten_grads(hv).detach()
    
    # Storage for eigenvalues and eigenvectors
    eigvals = []
    eigvecs = []
    convergence_info = []
    
    n = n_params
    
    for eig_idx in range(k):
        # Initialize random vector
        vec = torch.randn(n, device=device)
        vec = vec / vec.norm()
        
        prev_eigenvalue = None
        eigenvalue_history = []
        
        for it in range(max_iter):
            # Gram-Schmidt orthogonalization against previously found eigenvectors
            for prev_vec in eigvecs:
                proj = torch.dot(vec, prev_vec)
                vec = vec - proj * prev_vec
            
            # Normalize
            vec_norm = vec.norm()
            if vec_norm < 1e-10:
                # Vector became too small, reinitialize
                vec = torch.randn(n, device=device)
                for prev_vec in eigvecs:
                    proj = torch.dot(vec, prev_vec)
                    vec = vec - proj * prev_vec
                vec_norm = vec.norm()
            
            vec = vec / vec_norm
            
            # Compute Hessian-vector product
            hv = hvp(vec)
            
            # Compute eigenvalue using Rayleigh quotient
            eigenvalue = torch.dot(vec, hv).item()
            eigenvalue_history.append(eigenvalue)
            
            if verbose and it % 10 == 0:
                print(f"[Hessian] Eigen {eig_idx+1}/{k}, iter {it+1}/{max_iter}: λ={eigenvalue:.6f}")
            
            # Check convergence
            if prev_eigenvalue is not None:
                change = abs(eigenvalue - prev_eigenvalue)
                if change < tol:
                    if verbose:
                        print(f"[Hessian] Converged at iteration {it+1} (change={change:.2e})")
                    break
            
            prev_eigenvalue = eigenvalue
            
            # Update vector for next iteration (power iteration step)
            hv_norm = hv.norm()
            if hv_norm < 1e-10:
                if verbose:
                    print(f"[Hessian] Warning: HV product has very small norm at iter {it+1}")
                break
            vec = hv / hv_norm
        
        # Store results
        if prev_eigenvalue is not None:
            eigvals.append(eigenvalue) # type: ignore
            eigvecs.append(vec.detach().clone())
            convergence_info.append({
                'eigenvalue_index': eig_idx + 1,
                'converged': it < max_iter - 1, # type: ignore
                'iterations': it + 1, # type: ignore
                'final_eigenvalue': eigenvalue, # type: ignore
                'eigenvalue_history': eigenvalue_history
            })
        else:
            if verbose:
                print(f"[Hessian] Warning: Eigenvalue {eig_idx+1} did not converge properly")
    
    if verbose:
        print(f"\n[Hessian] Completed. Found {len(eigvals)}/{k} eigenvalues")
        if len(eigvals) > 0:
            print(f"[Hessian] Top eigenvalue: {eigvals[0]:.6f}")
            print(f"[Hessian] Bottom eigenvalue: {eigvals[-1]:.6f}")
    
    return {
        'top_eigenvalues': np.array(eigvals),
        'top_eigenvectors': [v.cpu().numpy() for v in eigvecs],
        'convergence_info': convergence_info,
        'num_parameters': n_params,
        'num_batches_used': len(batch_data)
    }