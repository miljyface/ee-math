import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def plot_3d_landscape(json_path: str,
                     title: str = "Loss Landscape",
                     elev: int = 30,
                     azim: int = 45,
                     colormap: str = 'plasma',
                     save_path: Optional[str] = None,
                     show_plot: bool = True) -> None:
    # Load data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    landscape = data['loss_landscape']
    
    # Extract grids
    alpha_grid = np.array(landscape['alpha_grid'])
    beta_grid = np.array(landscape['beta_grid'])
    loss_grid = np.array(landscape['loss_grid'])
    
    # Find minimum
    min_loss = landscape['loss_min']
    min_idx = np.unravel_index(np.argmin(loss_grid), loss_grid.shape)
    
    # Create figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface( # type: ignore
        alpha_grid,
        beta_grid,
        loss_grid,
        cmap=colormap,
        linewidth=0,
        antialiased=True,
        alpha=0.9,
        edgecolor='none'
    )
    
    # Mark minimum point
    ax.scatter(
        [alpha_grid[min_idx]],
        [beta_grid[min_idx]],
        [loss_grid[min_idx]],
        color='red',
        marker='*',
        label=f'Min Loss: {min_loss:.4f}',
        zorder=10,
        edgecolors='white',
        linewidths=1.5
    )
    
    # Labels
    ax.set_xlabel('Direction α', fontsize=12, labelpad=10)
    ax.set_ylabel('Direction β', fontsize=12, labelpad=10)
    ax.set_zlabel('Loss', fontsize=12, labelpad=10) # type: ignore
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Set viewing angle
    ax.view_init(elev=elev, azim=azim) # type: ignore
    
    # Colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label('Loss Value', rotation=270, labelpad=20, fontsize=11)
    
    # Legend
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 3D plot to {save_path}")
    
    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close()

norm = 'layer'
plot_3d_landscape(
    json_path=f"/Users/guanrong/Desktop/ee_shit/ee-math/output/json/landscape_{norm}.json",
    title=f"{norm} norm Loss Landscape",
    save_path=f"/Users/guanrong/Desktop/ee_shit/ee-math/output/images/{norm}_3d.png"
)