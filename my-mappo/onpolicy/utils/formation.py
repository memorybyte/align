"""
Formation utilities.

Formation error computation:
1. Center both current positions and target positions
2. Compute optimal rotation R via SVD of P^T @ T
3. Compute error E = (1/N) * sum_i || R @ p_i - t_i ||^2
4. Normalization G = max pairwise distance squared of target formation
"""

import numpy as np
from typing import Tuple


def procrustes_alignment(
    source: np.ndarray, 
    target: np.ndarray,
    allow_reflection: bool = False
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute the optimal rigid transformation (rotation only) from source to target
    using Procrustes analysis / Umeyama algorithm.
    
    - Center both sets: P = positions - mean(positions), T = targets - mean(targets)
    - Compute rotation R via SVD of P^T @ T
    
    Args:
        source: Source points (N, D) - current drone positions (centered)
        target: Target points (N, D) - target formation (centered)
        allow_reflection: Whether to allow reflection (improper rotation)
        
    Returns:
        R: Optimal rotation matrix (D, D)
        aligned_source: Source points after rotation (N, D)
        error: Sum of squared distances after alignment
    """
    # SVD of cross-covariance matrix
    # R via SVD of P^T @ T
    H = target.T @ source  # (D, D)
    U, S, Vt = np.linalg.svd(H)
    
    # Compute rotation
    R = Vt.T @ U.T
    
    # Handle reflection case (ensure proper rotation with det(R) = 1)
    if not allow_reflection and np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Apply rotation
    aligned_target = target @ R.T
    
    # Compute error (sum of squared differences)
    error = np.sum((aligned_target - source) ** 2)
    
    return R, aligned_target, error


def compute_formation_error(
    positions: np.ndarray, 
    target_formation: np.ndarray,
    return_details: bool = False
) -> Tuple[float, float]:
    """
    Compute formation error E and normalization factor G.
    
    E = (1/N) * sum_i || R @ p_i - t_i ||^2
    G = max pairwise distance squared of target formation
    
    Formation reward: r_form = -E / (G + eps)
    
    Args:
        positions: Current drone positions (N, 3)
        target_formation: Target formation positions (N, 3)
        return_details: Whether to return additional alignment details
        
    Returns:
        E: Formation error (mean squared distance after alignment)
        G: Normalization factor (max pairwise distance squared)
    """
    N = positions.shape[0]
    
    if N < 2:
        # Single drone: no formation error
        return 0.0, 1.0
    
    # Center both point sets
    positions_centered = positions - np.mean(positions, axis=0, keepdims=True)
    target_centered = target_formation - np.mean(target_formation, axis=0, keepdims=True)
    
    # Compute optimal rotation and aligned positions
    R, aligned_positions, total_error = procrustes_alignment(
        positions_centered, 
        target_centered
    )
    
    # Formation error: mean squared error
    # Reference: E = (1/N) * sum_i || R p_i - t_i ||^2
    E = total_error
    
    # Normalization factor: max pairwise distance squared of target formation
    G = compute_normalization_factor(target_formation)
    
    return E, G


def compute_normalization_factor(formation: np.ndarray) -> float:
    """
    Compute normalization factor G as max pairwise distance squared.
    G = max_pairwise_dist^2
    
    Args:
        formation: Formation positions (N, 3)
        
    Returns:
        G: Max pairwise distance squared
    """
    N = formation.shape[0]
    
    if N < 2:
        return 1.0
    
    max_dist_sq = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dist_sq = np.sum((formation[i] - formation[j]) ** 2)
            max_dist_sq = max(max_dist_sq, dist_sq)
    
    return max(max_dist_sq, 1e-8)


def generate_formation(
    formation_type: str,
    num_agents: int,
    scale: float = 1.0,
    height: float = 1.0
) -> np.ndarray:
    """
    Generate common formation shapes.
    
    Args:
        formation_type: Type of formation ('triangle', 'line', 'circle', 'grid')
        num_agents: Number of agents
        scale: Scale factor for formation size
        height: Height of the formation
        
    Returns:
        formation: Target positions (num_agents, 3)
    """
    if formation_type == 'triangle' or (formation_type == 'circle' and num_agents == 3):
        # Equilateral triangle
        angles = np.linspace(0, 2 * np.pi, num_agents, endpoint=False) + np.pi/2
        x = scale * np.cos(angles)
        y = scale * np.sin(angles)
        z = np.ones(num_agents) * height
        
    elif formation_type == 'line':
        # Line formation
        x = np.linspace(-scale * (num_agents - 1) / 2, scale * (num_agents - 1) / 2, num_agents)
        y = np.zeros(num_agents)
        z = np.ones(num_agents) * height
        
    elif formation_type == 'circle':
        # Circular formation
        angles = np.linspace(0, 2 * np.pi, num_agents, endpoint=False)
        x = scale * np.cos(angles)
        y = scale * np.sin(angles)
        z = np.ones(num_agents) * height
        
    elif formation_type == 'grid':
        # Grid formation (best effort for non-square numbers)
        side = int(np.ceil(np.sqrt(num_agents)))
        x = []
        y = []
        for i in range(num_agents):
            x.append((i % side - (side - 1) / 2) * scale)
            y.append((i // side - (side - 1) / 2) * scale)
        x = np.array(x)
        y = np.array(y)
        z = np.ones(num_agents) * height
        
    else:
        raise ValueError(f"Unknown formation type: {formation_type}")
    
    return np.stack([x, y, z], axis=1)


def compute_formation_velocity(
    positions: np.ndarray,
    target_formation: np.ndarray,
    target_center: np.ndarray,
    k_formation: float = 1.0,
    k_navigation: float = 0.5,
) -> np.ndarray:
    """
    Compute desired velocities to achieve formation while navigating to target.
    
    This is a simple proportional controller for formation keeping and navigation.
    
    Args:
        positions: Current positions (N, 3)
        target_formation: Target formation shape (N, 3)
        target_center: Target center position (3,)
        k_formation: Gain for formation keeping
        k_navigation: Gain for center navigation
        
    Returns:
        velocities: Desired velocities for each agent (N, 3)
    """
    N = positions.shape[0]
    
    # Center target formation at target_center
    formation_centered = target_formation - np.mean(target_formation, axis=0)
    desired_positions = formation_centered + target_center
    
    # Position error
    position_error = desired_positions - positions
    
    # Simple proportional control
    velocities = k_formation * position_error
    
    # Add navigation component (move towards target center)
    current_center = np.mean(positions, axis=0)
    nav_error = target_center - current_center
    velocities += k_navigation * nav_error
    
    return velocities