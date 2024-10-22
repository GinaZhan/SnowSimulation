# particles.py

import numpy as np
import warp as wp

def polar_decomposition(F_E): # ???
    # Perform polar decomposition to get rotation R_E and stretch S_E
    U, S, Vt = np.linalg.svd(F_E)
    R_E = U @ Vt
    S_E = Vt.T @ np.diag(S) @ Vt  # Stretch matrix
    return R_E, S_E

def plastic_deformation_thresholds(F_E, epsilon_c, epsilon_s): # ???
    # Compute singular values of F_E
    _, singular_values, _ = np.linalg.svd(F_E)
    
    # Check if singular values exceed the critical thresholds
    for sigma in singular_values:
        if sigma < (1 - epsilon_c) or sigma > (1 + epsilon_s):
            return True  # Plastic deformation starts
    return False  # No plastic deformation



class Particle:
    def __init__(self, position, velocity, mass):
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.mass = mass
        self.volume = 0
        self.initial_volume = 0
        self.FE = np.eye(3)  # Elastic part of deformation gradient - 3x3 identity matrix
        self.FP = np.eye(3)  # Plastic part of deformation gradient - 3x3 identity matrix
        self.density = 0

    def update_velocity(self, grid, time_step):
        # Interpolate velocity from grid nodes to particle
        self.velocity = grid.interpolate_velocity_to_particle(self.position)

    def update_position(self, time_step):
        # Simple forward Euler update for position
        self.position += self.velocity * time_step

    def update_deformation_gradient(self, grid, time_step):
        # Update deformation gradient based on velocity gradient from grid
        velocity_gradient = grid.compute_velocity_gradient(self.position)
        self.deformation_gradient = (np.eye(3) + time_step * velocity_gradient) @ self.deformation_gradient

    def position_x(self):
        return self.position[0]
    
    def position_y(self):
        return self.position[1]
    
    def position_z(self):
        return self.position[2]
    
    # def stress_tensor(self):
    #     pass

    def stress_tensor(self): # ???
        # Compute the determinants
        J_E = np.linalg.det(self.F_E)
        J_P = np.linalg.det(self.F_P)
        
        # Compute Lamé parameters with hardening
        mu = self.mu_0 * np.exp(self.alpha * (1 - J_P))
        lambda_ = self.lambda_0 * np.exp(self.alpha * (1 - J_P))
        
        # Polar decomposition to get R_E
        R_E, _ = polar_decomposition(self.F_E)
        
        # Derivative of the elasto-plastic potential energy density function w.r.t. F_E
        dpsi_dF_E = 2 * mu * (self.F_E - R_E) + lambda_ * (J_E - 1) * J_E * np.linalg.inv(self.F_E).T
        
        # Compute the Cauchy stress tensor
        stress = (1 / J_E) * dpsi_dF_E @ self.F_E.T
        
        return stress


class ParticleSystem:
    def __init__(self):
        self.particles = []

    def add_particle(self, particle: Particle):
        self.particles.append(particle)
