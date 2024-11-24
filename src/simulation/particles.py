# particles.py

import numpy as np
import warp as wp

from .constants import *

def polar_decomposition(F_E):
    # Perform polar decomposition to get rotation matrix  R_E and stretch matrix S_E
    # S is returned as a 1D array by np, we need to form a diagonal matrix from it
    U, S, Vt = np.linalg.svd(F_E)   # left singular vectors, singular vectors, right singular vectors
    R_E = U @ Vt                    # rotation matrix
    S_E = Vt.T @ np.diag(S) @ Vt    # stretch matrix
    return R_E, S_E

def plastic_deformation_thresholds(F_E, epsilon_c, epsilon_s):
    # Compute singular values of F_E - represent the amount of stretch or compression
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
        self.initial_volume = 1.0
        self.F_E = np.eye(3)  # Elastic part of deformation gradient - 3x3 identity matrix
        self.F_P = np.eye(3)  # Plastic part of deformation gradient - 3x3 identity matrix
        self.deformation_gradient = np.eye(3)
        self.density = 0
        self.mu_0 = MU
        self.lambda_0 = LAMBDA
        self.alpha = ALPHA # hardening parameter

    def position_x(self):
        return self.position[0]
    
    def position_y(self):
        return self.position[1]
    
    def position_z(self):
        return self.position[2]

    def stress_tensor(self):
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
        
        # Compute the total J = J_E * J_P
        J = J_E * J_P
        
        # Compute the Cauchy stress tensor
        stress = (1 / J) * dpsi_dF_E @ self.F_E.T
    
        
        return stress

    # def stress_tensor(self):
    #     # Compute the determinants of F_E (elastic deformation gradient) and F_P (plastic deformation gradient)
    #     J_E = np.linalg.det(self.F_E)
    #     J_P = np.linalg.det(self.F_P)

    #     # Compute the exponential hardening factor
    #     exp_factor = np.exp(self.alpha * (1 - J_P))

    #     # Compute Lamé parameters with hardening
    #     mu = self.mu_0 * exp_factor
    #     lambda_ = self.lambda_0 * exp_factor

    #     # Polar decomposition of F_E to obtain R_E
    #     R_E, _ = polar_decomposition(self.F_E)

    #     # Compute the first Piola-Kirchhoff stress tensor (P)
    #     # P = 2 * mu * (F_E - R_E) @ F_E^T + lambda * (J_E - 1) * J_E * I
    #     stress = (
    #         2.0 * mu * (self.F_E - R_E) @ self.F_E.T
    #         + lambda_ * (J_E - 1) * J_E * np.eye(3)
    #     )

    #     return stress  
    
    # def delta_force(F, r, weight_gradient, initial_volume, mu, lambda_, matrix_epsilon=1e-6):
    #     # Step 6 - Implicit velocity update
    #     """
    #     Compute the incremental force (deltaForce) based on residual velocity.
    #     """
    #     # 1. Compute delta(F): Change in the deformation gradient
    #     delta_F = TIMESTEP * np.outer(r, weight_gradient) @ F

    #     # Check small perturbations in delta_F
    #     if np.abs(delta_F).max() < matrix_epsilon:
    #         return np.zeros(3)

    #     # 2. Polar decomposition of F to get R (rotation) and S (stretching)
    #     U, S, Vt = np.linalg.svd(F)
    #     polar_r = U @ Vt  # Rotation matrix
    #     polar_s = Vt.T @ np.diag(S) @ Vt  # Symmetric stretch matrix

    #     # 3. Compute delta(R): Change in rotation matrix
    #     # Skew symmetric part of R^T delta(F) - delta(F)^T R
    #     skew_sym = polar_r.T @ delta_F - delta_F.T @ polar_r
    #     trace_s = np.trace(polar_s)  # Sum of diagonal entries of S
    #     if trace_s == 0:  # Avoid division by zero
    #         trace_s = matrix_epsilon

    #     x = np.trace(skew_sym) / trace_s  # Solve for single value
    #     delta_R = polar_r @ np.array([[0, -x, 0], [x, 0, 0], [0, 0, 0]])

    #     # 4. Compute cofactor matrix and delta(cofactor)
    #     cofactor = np.linalg.inv(F.T) * np.linalg.det(F)  # Cofactor matrix of F
    #     delta_cofactor = np.linalg.inv(delta_F.T) * np.linalg.det(delta_F)  # Approximation

    #     # 5. Compute A (force matrix) using the co-rotational term
    #     stress_delta = 2 * mu * (delta_F - delta_R)  # Co-rotational term
    #     stress_delta += lambda_ * np.trace(delta_F) * np.identity(3)  # Primary contour term

    #     # 6. Combine everything to compute delta(force)
    #     delta_force = initial_volume * stress_delta @ (F.T @ weight_gradient)
    #     return delta_force

    def update_deformation_gradient(self, grid, time_step):
        # Step 7: Update deformation gradient based on velocity gradient from grid
        # velocity_gradient = grid.compute_velocity_gradient(self.position)
        # self.deformation_gradient = (np.eye(3) + time_step * velocity_gradient) @ self.deformation_gradient

        I = np.identity(3)
        rv_np1 = np.zeros((3, 3))  # Velocity gradient (r v^{n+1}_p)

        # Compute gradient of v^{n+1}_p = ∑_i v^{n+1}_i (∇w_{ip})^T
        for node in grid.nodes:
            weight_gradient = node.compute_weight_gradient(self)
            rv_np1 += np.outer(node.velocity, weight_gradient)

        # Update F^{n+1}_p
        F_np1 = (I + time_step * rv_np1) @ self.deformation_gradient

        # Perform SVD on F^{n+1}_Ep = (I + Δt r v^{n+1}_p) F^n_Ep
        U, S, Vt = np.linalg.svd(F_np1 @ self.F_P)
        S_clamped = np.clip(S, 1 - CRIT_COMPRESS, 1 + CRIT_STRETCH)

        # Update elastic and plastic gradients
        self.F_E = U @ np.diag(S_clamped) @ Vt
        self.F_P = Vt.T @ np.diag(1.0 / S_clamped) @ U.T @ F_np1

        # Update total deformation gradient
        self.deformation_gradient = self.F_E @ self.F_P

    def update_velocity(self, grid, alpha=0.95):
        # Step 8 - Update particle velocities
        velocity_PIC = np.zeros(3)
        velocity_FLIP = self.velocity
        nearby_nodes = grid.get_nearby_nodes(self.position)
        for node in nearby_nodes:
            weight_ip = node.compute_weight(self)
            velocity_PIC += node.new_velocity * weight_ip
            velocity_FLIP += (node.new_velocity - node.velocity) * weight_ip
        self.velocity = (1 - alpha) * velocity_PIC + alpha * velocity_FLIP
        self.velocity[np.abs(self.velocity) < 1e-6] = 0
        print("Particle velocity: ", self.velocity)

    def apply_collision(self, collision_objects):
        # Step 9 - Particle-based body collisions
        for obj in collision_objects:
            if obj.is_colliding(self.position):
                    self.velocity = obj.collision_response(self.velocity, self.position + TIMESTEP * self.velocity)
                    self.velocity[np.abs(self.velocity) < 1e-6] = 0
                    print("Particle velocity after collision: ", self.velocity)
                    break   # The first collision dominates
            # How to handle multiple collisions at the same time?

    def update_position(self):
        # Step 10 - Update particle positions
        self.position += self.velocity * TIMESTEP


class ParticleSystem:
    def __init__(self):
        self.particles = []

    def add_particle(self, particle: Particle):
        self.particles.append(particle)
