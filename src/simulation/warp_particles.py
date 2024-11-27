# particles.py

# import numpy as np
import warp as wp

from .constants import *

@wp.func
def polar_decomposition(F: wp.mat33):
    U = wp.mat33()
    sigma = wp.vec3()
    V = wp.mat33()
    wp.svd3(F, U, sigma, V)

    # Compute rotation matrix (U @ V^T)
    R = U @ V.transpose()

    # Compute stretch matrix (V @ diag(sigma) @ V^T)
    S = V @ wp.mat33(sigma[0], 0, 0, 0, sigma[1], 0, 0, 0, sigma[2]) @ V.transpose()

    return R, S

@wp.func
def plastic_deformation_thresholds(F_E: wp.mat33, epsilon_c: float, epsilon_s: float) -> bool:
    """
    Check if the singular values of F_E exceed the critical thresholds.
    """
    # Perform SVD on F_E
    U = wp.mat33()
    sigma = wp.vec3()
    V = wp.mat33()
    wp.svd3(F_E, U, sigma, V)

    # Check if any singular value exceeds the thresholds
    for i in range(3):
        if sigma[i] < (1.0 - epsilon_c) or sigma[i] > (1.0 + epsilon_s):
            return True  # Plastic deformation starts

    return False  # No plastic deformation

class ParticleSystem:
    def __init__(self, num_particles):
        self.num_particles = num_particles

        # Particle properties
        self.positions = wp.zeros(num_particles, dtype=wp.vec3, device="cuda")
        self.velocities = wp.zeros(num_particles, dtype=wp.vec3, device="cuda")
        self.masses = wp.zeros(num_particles, dtype=float, device="cuda")
        self.initial_volumes = wp.zeros(num_particles, dtype=float, device="cuda")

        # Deformation gradients
        self.F_E = wp.zeros(num_particles, dtype=wp.mat33, device="cuda")  # Elastic part
        self.F_P = wp.zeros(num_particles, dtype=wp.mat33, device="cuda")  # Plastic part
        self.density = wp.zeros(num_particles, dtype=float, device="cuda")

        # Material properties
        self.mu_0 = MU
        self.lambda_0 = LAMBDA
        self.alpha = ALPHA

    def initialize_particle(self, index, position, velocity, mass):
        self.positions[index] = wp.vec3(*position)
        self.velocities[index] = wp.vec3(*velocity)
        self.masses[index] = mass
        self.initial_volumes[index] = 1.0
        self.F_E[index] = wp.identity(wp.mat33)
        self.F_P[index] = wp.identity(wp.mat33)
    
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

@wp.kernel
def compute_stress_tensor_kernel(
    F_E: wp.array(dtype=wp.mat33),
    F_P: wp.array(dtype=wp.mat33),
    mu_0: float,
    lambda_0: float,
    alpha: float,
    stress: wp.array(dtype=wp.mat33),
    use_cauchy: int):  # 1 for Cauchy, 0 for Piola-Kirchhoff

    tid = wp.tid()

    F_E_local = F_E[tid]
    F_P_local = F_P[tid]

    # Compute determinants
    J_E = wp.mat33_determinant(F_E_local)
    J_P = wp.mat33_determinant(F_P_local)

    # Compute Lamé parameters with hardening
    exp_factor = wp.exp(alpha * (1.0 - J_P))
    mu = mu_0 * exp_factor
    lambda_ = lambda_0 * exp_factor

    # Perform SVD for polar decomposition
    U = wp.mat33()
    sigma = wp.vec3()
    V = wp.mat33()
    wp.svd3(F_E_local, U, sigma, V)
    R_E = U @ V.transpose()

    # Compute stress tensor
    if use_cauchy:
        dpsi_dF_E = 2 * mu * (F_E_local - R_E) + lambda_ * (J_E - 1) * J_E * wp.inverse(F_E_local).transpose()
        J = J_E * J_P
        stress[tid] = (1.0 / J) * dpsi_dF_E @ F_E_local.transpose()
    else:
        stress[tid] = (
            2.0 * mu * (F_E_local - R_E) @ F_E_local.transpose()
            + lambda_ * (J_E - 1) * J_E * wp.identity(wp.mat33)
        )

@wp.kernel
def update_deformation_gradient_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    F_E: wp.array(dtype=wp.mat33),
    F_P: wp.array(dtype=wp.mat33),
    deformation_gradient: wp.array(dtype=wp.mat33),
    time_step: float,
    crit_compress: float,
    crit_stretch: float):

    tid = wp.tid()

    # Compute velocity gradient
    rv_np1 = wp.mat33(0.0)  # Replace with grid-based computation if necessary

    # Update F_E^{n+1}_p
    F_Epn1 = (wp.identity(wp.mat33) + time_step * rv_np1) @ F_E[tid]

    # Update F^{n+1}_p
    F_np1 = (wp.identity(wp.mat33) + time_step * rv_np1) @ deformation_gradient[tid]

    # Perform SVD on F_E^{n+1}_p
    U = wp.mat33()
    sigma = wp.vec3()
    V = wp.mat33()
    wp.svd3(F_Epn1, U, sigma, V)

    # Clamp singular values
    sigma_clamped = wp.vec3(
        wp.clamp(sigma[0], 1.0 - crit_compress, 1.0 + crit_stretch),
        wp.clamp(sigma[1], 1.0 - crit_compress, 1.0 + crit_stretch),
        wp.clamp(sigma[2], 1.0 - crit_compress, 1.0 + crit_stretch),
    )

    # Update F_E and F_P
    F_E[tid] = U @ wp.diag(sigma_clamped) @ V.transpose()
    F_P[tid] = V.transpose() @ wp.diag(1.0 / sigma_clamped) @ U.transpose() @ F_np1

    # Update total deformation gradient
    deformation_gradient[tid] = F_E[tid] @ F_P[tid]

@wp.kernel
def update_velocity_kernel(
    velocities: wp.array(dtype=wp.vec3),
    grid_velocities: wp.array(dtype=wp.vec3),  # Replace with appropriate grid data
    weights: wp.array(dtype=float),
    alpha: float):

    tid = wp.tid()
    velocity_pic = wp.vec3(0.0)
    velocity_flip = velocities[tid]

    for i in range(3):  # Replace with logic for accessing nearby grid nodes
        velocity_pic += grid_velocities[i] * weights[i]
        velocity_flip += (grid_velocities[i] - velocities[tid]) * weights[i]

    velocities[tid] = (1.0 - alpha) * velocity_pic + alpha * velocity_flip

@wp.kernel
def apply_collision_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    collision_objects_level_sets: wp.array(dtype=wp.mat33),  # Array of level-set representations
    collision_objects_velocities: wp.array(dtype=wp.vec3),
    collision_objects_friction: wp.array(dtype=float),
    num_objects: int,
    timestep: float):

    tid = wp.tid()  # Thread ID for the current particle

    position = positions[tid]
    velocity = velocities[tid]

    for obj_idx in range(num_objects):
        # Example collision object handling (level set)
        level_set = collision_objects_level_sets[obj_idx]
        object_velocity = collision_objects_velocities[obj_idx]
        friction_coefficient = collision_objects_friction[obj_idx]

        # Check for collision (simplified; replace with your level-set logic)
        if level_set(position) <= 0.0:  # Particle is inside or on the object
            relative_velocity = velocity - object_velocity

            # Compute collision response (simplified example)
            normal = wp.vec3(0.0, 1.0, 0.0)  # Replace with actual normal from level set
            vn = wp.dot(relative_velocity, normal)

            if vn < 0.0:  # Collision detected
                # Friction response
                vt = relative_velocity - vn * normal
                if wp.length(vt) <= -friction_coefficient * vn:
                    relative_velocity = wp.vec3(0.0)
                else:
                    relative_velocity = vt + friction_coefficient * vn * vt / wp.length(vt)

                # Final velocity
                velocity = relative_velocity + object_velocity

                # Break after handling one collision
                break

    velocities[tid] = velocity

@wp.kernel
def update_position_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    timestep: float):

    tid = wp.tid()  # Thread ID for the current particle

    positions[tid] += velocities[tid] * timestep