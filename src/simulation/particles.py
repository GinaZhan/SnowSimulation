# particles.py

import numpy as np
import warp as wp


class Particle:
    def __init__(self, position, velocity, mass, volume):
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.mass = mass
        self.volume = volume
        self.deformation_gradient = np.eye(3)  # 3x3 identity matrix

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

class ParticleSystem:
    def __init__(self):
        self.particles = []

    def add_particle(self, particle: Particle):
        self.particles.append(particle)
