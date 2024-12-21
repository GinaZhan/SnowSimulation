import numpy as np
import warp as wp

from .particles import ParticleSystem

class SnowObject:
    def __init__(self, particle_diameter=0.0072, target_density=400):
        self.particle_diameter = particle_diameter
        self.target_density = target_density
        self.particle_volume = particle_diameter**3
        self.particle_mass = self.particle_volume * target_density

    def create_snowball(self, radius, center, velocity):
        """
        Create a spherical snowball with specified radius, center, and velocity.
        Returns NumPy arrays for positions and velocities.
        """
        snowball_volume = (4 / 3) * np.pi * radius**3
        num_particles = int(snowball_volume / self.particle_volume)
        
        # Generate particle positions
        positions = []
        for _ in range(num_particles):
            pos = np.random.uniform(-radius, radius, 3)
            while np.linalg.norm(pos) > radius:
                pos = np.random.uniform(-radius, radius, 3)
            pos += np.array(center)
            positions.append(pos)
        
        # Create velocity array
        velocities = [velocity] * num_particles
        
        return {
            "positions": np.array(positions),
            "velocities": np.array(velocities),
            "num_particles": num_particles
        }

    def create_snowcube(self, side_length, center, velocity):
        """
        Create a cubic snow object with specified side length, center, and velocity.
        Returns NumPy arrays for positions and velocities.
        """
        snowcube_volume = side_length**3
        num_particles = int(snowcube_volume / self.particle_volume)
        
        # Generate particle positions
        positions = []
        for _ in range(num_particles):
            pos = np.random.uniform(-side_length / 2, side_length / 2, 3)
            pos += np.array(center)
            positions.append(pos)
        
        # Create velocity array
        velocities = [velocity] * num_particles
        
        return {
            "positions": np.array(positions),
            "velocities": np.array(velocities),
            "num_particles": num_particles
        }

def create_particle_system(snow_objects):
    """
    Aggregate multiple snow objects into a single particle system.
    Converts NumPy arrays to Warp arrays at the end.
    """
    all_positions = []
    all_velocities = []
    total_particles = 0

    for obj in snow_objects:
        all_positions.append(obj["positions"])
        all_velocities.append(obj["velocities"])
        total_particles += obj["num_particles"]

    positions = wp.array(
        np.concatenate(all_positions), dtype=wp.vec3, device="cuda"
    )
    velocities = wp.array(
        np.concatenate(all_velocities), dtype=wp.vec3, device="cuda"
    )

    particle_system = ParticleSystem(num_particles=total_particles, positions=positions, velocities=velocities)
    return particle_system