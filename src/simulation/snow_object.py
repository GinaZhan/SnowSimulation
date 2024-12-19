import numpy as np
import warp as wp

class SnowObject:
    def __init__(self, particle_diameter=0.0072, target_density=400):
        self.particle_diameter = particle_diameter
        self.target_density = target_density
        self.particle_volume = particle_diameter**3
        self.particle_mass = self.particle_volume * target_density

    def create_snowball(self, radius, center, velocity):
        """
        Create a spherical snowball with specified radius, center, and velocity.
        """
        # Calculate volume of snowball
        snowball_volume = (4 / 3) * np.pi * radius**3
        
        # Calculate the number of particles
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
        velocities = [wp.vec3(*velocity) for _ in range(num_particles)]
        
        return {
            "positions": wp.array([wp.vec3(*p) for p in positions], dtype=wp.vec3, device="cuda"),
            "velocities": wp.array(velocities, dtype=wp.vec3, device="cuda"),
            "num_particles": num_particles
        }

    def create_snowcube(self, side_length, center, velocity):
        """
        Create a cubic snow object with specified side length, center, and velocity.
        """
        # Calculate volume of cube
        snowcube_volume = side_length**3
        
        # Calculate the number of particles
        num_particles = int(snowcube_volume / self.particle_volume)
        
        # Generate particle positions
        positions = []
        for _ in range(num_particles):
            pos = np.random.uniform(-side_length / 2, side_length / 2, 3)
            pos += np.array(center)
            positions.append(pos)
        
        # Create velocity array
        velocities = [wp.vec3(*velocity) for _ in range(num_particles)]
        
        return {
            "positions": wp.array([wp.vec3(*p) for p in positions], dtype=wp.vec3, device="cuda"),
            "velocities": wp.array(velocities, dtype=wp.vec3, device="cuda"),
            "num_particles": num_particles
        }
