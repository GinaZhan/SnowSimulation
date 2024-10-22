import numpy as np
import warp as wp

def N(x):
    if np.abs(x) >= 0 and np.abs(x) < 1:
        result = 1/2*(np.abs(x))**3 - x**2 + 2/3
        return result
    elif np.abs(x) >= 1 and np.abs(x) < 2:
        result = -1/6*(np.abs(x))**3 + x**2 - 2*np.abs(x) + 4/3
        return result
    else:
        return 0
    
def N_prime(x):
    if np.abs(x) >= 0 and np.abs(x) < 1:
        result = 3/2*np.abs(x)*x - 2*x
        return result
    elif np.abs(x) >= 1 and np.abs(x) < 2:
        result = -1/2*np.abs(x)*x + 2*x - 2*x/np.abs(x)
        return result
    else:
        return 0

class GridNode:
    def __init__(self, position, grid_space=1):
        self.position = np.array(position)
        self.mass = 0.0
        self.velocity = np.zeros(3)  # 3D velocity
        self.force = np.zeros(3)     # Force vector
        self.grid_space = grid_space
        self.density = 0

    def clear(self):
        self.mass = 0.0
        self.velocity.fill(0)
        self.force.fill(0)

    def update_velocity(self, time_step):
        if self.mass > 0:
            self.velocity += (self.force / self.mass) * time_step # f = ma

    def compute_weight(self, particle):
        x_weight = N((particle.position_x() - self.position[0]*self.grid_space)/self.grid_space)
        y_weight = N((particle.position_y() - self.position[1]*self.grid_space)/self.grid_space)
        z_weight = N((particle.position_z() - self.position[2]*self.grid_space)/self.grid_space)
        return x_weight*y_weight*z_weight
    
    def compute_weight_gradient(self, particle):
        x_diff = (particle.position_x() - self.position[0]*self.grid_space)/self.grid_space
        y_diff = (particle.position_y() - self.position[1]*self.grid_space)/self.grid_space
        z_diff = (particle.position_z() - self.position[2]*self.grid_space)/self.grid_space
        
        x_weight_gradient = (1/self.grid_space) * N_prime(x_diff) * N(y_diff) * N(z_diff)
        y_weight_gradient = (1/self.grid_space) * N(x_diff) * N_prime(y_diff) * N(z_diff)
        z_weight_gradient = (1/self.grid_space) * N(x_diff) * N(y_diff) * N_prime(z_diff)

        return np.array([x_weight_gradient, y_weight_gradient, z_weight_gradient])

class Grid:
    def __init__(self, size):
        # Initialize a grid with a given size
        self.nodes = [[GridNode((i, j, k)) for k in range(size)] for i in range(size) for j in range(size)]

    def clear(self):
        for node in self.nodes:
            node.clear()

    def transfer_mass_and_velocity(self, particles):
        # Step 1 - Rasterize particle data to the grid - Transfer mass and velocity to grid nodes from particles
        for p in particles:
            for node in self.nodes:
                dist = np.linalg.norm(p.position - node.position)
                if dist > 2:
                    continue

                weight_node_particle = node.compute_weight(p)
                node.mass += p.mass * weight_node_particle
                node.velocity += p.velocity * p.mass * weight_node_particle

        for node in self.nodes:
            if node.mass > 0:
                node.velocity /= node.mass

    def setup_particle_density_volume(self, particles):
        # Step 2 - Compute particle volumes and densities - first timestamp only
        # Here we don't reset node and particle density because this function is only called once at first timestamp
        for p in particles:
            for node in self.nodes:
                node.density = self.mass / node.grid_space**3
                p.density += node.mass * node.compute_weight(p) / node.grid_space**3

            if p.density > 0:
                p.initial_volume = p.mass / p.density
            else:
                raise ValueError("This particle has 0 density!")

    def compute_grid_forces(self, particles):
        # Step 3 - Compute grid forces
        for node in self.nodes:
            node.force.fill(0)

        for p in particles:
            stress_tensor = p.stress_tensor()
            Jpn = np.linalg.det(p.deformation_gradient)
            Vpn = Jpn * p.initial_volume

            for node in self.nodes:
                weight_gradient = node.compute_weight_gradient(p)
                node.force -= Vpn * stress_tensor @ weight_gradient

    def interpolate_velocity_to_particle(self, particle_position):
        # Interpolate velocity from grid nodes to the particle's position
        # (Weighting/interpolation logic here)
        return np.zeros(3)

    def compute_velocity_gradient(self, particle_position):
        # Compute velocity gradient at the particle's position
        # (Interpolation logic to get gradients)
        return np.zeros((3, 3))  # 3x3 velocity gradient matrix
