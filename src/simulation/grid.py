import numpy as np
import warp as wp

from .particles import *
from .constants import *

# wp.init()

GRAVITY = np.array([0.0, -9.8, 0.0])  # Gravity vector pointing down (negative Y-axis)
# GRAVITY = wp.vec3(0.0, -9.8, 0.0)  # Gravity vector pointing down (negative Y-axis)

def N(x):
    if np.abs(x) >= 0 and np.abs(x) < 1:
        result = 1/2*(np.abs(x))**3 - x**2 + 2/3
    elif np.abs(x) >= 1 and np.abs(x) < 2:
        result = -1/6*(np.abs(x))**3 + x**2 - 2*np.abs(x) + 4/3
    else:
        return 0
    
    if result < WEIGHT_EPSILON:
        return 0
    else:
        return result
    
def N_prime(x):
    if np.abs(x) >= 0 and np.abs(x) < 1:
        result = 3/2*np.abs(x)*x - 2*x                  # 1.5*|x|**2 * sign(x) + 2*x
        return result
    elif np.abs(x) >= 1 and np.abs(x) < 2:
        result = -1/2*np.abs(x)*x + 2*x - 2*x/np.abs(x) # -0.5*|x|**2 * sign(x) + 2*x - 2*sign(x) 
        return result
    else:
        return 0

class GridNode:
    def __init__(self, position, grid_space=1.0):
        self.position = np.array(position)
        self.mass = 0.0
        self.velocity = np.zeros(3)     # 3D velocity
        self.new_velocity = np.zeros(3) # next moment velocity
        self.velocity_star = np.zeros(3) # temporary velocity to store the new velocity in calculation
        self.force = np.zeros(3)        # Force vector
        self.grid_space = grid_space
        self.density = 0
        self.active = False
        self.imp_active = False
        self.err = np.zeros(3)      # error of estimate
        self.r = np.zeros(3)        # residual of estimate
        self.p = np.zeros(3)        # Conjugate direction
        self.Ep = np.zeros(3)       # Ep term in Conjugate Residuals
        self.Er = np.zeros(3)       # Er term in Conjugate Residuals
        self.rEr = 0.0              # cached value for r.dot(Er)

    def clear(self):
        self.mass = 0.0
        self.velocity.fill(0)
        self.force.fill(0)
        self.velocity_star.fill(0)

    def update_velocity_star(self):
        # Step 4
        if self.mass > 0:
            self.velocity_star = self.velocity + (self.force / self.mass + GRAVITY) * TIMESTEP # f = ma
            # print("GridNode force: ", self.force)
            # print("GridNode velocity: ", self.velocity_star)

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
        # self.nodes = [[GridNode((i, j, k)) for k in range(size)] for i in range(size) for j in range(size)]
        # self.nodes = [[[GridNode((i, j, k)) for k in range(size)] for j in range(size)] for i in range(size)]
        self.nodes = []
        self.size = size
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    self.nodes.append(GridNode((i, j, k)))

    def clear(self):
        for node in self.nodes:
            node.clear()

    def get_node(self, i, j, k):
        """
        Access a node using 3D indices.
        """
        index = i * self.size * self.size + j * self.size + k
        return self.nodes[index]

    def get_nearby_nodes(self, particle_position):
        """
        Get grid nodes within a given radius of the particle's position.
        """
        nearby_nodes = []
        grid_index = np.floor(particle_position / GRID_SPACE).astype(int)

        for dx in range(-2, 3):  # Search in a 5x5x5 cube
            for dy in range(-2, 3):
                for dz in range(-2, 3):
                    neighbor_index = grid_index + np.array([dx, dy, dz])
                    if (0 <= neighbor_index[0] < self.size and
                        0 <= neighbor_index[1] < self.size and
                        0 <= neighbor_index[2] < self.size):
                        nearby_nodes.append(self.get_node(neighbor_index[0], neighbor_index[1], neighbor_index[2]))
        return nearby_nodes

    def transfer_mass_and_velocity(self, particles):
        # Step 1 - Rasterize particle data to the grid - Transfer mass and velocity to grid nodes from particles

        # self.clear()    # prevent nodes from collecting velocity, mass, force from previous steps

        for p in particles:
            nearby_nodes = self.get_nearby_nodes(p.position)
            # total_weight = 0
            for node in nearby_nodes:
            # for node in self.nodes:
                dist = np.linalg.norm(p.position - node.position)
                if dist > 2:
                    continue

                weight_node_particle = node.compute_weight(p)
                # total_weight += weight_node_particle
                node.mass += p.mass * weight_node_particle
                node.velocity += p.velocity * p.mass * weight_node_particle
                # print("node mass has been updated in step 1: ", node.mass)

            # print("Total weight: ", total_weight)

        for node in self.nodes:
            if node.mass > 0:
                node.velocity /= node.mass
                # print("node velocity has been updated in step 1")
                # print("node mass has been updated in step 1: ", node.mass)


    def setup_particle_density_volume(self, particles):
        # Step 2 - Compute particle volumes and densities - first timestamp only
        # Here we don't reset node and particle density because this function is only called once at first timestamp
        for p in particles:
            p.density = 0
            nearby_nodes = self.get_nearby_nodes(p.position)
            for node in nearby_nodes:
                node.density = node.mass / node.grid_space**3
                weight = node.compute_weight(p)
                if weight > WEIGHT_EPSILON:
                    p.density += node.density * weight

            if p.density > 0:
                p.initial_volume = p.mass / p.density
                # print("particle density has been updated in step 2")
            else:
                # may cause issue; TODO: add small density value
                raise ValueError("This particle has 0 density!")

    def compute_grid_forces(self, particles):
        # Step 3 - Compute grid forces
        for node in self.nodes:
            node.force.fill(0)

        for p in particles:
            stress_tensor = p.stress_tensor()
            print("Stress tensor: ", stress_tensor)
            Jpn = np.linalg.det(p.deformation_gradient)
            Vpn = Jpn * p.initial_volume

            for node in self.get_nearby_nodes(p.position):
                weight_gradient = node.compute_weight_gradient(p)
                node.force -= Vpn * stress_tensor @ weight_gradient
        # print("node force has been updated in step 3")

    def update_grid_velocity_star(self):
        # Step 4 - Update grid velocity
        for node in self.nodes:
            node.update_velocity_star()
        # print("node star velocity has been updated in step 4")

    def apply_collisions(self, collision_objects):
        # Step 5
        """Apply collisions to each grid node's velocity based on collision objects."""
        for node in self.nodes:
            for obj in collision_objects:
                if obj.is_colliding(node.position):
                    node.velocity_star = obj.collision_response(node.velocity_star, node.position + TIMESTEP * node.velocity_star)

    # def recomputeImplicitForces(self, grid_nodes):
    #     # Step 6
    #     for node in grid_nodes:
    #         if node.imp_active:
    #             node.force = 0

    #     # Update Er values based on the updated forces
    #     for node in grid_nodes:
    #         if node.imp_active:
    #             # Er = r - IMPLICIT_RATIO * TIMESTEP * force / mass
    #             node.Er = node.r - (IMPLICIT_RATIO * TIMESTEP / node.mass) * node.force

    # def conjugate_residuals_with_force_recalculation(self, grid_nodes, gravity):
    #     # Step 6
    #     for node in grid_nodes:
    #         if node.active:
    #             node.imp_active = True
    #             node.r = node.velocity_new  # Initial guess for r
    #             node.err = np.ones(3)       # Initialize the error term
    #             node.force = np.zeros(3)    # Clear previous force
                
    #     self.recomputeImplicitForces(grid_nodes)
        
    #     for node in grid_nodes:
    #         if node.imp_active:
    #             # Initial residual r = v* - E*v*
    #             node.r = node.velocity_new - node.Er  # Starting residual
    #             node.p = node.r                       # Set initial conjugate direction
    #             node.rEr = np.dot(node.r, node.Er)    # Cache r.dot(Er)

    #     for iter_num in range(MAX_IMPLICIT_ITERS):
    #         done = True
    #         for node in grid_nodes:
    #             if node.imp_active:
    #                 # Update velocity guess based on the residual
    #                 alpha = node.rEr / np.dot(node.Ep, node.Ep)
    #                 node.err = alpha * node.p  # Update error term
    #                 err = np.linalg.norm(node.err)
    #                 if err < MAX_IMPLICIT_ERR or np.isnan(err):
    #                     node.imp_active = False  # Converged, deactivate node
    #                     continue
    #                 else:
    #                     done = False  # If any node is still active, we're not done

    #                 # Update velocity and residual
    #                 node.velocity_new += node.err  # Update velocity
    #                 node.r -= alpha * node.Ep      # Update residual

    #         if done:
    #             break  # Exit if all nodes have converged

    #         # Step 3.3: Recompute forces and residuals
    #         self.recomputeImplicitForces(grid_nodes)

    #         # Step 3.4: Update direction vector p
    #         for node in grid_nodes:
    #             if node.imp_active:
    #                 new_rEr = np.dot(node.r, node.Er)
    #                 beta = new_rEr / node.rEr
    #                 node.rEr = new_rEr            # Update rEr for the next iteration
    #                 node.p = node.r + beta * node.p  # Update conjugate direction
    #                 node.Ep = node.Er + beta * node.Ep  # Update Ep for next iteration

    #     # Finalize velocities
    #     for node in grid_nodes:
    #         if node.active:
    #             node.velocity = node.velocity_new

    def explicit_update_velocity(self):
        # Step 6
        for node in self.nodes:
            node.new_velocity = node.velocity_star
        # print("node velocity has been updated in step 6")
