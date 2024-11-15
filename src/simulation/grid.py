import numpy as np
import warp as wp

from particles import *

MAX_IMPLICIT_ITERS = 30
MAX_IMPLICIT_ERR = 1e4
MIN_IMPLICIT_ERR = 1e-4
TIMESTEP = 0.05
IMPLICIT_RATIO = 0.5


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
        self.velocity = np.zeros(3) # 3D velocity
        self.force = np.zeros(3)    # Force vector
        self.grid_space = grid_space
        self.density = 0
        self.new_velocity = np.zeros(3)
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

    def update_velocity(self, time_step):
        # Step 4
        if self.mass > 0:
            self.new_velocity = self.velocity + (self.force / self.mass) * time_step # f = ma

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
        self.nodes = [[[GridNode((i, j, k)) for k in range(size)] for j in range(size)] for i in range(size)]

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
                node.density = node.mass / node.grid_space**3
                p.density += node.density * node.compute_weight(p)

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

    def update_grid_velocity(self, time_step):
        # Step 4 - Update grid velocity
        for node in self.nodes:
            node.update_velocity(time_step)

    def recomputeImplicitForces(self, grid_nodes):
        for node in grid_nodes:
            if node.imp_active:
                node.force = 0

        # Update Er values based on the updated forces
        for node in grid_nodes:
            if node.imp_active:
                # Er = r - IMPLICIT_RATIO * TIMESTEP * force / mass
                node.Er = node.r - (IMPLICIT_RATIO * TIMESTEP / node.mass) * node.force

    def conjugate_residuals_with_force_recalculation(self, grid_nodes, gravity):
        for node in grid_nodes:
            if node.active:
                node.imp_active = True
                node.r = node.velocity_new  # Initial guess for r
                node.err = np.ones(3)       # Initialize the error term
                node.force = np.zeros(3)    # Clear previous force
                
        self.recomputeImplicitForces(grid_nodes)
        
        for node in grid_nodes:
            if node.imp_active:
                # Initial residual r = v* - E*v*
                node.r = node.velocity_new - node.Er  # Starting residual
                node.p = node.r                       # Set initial conjugate direction
                node.rEr = np.dot(node.r, node.Er)    # Cache r.dot(Er)

        for iter_num in range(MAX_IMPLICIT_ITERS):
            done = True
            for node in grid_nodes:
                if node.imp_active:
                    # Step 3.1: Update velocity guess based on the residual
                    alpha = node.rEr / np.dot(node.Ep, node.Ep)
                    node.err = alpha * node.p  # Update error term
                    err = np.linalg.norm(node.err)
                    if err < MAX_IMPLICIT_ERR or np.isnan(err):
                        node.imp_active = False  # Converged, deactivate node
                        continue
                    else:
                        done = False  # If any node is still active, we're not done

                    # Step 3.2: Update velocity and residual
                    node.velocity_new += node.err  # Update velocity
                    node.r -= alpha * node.Ep      # Update residual

            if done:
                break  # Exit if all nodes have converged

            # Step 3.3: Recompute forces and residuals
            self.recomputeImplicitForces(grid_nodes)

            # Step 3.4: Update direction vector p
            for node in grid_nodes:
                if node.imp_active:
                    new_rEr = np.dot(node.r, node.Er)
                    beta = new_rEr / node.rEr
                    node.rEr = new_rEr            # Update rEr for the next iteration
                    node.p = node.r + beta * node.p  # Update conjugate direction
                    node.Ep = node.Er + beta * node.Ep  # Update Ep for next iteration

        # Finalize velocities
        for node in grid_nodes:
            if node.active:
                node.velocity = node.velocity_new
    def apply_collisions(self, collision_objects):
        # Step 5
        """Apply collisions to each grid node's velocity based on collision objects."""
        for node in self.nodes:
            for obj in collision_objects:
                if obj.is_colliding(node.position):
                    node.new_velocity = obj.collision_response(node.velocity, node.position)

    def interpolate_velocity_to_particle(self, particle_position):
        # Interpolate velocity from grid nodes to the particle's position
        # (Weighting/interpolation logic here)
        return np.zeros(3)

    def compute_velocity_gradient(self, particle_position):
        # Compute velocity gradient at the particle's position
        # (Interpolation logic to get gradients)
        return np.zeros((3, 3))  # 3x3 velocity gradient matrix


class CollisionObject:
    def __init__(self, level_set, velocity_function, friction_coefficient=0.5):
        """
        level_set: function that returns the level set value φ(x) for a given position x.
        velocity_function: function that returns the object velocity v_co at a given position.
        friction_coefficient: coefficient of friction, μ.
        """
        self.level_set = level_set
        self.velocity_function = velocity_function
        self.friction_coefficient = friction_coefficient

    def is_colliding(self, position):
        """Check if the position is colliding (φ <= 0)."""
        return self.level_set(position) <= 0

    def normal(self, position):
        """Compute the normal vector n = ∇φ at the position."""
        epsilon = 1e-5
        grad_phi = np.array([
            (self.level_set(position + np.array([epsilon, 0, 0])) - self.level_set(position)) / epsilon,
            (self.level_set(position + np.array([0, epsilon, 0])) - self.level_set(position)) / epsilon,
            (self.level_set(position + np.array([0, 0, epsilon])) - self.level_set(position)) / epsilon
        ])
        return grad_phi / np.linalg.norm(grad_phi)

    def collision_response(self, velocity, position):
        """Compute the new velocity after collision."""
        # Get the object's velocity and normal at the collision point
        v_co = self.velocity_function(position)
        n = self.normal(position)
        
        # Relative velocity in the collision object’s frame
        v_rel = velocity - v_co
        vn = np.dot(v_rel, n)
        
        if vn >= 0:
            # No collision (objects are separating)
            return velocity
        
        # Tangential component
        vt = v_rel - n * vn
        if np.linalg.norm(vt) <= self.friction_coefficient * abs(vn):
            # Stick condition
            v_rel_new = np.zeros(3)
        else:
            # Apply dynamic friction
            v_rel_new = vt + self.friction_coefficient * vn * vt / np.linalg.norm(vt)
        
        # Final velocity in world coordinates
        v_new = v_rel_new + v_co
        return v_new
