import numpy as np
import warp as wp

from .warp_particles import ParticleSystem
from .constants import *
from .warp_collision_object import apply_collision_kernel

wp.init()

GRAVITY = wp.vec3(0.0, -9.8, 0.0)  # Gravity vector pointing down (negative Y-axis)

@wp.func
def N(x: float) -> float:
    abs_x = wp.abs(x)
    if abs_x >= 0.0 and abs_x < 1.0:
        result = 0.5 * wp.pow(abs_x, 3.0) - wp.pow(x, 2.0) + 2.0 / 3.0
    elif abs_x >= 1.0 and abs_x < 2.0:
        result = -1.0 / 6.0 * wp.pow(abs_x, 3.0) + wp.pow(x, 2.0) - 2.0 * abs_x + 4.0 / 3.0
    else:
        return 0.0

    # if result < WEIGHT_EPSILON:
    #     return 0.0
    # else:
    #     return result
    return result
    
@wp.func
def N_prime(x: float) -> float:
    abs_x = wp.abs(x)
    if abs_x >= 0.0 and abs_x < 1.0:
        result = 1.5 * abs_x * x - 2.0 * x  # 1.5 * |x| * x + 2.0 * x
        return result
    elif abs_x >= 1.0 and abs_x < 2.0:
        result = -0.5 * abs_x * x + 2.0 * x - 2.0 * x / abs_x  # -0.5 * |x| * x + 2.0 * x - 2.0 * x / |x|
        return result
    else:
        return 0.0

@wp.kernel
def initialize_positions_kernel(
    positions: wp.array(dtype=wp.vec3),
    grid_size: int,
    grid_space: float):
    # Grid position is actually grid index
    
    tid = wp.tid()  # Thread ID for the current grid node

    # Compute 3D grid indices from flat tid
    x = tid // (grid_size * grid_size)
    y = (tid % (grid_size * grid_size)) // grid_size
    z = tid % grid_size

    # Set the position of the grid node
    # positions[tid] = wp.vec3(float(x), float(y), float(z)) * grid_space
    positions[tid] = wp.vec3(float(x), float(y), float(z))

@wp.kernel
def update_velocity_star_kernel(
    velocities: wp.array(dtype=wp.vec3),
    velocities_star: wp.array(dtype=wp.vec3),
    forces: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    gravity: wp.vec3,
    timestep: float):
    
    tid = wp.tid()
    if mass[tid] > 0.0:
        velocities_star[tid] = velocities[tid] + (forces[tid] / mass[tid] + gravity) * timestep

        # velocities_star[tid] = velocities[tid] + gravity * timestep
        # print("Gravity")
        # print(gravity)

# @wp.kernel
# def compute_weights_kernel(
#     particle_positions: wp.array(dtype=wp.vec3),
#     grid_positions: wp.array(dtype=wp.vec3),
#     weights: wp.array(dtype=float),
#     grid_size: int,
#     grid_space: float):
    
#     tid = wp.tid()
#     particle_pos = particle_positions[tid]

#     for i in range(-2, 3):
#         for j in range(-2, 3):
#             for k in range(-2, 3):
#                 x = wp.floor(particle_pos[0] / grid_space) + float(i)
#                 y = wp.floor(particle_pos[1] / grid_space) + float(j)
#                 z = wp.floor(particle_pos[2] / grid_space) + float(k)

#                 # Ensure each coordinate is within valid bounds
#                 if x < 0 or x >= grid_size or y < 0 or y >= grid_size or z < 0 or z >= grid_size:
#                     continue

#                 # Compute flat index from valid 3D indices
#                 node_idx = int(x) * grid_size * grid_size + int(y) * grid_size + int(z)
                
#                 grid_pos = grid_positions[node_idx]
#                 dx = (particle_pos[0] - grid_pos[0]*grid_space - 0.5 * grid_space) / grid_space
#                 dy = (particle_pos[1] - grid_pos[1]*grid_space - 0.5 * grid_space) / grid_space
#                 dz = (particle_pos[2] - grid_pos[2]*grid_space - 0.5 * grid_space) / grid_space
#                 weight = N(dx) * N(dy) * N(dz)
#                 wp.atomic_add(weights, node_idx, weight)

@wp.kernel
def clear_kernel(
    grid_mass: wp.array(dtype=float),
    grid_velocities: wp.array(dtype=wp.vec3),
    grid_velocities_star: wp.array(dtype=wp.vec3),
    grid_new_velocities: wp.array(dtype=wp.vec3),
    grid_forces: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    grid_mass[tid] = 0.0
    grid_velocities[tid] = wp.vec3(0.0, 0.0, 0.0)
    grid_velocities_star[tid] = wp.vec3(0.0, 0.0, 0.0)
    grid_new_velocities[tid] = wp.vec3(0.0, 0.0, 0.0)
    grid_forces[tid] = wp.vec3(0.0, 0.0, 0.0)

@wp.kernel
def transfer_mass_and_velocity_kernel(
    # Step 1 - Rasterize particle data to the grid - Transfer mass and velocity to grid nodes from particles
    particle_positions: wp.array(dtype=wp.vec3),
    particle_velocities: wp.array(dtype=wp.vec3),
    particle_masses: wp.array(dtype=float),
    grid_positions: wp.array(dtype=wp.vec3),
    grid_velocities: wp.array(dtype=wp.vec3),
    grid_mass: wp.array(dtype=float),
    grid_space: float,
    grid_size: int):

    tid = wp.tid()
    particle_pos = particle_positions[tid]
    particle_vel = particle_velocities[tid]
    particle_mass = particle_masses[tid]

    grid_center = wp.vec3(
        wp.floor(particle_pos[0] / grid_space),
        wp.floor(particle_pos[1] / grid_space),
        wp.floor(particle_pos[2] / grid_space)
    )

    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                grid_idx = wp.vec3(grid_center[0] + float(i), grid_center[1] + float(j), grid_center[2] + float(k))

                # Ensure the index is within grid bounds
                if (grid_idx[0] < 0 or grid_idx[0] >= grid_size or
                    grid_idx[1] < 0 or grid_idx[1] >= grid_size or
                    grid_idx[2] < 0 or grid_idx[2] >= grid_size):
                    continue

                node_idx = (
                    int(grid_idx[0]) * (grid_size * grid_size)
                    + int(grid_idx[1]) * grid_size
                    + int(grid_idx[2])
                )
                
                dx = (particle_pos[0] - grid_positions[node_idx][0]*grid_space - 0.5 * grid_space) / grid_space
                dy = (particle_pos[1] - grid_positions[node_idx][1]*grid_space - 0.5 * grid_space) / grid_space
                dz = (particle_pos[2] - grid_positions[node_idx][2]*grid_space - 0.5 * grid_space) / grid_space

                weight = N(dx) * N(dy) * N(dz)

                wp.atomic_add(grid_mass, node_idx, particle_mass * weight)
                wp.atomic_add(grid_velocities, node_idx, particle_vel * particle_mass * weight)

@wp.kernel
def normalize_grid_velocity_kernel(
    # Step 1 - Part 2
    grid_velocities: wp.array(dtype=wp.vec3),
    grid_mass: wp.array(dtype=float)):
    tid = wp.tid()

    # Only normalize if mass is non-zero
    if grid_mass[tid] > 0.0:
        grid_velocities[tid] = grid_velocities[tid] / grid_mass[tid]
        # print("Grid transfer velocities:")
        # print(grid_velocities[tid])
    else:
        grid_velocities[tid] = wp.vec3(0.0, 0.0, 0.0)

@wp.kernel
def setup_particle_density_volume_kernel(
    # Step 2
    particle_positions: wp.array(dtype=wp.vec3),
    particle_masses: wp.array(dtype=float),
    particle_densities: wp.array(dtype=float),
    particle_volumes: wp.array(dtype=float),
    grid_positions: wp.array(dtype=wp.vec3),
    grid_masses: wp.array(dtype=float),
    grid_space: float,
    grid_size: int,
    weight_epsilon: float
):
    tid = wp.tid()
    particle_pos = particle_positions[tid]
    particle_density = float(0.0)  # Explicitly declare as dynamic

    # total_weight = float(0.0)
    # total_mass = float(0.0)
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                grid_idx = wp.vec3(
                    wp.floor(particle_pos[0] / grid_space) + float(i),
                    wp.floor(particle_pos[1] / grid_space) + float(j),
                    wp.floor(particle_pos[2] / grid_space) + float(k),
                )

                if (grid_idx[0] < 0 or grid_idx[0] >= grid_size or
                    grid_idx[1] < 0 or grid_idx[1] >= grid_size or
                    grid_idx[2] < 0 or grid_idx[2] >= grid_size):
                    continue

                node_idx = (
                    int(grid_idx[0]) * (grid_size * grid_size)
                    + int(grid_idx[1]) * grid_size
                    + int(grid_idx[2])
                )

                dx = (particle_pos[0] - grid_positions[node_idx][0]*grid_space - 0.5 * grid_space) / grid_space
                dy = (particle_pos[1] - grid_positions[node_idx][1]*grid_space - 0.5 * grid_space) / grid_space
                dz = (particle_pos[2] - grid_positions[node_idx][2]*grid_space - 0.5 * grid_space) / grid_space

                # if i==0 and j==0 and k==0:
                #     print("dx")
                #     print(dx)
                #     print("dy")
                #     print(dy)
                #     print("dz")
                #     print(dz)

                weight = N(dx) * N(dy) * N(dz)
                # total_weight += weight + 0.01
                # if weight > weight_epsilon:
                    # particle_density += grid_masses[node_idx] / (grid_space * grid_space * grid_space) * weight
                particle_density += grid_masses[node_idx] / (grid_space * grid_space * grid_space) * weight
                # total_mass += grid_masses[node_idx]

    # print("Density total weight")
    # print(total_weight)
    # print("Density total mass")
    # print(total_mass)
    particle_densities[tid] = particle_density
    if particle_density > 0.0:
        particle_volumes[tid] = particle_masses[tid] / particle_density
    else:
        particle_volumes[tid] = 1e-8  # TODO: Or small positive value to avoid divide-by-zero???

@wp.kernel
def compute_grid_forces_kernel(
    # Step 3
    particle_positions: wp.array(dtype=wp.vec3),
    particle_volumes: wp.array(dtype=float),
    stress_tensors: wp.array(dtype=wp.mat33),
    grid_positions: wp.array(dtype=wp.vec3),
    grid_forces: wp.array(dtype=wp.vec3),
    grid_size: int,
    grid_space: float):

    tid = wp.tid()

    # Access particle data
    particle_pos = particle_positions[tid]
    particle_volume = particle_volumes[tid]
    stress_tensor = stress_tensors[tid]

    # Loop over grid nodes within range
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                x = wp.floor(particle_pos[0] / grid_space) + float(i)
                y = wp.floor(particle_pos[1] / grid_space) + float(j)
                z = wp.floor(particle_pos[2] / grid_space) + float(k)

                # Ensure each coordinate is within valid bounds
                if x < 0 or x >= grid_size or y < 0 or y >= grid_size or z < 0 or z >= grid_size:
                    continue

                # Compute flat index from valid 3D indices
                node_idx = int(x) * grid_size * grid_size + int(y) * grid_size + int(z)

                grid_pos = grid_positions[node_idx]

                # Compute weight gradient
                dx = (particle_pos[0] - grid_pos[0]*grid_space - 0.5 * grid_space) / grid_space
                dy = (particle_pos[1] - grid_pos[1]*grid_space - 0.5 * grid_space) / grid_space
                dz = (particle_pos[2] - grid_pos[2]*grid_space - 0.5 * grid_space) / grid_space

                weight_gradient = wp.vec3(
                    (1.0 / grid_space) * N_prime(dx) * N(dy) * N(dz),
                    (1.0 / grid_space) * N(dx) * N_prime(dy) * N(dz),
                    (1.0 / grid_space) * N(dx) * N(dy) * N_prime(dz),
                )

                # Compute force contribution
                force_contribution = -particle_volume * (stress_tensor @ weight_gradient)
                wp.atomic_add(grid_forces, node_idx, force_contribution)


@wp.kernel
def explicit_update_velocity_kernel(
    # Step 6
    grid_new_velocities: wp.array(dtype=wp.vec3),
    grid_velocities_star: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    grid_new_velocities[tid] = grid_velocities_star[tid]

@wp.kernel
def update_predicted_positions_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    timestep: float,
    predicted_positions: wp.array(dtype=wp.vec3),
    grid_space: float):

    tid = wp.tid()  # Thread ID for the current particle
    predicted_positions[tid] = positions[tid] * grid_space + velocities[tid] * timestep

class Grid:
    def __init__(self, size, grid_space=1.0):
        # Initialize a grid with a given size
        # self.nodes = [[GridNode((i, j, k)) for k in range(size)] for i in range(size) for j in range(size)]
        # self.nodes = [[[GridNode((i, j, k)) for k in range(size)] for j in range(size)] for i in range(size)]
        self.nodes = []
        self.size = size
        self.grid_space = grid_space

        # Initialize grid properties as arrays
        self.positions = wp.zeros(size**3, dtype=wp.vec3, device="cuda")
        self.velocities = wp.zeros(size**3, dtype=wp.vec3, device="cuda")
        self.new_velocities = wp.zeros(size**3, dtype=wp.vec3, device="cuda")
        self.velocities_star = wp.zeros(size**3, dtype=wp.vec3, device="cuda")
        self.forces = wp.zeros(size**3, dtype=wp.vec3, device="cuda")
        self.mass = wp.zeros(size**3, dtype=float, device="cuda")

        wp.launch(
            kernel=initialize_positions_kernel,
            dim=size**3,
            inputs=[
                self.positions,
                size,
                grid_space,
            ],
        )

    def clear(self):
        wp.launch(
            kernel=clear_kernel,
            dim=self.size**3,
            inputs=[self.mass, self.velocities, self.velocities_star, self.new_velocities, self.forces]
        )

    def transfer_mass_and_velocity(self, particle_system: ParticleSystem):
        # Step 1 - Rasterize particle data to the grid - Transfer mass and velocity to grid nodes from particles

        # self.clear()    # prevent nodes from collecting velocity, mass, force from previous steps

        wp.launch(
            kernel=transfer_mass_and_velocity_kernel,
            dim=particle_system.num_particles,  # Number of particles
            inputs=[
                particle_system.positions,
                particle_system.velocities,
                particle_system.masses,
                self.positions,
                self.velocities,
                self.mass,
                self.grid_space,
                self.size,
            ],
        )

        # Normalize velocities after mass and velocity transfer
        wp.launch(
            kernel=normalize_grid_velocity_kernel,
            dim=len(self.positions),  # Number of grid nodes
            inputs=[self.velocities, self.mass],
        )

    def setup_particle_density_volume(self, particle_system: ParticleSystem):
        # Step 2 - Compute particle volumes and densities - first timestamp only
        # Here we don't reset node and particle density because this function is only called once at first timestamp
        wp.launch(
            kernel=setup_particle_density_volume_kernel,
            dim=particle_system.num_particles,  # Number of particles
            inputs=[
                particle_system.positions,
                particle_system.masses,
                particle_system.densities,
                particle_system.initial_volumes,
                self.positions,
                self.mass,
                self.grid_space,
                self.size,
                WEIGHT_EPSILON
            ],
        )
    
    def compute_grid_forces(self, particle_system: ParticleSystem):
        # Step 3

        particle_system.compute_stress_tensors()

        wp.launch(
            kernel=compute_grid_forces_kernel,
            dim=particle_system.num_particles,
            inputs=[
                particle_system.positions,
                particle_system.initial_volumes,
                particle_system.stresses,       # This contains nan
                self.positions,
                self.forces,
                self.size,
                self.grid_space,
            ],
        )
        # nan_count = count_nan_values_in_array(self.forces)
        # print(f"Number of NaN values in grid_forces: {nan_count}")

    def update_grid_velocity_star(self):
        # Step 4 - Update grid velocity

        wp.launch(
            kernel=update_velocity_star_kernel,
            dim=self.size**3,
            inputs=[
                self.velocities,
                self.velocities_star,
                self.forces,        # This one has nan value
                self.mass,
                GRAVITY,
                TIMESTEP
            ]
        )

        # print("After Gravity")
        # grid_velocities = self.velocities_star.numpy()

        # # # Find the indices of nonzero vectors
        # nonzero_indices = np.where(np.linalg.norm(grid_velocities, axis=1) > 10)[0]  # Use a small threshold to avoid floating-point noise
        # nonzero_vectors = grid_velocities[nonzero_indices]
        # print("After gravity grid has ", len(nonzero_vectors))

    def apply_collisions(self, collision_objects):
        # Step 5
        """Apply collisions to each grid node's velocity based on collision objects."""

        num_grids = self.size**3
        num_objects = len(collision_objects)

        level_set_values_np = np.zeros((num_grids, num_objects), dtype=np.float32)
        normals_np = np.zeros((num_grids, num_objects, 3), dtype=np.float32)
        velocities_np = np.zeros((num_grids, num_objects, 3), dtype=np.float32)
        friction_coefficients_np = np.zeros(num_objects, dtype=np.float32)

        # Estimate future positions based on current velocities
        predicted_positions = wp.zeros_like(self.positions)
        wp.launch(
            kernel=update_predicted_positions_kernel,  # Custom kernel to compute predicted positions
            dim=num_grids,
            inputs=[self.positions, self.velocities_star, TIMESTEP, predicted_positions, self.grid_space],  # predicted positions are real positions but not indexes
        )

        for i, obj in enumerate(collision_objects):
            lv, n, v = obj.precompute_for_kernel(predicted_positions.numpy())
            level_set_values_np[:, i] = lv.numpy()
            normals_np[:, i, :] = n.numpy()
            velocities_np[:, i, :] = v.numpy()
            friction_coefficients_np[i] = obj.friction_coefficient

        # print("min level set values: ", level_set_values_np.min())

        # Flatten the arrays for Warp
        level_set_values = wp.array2d(level_set_values_np.reshape(num_grids, num_objects), dtype=float, device="cuda")
        # normals = wp.array(normals_np.reshape(num_grids, num_objects, 3), dtype=wp.vec3, device="cuda")
        # velocities = wp.array(velocities_np.reshape(num_grids, num_objects, 3), dtype=wp.vec3, device="cuda")
        normals = wp.array2d(normals_np, dtype=wp.vec3, device="cuda")  # Shape: [num_grids, num_objects]
        velocities = wp.array2d(velocities_np, dtype=wp.vec3, device="cuda")  # Shape: [num_grids, num_objects]
        friction_coefficients = wp.array(friction_coefficients_np, dtype=float, device="cuda")

        # grid_velocities = self.velocities_star.numpy()

        # # # Find the indices of nonzero vectors
        # nonzero_indices = np.where(np.linalg.norm(grid_velocities, axis=1) > 10)[0]  # Use a small threshold to avoid floating-point noise
        # nonzero_vectors = grid_velocities[nonzero_indices]
        # print("Before collision grid has ", len(nonzero_vectors))

        # Launch the kernel
        wp.launch(
            kernel=apply_collision_kernel,
            dim=num_grids,
            inputs=[
                self.velocities_star,
                level_set_values,
                normals,
                velocities,
                friction_coefficients
            ],
        )

        # print("Grid velocity after collision")
        # print(self.velocities_star)
        # grid_velocities = self.velocities_star.numpy()

        # # # Find the indices of nonzero vectors
        # nonzero_indices = np.where(np.linalg.norm(grid_velocities, axis=1) > 10)[0]  # Use a small threshold to avoid floating-point noise
        # nonzero_vectors = grid_velocities[nonzero_indices]
        # print("After collision grid has ", len(nonzero_vectors))

        # nan_count = count_nan_values_in_array(self.velocities_star)
        # print(f"Number of NaN values in grid_velocities_star: {nan_count}")

    def explicit_update_velocity(self):
        # Step 6
        wp.launch(
            kernel=explicit_update_velocity_kernel,
            dim=self.size**3,
            inputs=[self.new_velocities, self.velocities_star],
        )

def count_nan_values_in_array(array: wp.array):
    num_elements = array.shape[0]

    # Create a Warp array to store the count
    nan_count = wp.zeros(1, dtype=int, device="cuda")

    # Launch the kernel
    wp.launch(
        kernel=count_nan_values_kernel,
        dim=num_elements,
        inputs=[array, nan_count],
    )

    # Copy the count back to CPU and return
    return nan_count.numpy()[0]

@wp.kernel
def count_nan_values_kernel(data: wp.array(dtype=wp.vec3), count: wp.array(dtype=int)):
    tid = wp.tid()
    value = data[tid]

    # Check if any component is NaN
    if wp.isnan(value[0]) or wp.isnan(value[1]) or wp.isnan(value[2]):
        wp.atomic_add(count, 0, 1)  # Increment count at index 0