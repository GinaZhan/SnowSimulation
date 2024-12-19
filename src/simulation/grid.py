import numpy as np
import warp as wp

from .particles import ParticleSystem
from .constants import *
from .collision_object import apply_grid_collision_kernel

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
    
@wp.func
def cofactor_3x3(m: wp.mat33) -> wp.mat33:
    """
    Compute the cofactor matrix of a 3x3 matrix.
    """
    return wp.mat33(
        # Row 0
        m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1],
        -(m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0]),
        m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0],

        # Row 1
        -(m[0, 1] * m[2, 2] - m[0, 2] * m[2, 1]),
        m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0],
        -(m[0, 0] * m[2, 1] - m[0, 1] * m[2, 0]),

        # Row 2
        m[0, 1] * m[1, 2] - m[0, 2] * m[1, 1],
        -(m[0, 0] * m[1, 2] - m[0, 2] * m[1, 0]),
        m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0],
    )

@wp.func
def frobenius_inner_product(m1: wp.mat33, m2: wp.mat33) -> float:
    """
    Compute the Frobenius inner product of two 3x3 matrices.
    """
    result = 0.0
    for i in range(3):
        for j in range(3):
            result += m1[i, j] * m2[i, j]
    return result

@wp.func
def delta_force_3d(u: wp.vec3, weight_gradient: wp.vec3, F_E: wp.mat33, polar_r: wp.mat33, polar_s: wp.mat33, volume: float, mu: float, lambda_: float, matrix_epsilon: float, timestep: float) -> wp.vec3:
    """
    Compute the change in force deltaForce in 3D.

    Args:
        u: The velocity vector, the residual in this case.
        weight_gradient: The gradient of the weight function.
        F_E: Elastic deformation gradient.
        polar_r: Rotation matrix from polar decomposition.
        polar_s: Stretch matrix from polar decomposition.
        volume: Particle volume.
        ...

    Returns:
        deltaForce as a 3D vector.
    """
    # Compute delta(Fe) = timestep * outer_product(u, weight_grad) * def_elastic
    del_elastic = TIMESTEP * wp.outer(u, weight_gradient) @ F_E

    # Check if delta(Fe) is negligible
    if wp.length(del_elastic[0]) < matrix_epsilon and wp.length(del_elastic[1]) < matrix_epsilon and wp.length(del_elastic[2]) < matrix_epsilon:
        return wp.vec3(0.0, 0.0, 0.0)

    # Compute R^T * delta(Fe) - delta(Fe)^T * R (skew-symmetric part)
    skew = wp.transpose(polar_r) @ del_elastic - wp.transpose(del_elastic) @ polar_r

    # Extract components of the skew-symmetric matrix
    y = wp.vec3(skew[2, 1] - skew[1, 2], skew[0, 2] - skew[2, 0], skew[1, 0] - skew[0, 1])

    # Solve for x in MS + SM = R^T * delta(Fe) - delta(Fe)^T * R
    x = wp.vec3(
        y[0] / (polar_s[1, 1] + polar_s[2, 2]),
        y[1] / (polar_s[0, 0] + polar_s[2, 2]),
        y[2] / (polar_s[0, 0] + polar_s[1, 1]),
    )

    # Compute delta(R) = R * skew(x)
    del_rotate = polar_r @ wp.mat33(
        0.0, -x[2], x[1],
        x[2], 0.0, -x[0],
        -x[1], x[0], 0.0
    )

    # Compute cofactor matrix of Fe (JF^-T)
    cofactor = cofactor_3x3(F_E)

    # Compute delta(cofactor) for Fe
    del_cofactor = cofactor_3x3(del_elastic)

    # Compute "A" term for the force
    # Co-rotational term: 2 * mu * (delta(Fe) - delta(R))
    Ap = (del_elastic - del_rotate) * (2.0 * mu)

    # Primary contour term
    cofactor *= frobenius_inner_product(del_elastic, cofactor)  # TODO: frobeniusInnerProduct
    del_cofactor *= (wp.determinant(F_E) - 1.0)
    cofactor += del_cofactor
    cofactor *= lambda_
    Ap += cofactor

    # Compute deltaForce
    return volume * (Ap @ (wp.transpose(F_E) @ weight_gradient))


@wp.kernel
def initialize_positions_kernel(
    positions: wp.array(dtype=wp.vec3),
    grid_size: int):
    # Grid position is actually grid index
    
    tid = wp.tid()  # Thread ID for the current grid node

    # Compute 3D grid indices from flat tid
    x = tid // (grid_size * grid_size)
    y = (tid % (grid_size * grid_size)) // grid_size
    z = tid % grid_size

    # Set the position of the grid node
    positions[tid] = wp.vec3(float(x), float(y), float(z))

@wp.kernel
def update_velocity_star_kernel(
    velocities: wp.array(dtype=wp.vec3),
    velocities_star: wp.array(dtype=wp.vec3),
    forces: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    gravity: wp.vec3,
    timestep: float,
    active: wp.array(dtype=wp.bool)):
    
    tid = wp.tid()
    if active[tid]:
        velocities_star[tid] = velocities[tid] + (forces[tid] / mass[tid] + gravity) * timestep

@wp.kernel
def clear_kernel(
    grid_mass: wp.array(dtype=float),
    grid_velocities: wp.array(dtype=wp.vec3),
    grid_velocities_star: wp.array(dtype=wp.vec3),
    grid_new_velocities: wp.array(dtype=wp.vec3),
    grid_forces: wp.array(dtype=wp.vec3),
    active: wp.array(dtype=bool),
    imp_active: wp.array(dtype=bool)
):
    tid = wp.tid()
    grid_mass[tid] = 0.0
    grid_velocities[tid] = wp.vec3(0.0, 0.0, 0.0)
    grid_velocities_star[tid] = wp.vec3(0.0, 0.0, 0.0)
    grid_new_velocities[tid] = wp.vec3(0.0, 0.0, 0.0)
    grid_forces[tid] = wp.vec3(0.0, 0.0, 0.0)
    active[tid] = False
    # imp_active[tid] = False

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
    grid_size: int,
    active: wp.array(dtype=bool),
    weight_epsilon: float
    ):

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
                if weight > weight_epsilon:
                    wp.atomic_add(grid_mass, node_idx, particle_mass * weight)
                    wp.atomic_add(grid_velocities, node_idx, particle_vel * particle_mass * weight)
                    active[node_idx] = True

@wp.kernel
def normalize_grid_velocity_kernel(
    # Step 1 - Part 2
    grid_velocities: wp.array(dtype=wp.vec3),
    grid_mass: wp.array(dtype=float),
    active: wp.array(dtype=bool)):

    tid = wp.tid()

    if active[tid]:
        grid_velocities[tid] = grid_velocities[tid] / grid_mass[tid]
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
                particle_density += grid_masses[node_idx] / (grid_space * grid_space * grid_space) * weight # TODO: weight_epsilon?
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
    active: wp.array(dtype=bool)
):
    tid = wp.tid()
    if active[tid]:
        grid_new_velocities[tid] = grid_velocities_star[tid]

@wp.kernel
def implicit_initialize_r_kernel(
    new_velocities: wp.array(dtype=wp.vec3),
    r: wp.array(dtype=wp.vec3),
    active: wp.array(dtype=bool),
    imp_active: wp.array(dtype=bool),
    err: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    imp_active[tid] = active[tid]
    if imp_active[tid]:
        r[tid] = new_velocities[tid]
        err[tid] = wp.vec3(1.0, 1.0, 1.0)

@wp.kernel
def implicit_initialize_rEr_kernel(
    node_new_velocities: wp.array(dtype=wp.vec3),
    node_r: wp.array(dtype=wp.vec3),
    node_p: wp.array(dtype=wp.vec3),
    node_Er: wp.array(dtype=wp.vec3),
    node_rEr: wp.array(dtype=float),
    imp_active: wp.array(dtype=bool),
):
    tid = wp.tid()
    if imp_active[tid]:
        node_r[tid] = node_new_velocities[tid] - node_Er[tid]
        node_p[tid] = node_r[tid]
        node_rEr[tid] = wp.dot(node_r[tid], node_Er[tid])

@wp.kernel
def implicit_initialize_Ep_kernel(
    node_Ep: wp.array(dtype=wp.vec3),
    node_Er: wp.array(dtype=wp.vec3),
    imp_active: wp.array(dtype=bool),
):
    tid = wp.tid()
    if imp_active[tid]:
        node_Ep[tid] = node_Er[tid]

@wp.kernel
def implicit_update_velocity_kernel(
    new_velocities: wp.array(dtype=wp.vec3),
    r: wp.array(dtype=wp.vec3),
    Ep: wp.array(dtype=wp.vec3),
    p: wp.array(dtype=wp.vec3),
    rEr: wp.array(dtype=float),
    err: wp.array(dtype=wp.vec3),
    imp_active: wp.array(dtype=bool),
    max_implicit_error: float,
    done: wp.array(dtype=int)  # Shared across threads, 1 means "done", 0 means "not done"
):
    tid = wp.tid()
    if imp_active[tid]:
        div = wp.dot(Ep[tid], Ep[tid])
        if div < 1e-8:
            imp_active[tid] = False
            return
        # alpha = rEr[tid] / div
        alpha = wp.clamp(rEr[tid] / div, -1e4, 1e4)
        err[tid] = alpha * p[tid]

        # Check convergence
        error = wp.length(err[tid])
        if error < max_implicit_error or wp.isnan(error):
            imp_active[tid] = False
            return
        else:
            wp.atomic_min(done, 0, 0)

        new_velocities[tid] += err[tid]
        r[tid] -= alpha * Ep[tid]

@wp.kernel
def implicit_update_gradient_kernel(
    r: wp.array(dtype=wp.vec3),
    Er: wp.array(dtype=wp.vec3),
    p: wp.array(dtype=wp.vec3),
    Ep: wp.array(dtype=wp.vec3),
    rEr: wp.array(dtype=float),
    imp_active: wp.array(dtype=bool),
):
    tid = wp.tid()
    if imp_active[tid]:
        temp = wp.dot(r[tid], Er[tid])
        beta = temp / rEr[tid]
        rEr[tid] = temp
        p[tid] = beta * p[tid] + r[tid]
        Ep[tid] = beta * Ep[tid] + Er[tid]

@wp.kernel
def implicit_recompute_forces_kernel(
    F_Es: wp.array(dtype=wp.mat33),
    polar_rs: wp.array(dtype=wp.mat33),
    polar_ss: wp.array(dtype=wp.mat33),
    particle_volumes: wp.array(dtype=float),
    particle_positions: wp.array(dtype=wp.vec3),
    mu: float,
    lambda_: float,
    matrix_epsilon: float,
    grid_positions: wp.array(dtype=wp.vec3),
    grid_forces: wp.array(dtype=wp.vec3),
    imp_active: wp.array(dtype=bool),
    grid_r: wp.array(dtype=wp.vec3),
    grid_size: int,
    grid_space: float,
    timestep: float):

    tid = wp.tid()

    # Access particle data
    particle_pos = particle_positions[tid]
    particle_volume = particle_volumes[tid]

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

                if imp_active[node_idx]:
                    # Retrieve per-particle properties
                    F_E = F_Es[tid]
                    polar_r = polar_rs[tid]
                    polar_s = polar_ss[tid]

                    # Compute delta force
                    delta_f = delta_force_3d(grid_r[node_idx], weight_gradient, F_E, polar_r, polar_s, particle_volume, mu, lambda_, matrix_epsilon, timestep)
                    delta_f = wp.vec3(
                        wp.clamp(delta_f[0], -1e4, 1e4),
                        wp.clamp(delta_f[1], -1e4, 1e4),
                        wp.clamp(delta_f[2], -1e4, 1e4)
                    )


                    # Accumulate force
                    wp.atomic_add(grid_forces, node_idx, delta_f)

@wp.kernel
def implicit_recompute_grid_Er_kernel(
    imp_active: wp.array(dtype=bool),
    Er: wp.array(dtype=wp.vec3),
    r: wp.array(dtype=wp.vec3),
    implicit_ratio: float,
    timestep: float,
    mass: wp.array(dtype=float),
    force: wp.array(dtype=wp.vec3)):

    tid = wp.tid()
    if imp_active[tid]:
        if mass[tid] < 1e-8:
            imp_active[tid] = False
            return
        else:
            Er[tid] = r[tid] - implicit_ratio * timestep / mass[tid] * force[tid]

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
        self.size = size
        self.grid_space = grid_space

        # Initialize grid properties as arrays
        self.positions = wp.zeros(size**3, dtype=wp.vec3, device="cuda")
        self.velocities = wp.zeros(size**3, dtype=wp.vec3, device="cuda")
        self.new_velocities = wp.zeros(size**3, dtype=wp.vec3, device="cuda")
        self.velocities_star = wp.zeros(size**3, dtype=wp.vec3, device="cuda")
        self.forces = wp.zeros(size**3, dtype=wp.vec3, device="cuda")
        self.mass = wp.zeros(size**3, dtype=float, device="cuda")
        self.active = wp.zeros(size**3, dtype=wp.bool, device="cuda")

        self.r = wp.zeros(size**3, dtype=wp.vec3, device="cuda")
        self.p = wp.zeros(size**3, dtype=wp.vec3, device="cuda")
        self.Er = wp.zeros(size**3, dtype=wp.vec3, device="cuda")
        self.Ep = wp.zeros(size**3, dtype=wp.vec3, device="cuda")
        self.rEr = wp.zeros(size**3, dtype=float, device="cuda")
        self.err = wp.ones(size**3, dtype=wp.vec3, device="cuda")
        self.imp_active = wp.zeros(size**3, dtype=wp.bool, device="cuda")

        wp.launch(
            kernel=initialize_positions_kernel,
            dim=size**3,
            inputs=[
                self.positions,
                size,
            ],
        )

    def clear(self):
        wp.launch(
            kernel=clear_kernel,
            dim=self.size**3,
            inputs=[self.mass, self.velocities, self.velocities_star, self.new_velocities, self.forces, self.active, self.imp_active]
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
                self.active,
                WEIGHT_EPSILON
            ],
        )

        # Normalize velocities after mass and velocity transfer
        wp.launch(
            kernel=normalize_grid_velocity_kernel,
            dim=self.size**3,  # Number of grid nodes
            inputs=[self.velocities, self.mass, self.active],
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
                particle_system.stresses,
                self.positions,
                self.forces,
                self.size,
                self.grid_space,
            ],
        )

    def update_grid_velocity_star(self):
        # Step 4 - Update grid velocity

        wp.launch(
            kernel=update_velocity_star_kernel,
            dim=self.size**3,
            inputs=[
                self.velocities,
                self.velocities_star,
                self.forces,
                self.mass,
                GRAVITY,
                TIMESTEP,
                self.active
            ]
        )

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

        # Flatten the arrays for Warp
        level_set_values = wp.array2d(level_set_values_np.reshape(num_grids, num_objects), dtype=float, device="cuda")
        normals = wp.array2d(normals_np, dtype=wp.vec3, device="cuda")  # Shape: [num_grids, num_objects]
        velocities = wp.array2d(velocities_np, dtype=wp.vec3, device="cuda")  # Shape: [num_grids, num_objects]
        friction_coefficients = wp.array(friction_coefficients_np, dtype=float, device="cuda")

        # Launch the kernel
        wp.launch(
            kernel=apply_grid_collision_kernel,
            dim=num_grids,
            inputs=[
                self.velocities_star,
                level_set_values,
                normals,
                velocities,
                friction_coefficients,
                self.active
            ],
        )

    def explicit_update_velocity(self):
        # Step 6
        wp.launch(
            kernel=explicit_update_velocity_kernel,
            dim=self.size**3,
            inputs=[self.new_velocities, self.velocities_star, self.active],
        )

    def implicit_update_velocity(self, particle_system: ParticleSystem):
        # nan_count = count_nan_values_in_array(self.new_velocities)
        # print(f"Number of NaN values in grid_new_velocities stage 1: {nan_count}")

        wp.launch(
            implicit_initialize_r_kernel, 
            dim=self.size**3,
            inputs=[self.new_velocities, self.r, self.active, self.imp_active, self.err])
        self.recompute_implicit_forces(particle_system)

        # nan_count = count_nan_values_in_array(self.new_velocities)
        # print(f"Number of NaN values in grid_new_velocities stage 2: {nan_count}")

        wp.launch(
            implicit_initialize_rEr_kernel, 
            dim=self.size**3,
            inputs=[self.new_velocities, self.r, self.p, self.Er, self.rEr, self.imp_active])
        self.recompute_implicit_forces(particle_system)

        # nan_count = count_nan_values_in_array(self.new_velocities)
        # print(f"Number of NaN values in grid_new_velocities stage 3: {nan_count}")

        wp.launch(
            implicit_initialize_Ep_kernel, 
            dim=self.size**3,
            inputs=[self.Ep, self.Er, self.imp_active])
        
        # nan_count = count_nan_values_in_array(self.new_velocities)
        # print(f"Number of NaN values in grid_new_velocities stage 4: {nan_count}")

        for i in range(MAX_IMPLICIT_ITERS):
            done = wp.array([1], dtype=int, device="cuda")  # 1 means done, 0 means not done
            wp.launch(implicit_update_velocity_kernel, 
                      dim=self.size**3, 
                      inputs=[self.new_velocities,
                              self.r,
                              self.Ep,
                              self.p,
                              self.rEr,
                              self.err,
                              self.imp_active,
                              MAX_IMPLICIT_ERR,
                              done])
            
            # nan_count = count_nan_values_in_array(self.new_velocities)
            # print(f"Number of NaN values in grid_new_velocities stage 5: {nan_count}")
            print("Iteration ", i)

            done_value = done.numpy()[0]  # Copy the flag to the host
            if done_value == 1:
                print("Converged!")
                break
            self.recompute_implicit_forces(particle_system)

            wp.launch(implicit_update_gradient_kernel, 
                      dim=self.size**3, 
                      inputs=[self.r,
                              self.Er,
                              self.p,
                              self.Ep,
                              self.rEr,
                              self.imp_active])
            
        # nan_count = count_nan_values_in_array(self.new_velocities)
        # print(f"Number of NaN values in grid_new_velocities stage 6: {nan_count}")
        # nan_count = count_nan_values_in_array(self.velocities)
        # print(f"Number of NaN values in grid_velocities stage 6: {nan_count}")

    def recompute_implicit_forces(self, particle_system: ParticleSystem):
        wp.launch(
            implicit_recompute_forces_kernel, 
            dim=particle_system.num_particles,
            inputs=[particle_system.F_E,
                    particle_system.polar_r,
                    particle_system.polar_s,
                    particle_system.initial_volumes,
                    particle_system.positions,
                    particle_system.mu_0,
                    particle_system.lambda_0,
                    MATRIX_EPSILON,
                    self.positions,
                    self.forces,
                    self.imp_active,
                    self.r,
                    self.size,
                    self.grid_space,
                    TIMESTEP
                ],
        )
        
        wp.launch(
            implicit_recompute_grid_Er_kernel, 
            dim=self.size**3,
            inputs=[self.imp_active,
                    self.Er,
                    self.r,
                    IMPLICIT_RATIO,
                    TIMESTEP,
                    self.mass,
                    self.forces
                    ]
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