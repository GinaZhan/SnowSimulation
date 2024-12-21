import warp as wp
from .constants import *
from .collision_object import *

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
def polar_decomposition(F: wp.mat33):
    U = wp.mat33()
    sigma = wp.vec3()
    V = wp.mat33()
    wp.svd3(F, U, sigma, V)

    # Compute rotation matrix (U @ V^T)
    R = U @ wp.transpose(V)

    # Compute stretch matrix (V @ diag(sigma) @ V^T)
    S = V @ wp.mat33(sigma[0], 0, 0, 0, sigma[1], 0, 0, 0, sigma[2]) @ wp.transpose(V)

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
    J_E = wp.determinant(F_E_local)
    J_P = wp.determinant(F_P_local)

    # Compute Lamé parameters with hardening
    exp_factor = wp.exp(alpha * (1.0 - J_P))
    mu = mu_0 * exp_factor
    lambda_ = lambda_0 * exp_factor

    # Perform SVD for polar decomposition
    U = wp.mat33()
    sigma = wp.vec3()
    V = wp.mat33()
    wp.svd3(F_E_local, U, sigma, V)
    R_E = U @ wp.transpose(V)

    # Compute stress tensor
    if use_cauchy:
        # dpsi_dF_E = 2.0 * mu * (F_E_local - R_E) + lambda_ * (J_E - 1.0) * J_E * wp.transpose(wp.inverse(F_E_local))
        # J = J_E * J_P
        # stress[tid] = (1.0 / J) * dpsi_dF_E @ wp.transpose(F_E_local)
        
        temp = 2.0 * mu * (F_E_local - R_E) @ wp.transpose(F_E_local) + lambda_ * (J_E - 1.0) * J_E * wp.identity(3, dtype=wp.float32)
        J = J_E * J_P
        stress[tid] = (1.0 / J) * temp
    else:
        stress[tid] = (
            2.0 * mu * (F_E_local - R_E) @ wp.transpose(F_E_local)
            + lambda_ * (J_E - 1.0) * J_E * wp.identity(3, dtype=wp.float32)
        )

@wp.kernel
def compute_stress_tensor_debug_kernel(
    F_E: wp.array(dtype=wp.mat33),
    F_P: wp.array(dtype=wp.mat33),
    mu_0: float,
    lambda_0: float,
    alpha: float,
    stress: wp.array(dtype=wp.mat33),
    use_cauchy: int,  # 1 for Cauchy, 0 for Piola-Kirchhoff
    debug_flags: wp.array(dtype=int)):  # Debug array to record issues

    tid = wp.tid()

    F_E_local = F_E[tid]
    F_P_local = F_P[tid]

    # Initialize debug flag
    debug_flags[tid] = 0

    # Compute determinants
    J_E = wp.determinant(F_E_local)
    J_P = wp.determinant(F_P_local)

    if wp.isnan(J_E) or wp.isnan(J_P) or wp.isinf(J_E) or wp.isinf(J_P):
        debug_flags[tid] = 1  # Flag determinant issue
        return

    # Compute Lamé parameters with hardening
    exp_factor = wp.exp(alpha * (1.0 - J_P))
    mu = mu_0 * exp_factor
    lambda_ = lambda_0 * exp_factor

    if wp.isnan(mu) or wp.isnan(lambda_):
        debug_flags[tid] = 2  # Flag Lamé parameter issue
        return

    # Perform SVD for polar decomposition
    U = wp.mat33()
    sigma = wp.vec3()
    V = wp.mat33()
    wp.svd3(F_E_local, U, sigma, V)
    R_E = U @ wp.transpose(V)

    if wp.isnan(sigma[0]) or wp.isnan(sigma[1]) or wp.isnan(sigma[2]):
        debug_flags[tid] = 3  # Flag SVD issue
        return

    # Compute stress tensor
    if use_cauchy:
        temp = 2.0 * mu * (F_E_local - R_E) @ wp.transpose(F_E_local) + lambda_ * (J_E - 1.0) * J_E * wp.identity(3, dtype=wp.float32)
        J = J_E * J_P
        
        if J_E <= 0.0 or wp.isnan(J_E):
            debug_flags[tid] = 8  # Invalid determinant for F_E
            return
        if J_P <= 0.0 or wp.isnan(J_P):
            debug_flags[tid] = 9  # Invalid determinant for F_P
            return
        if J <= 0.0:
            debug_flags[tid] = 5  # Flag invalid determinant
            return
        if wp.isnan(J):
            debug_flags[tid] = 7

        stress[tid] = (1.0 / J) * temp
    else:
        dpsi_dF_E = (
            2.0 * mu * (F_E_local - R_E) @ wp.transpose(F_E_local)
            + lambda_ * (J_E - 1.0) * J_E * wp.identity(3, dtype=wp.float32)
        )
        if wp.isnan(dpsi_dF_E[0, 0]):
            debug_flags[tid] = 6  # Flag Piola-Kirchhoff stress computation issue
            return
        stress[tid] = dpsi_dF_E


@wp.kernel
def update_deformation_gradient_kernel(
    particle_positions: wp.array(dtype=wp.vec3),
    particle_F_E: wp.array(dtype=wp.mat33),
    particle_F_P: wp.array(dtype=wp.mat33),
    particle_deformation_gradient: wp.array(dtype=wp.mat33),
    particle_polar_r: wp.array(dtype=wp.mat33),
    particle_polar_s: wp.array(dtype=wp.mat33),
    grid_positions: wp.array(dtype=wp.vec3),
    grid_velocities: wp.array(dtype=wp.vec3),
    grid_size: int,
    grid_space: float,
    timestep: float,
    crit_compress: float,
    crit_stretch: float):

    tid = wp.tid()

    # Get particle position
    particle_pos = particle_positions[tid]

    # Initialize velocity gradient
    rv_np1 = wp.mat33(0.0)

    # Compute grid center index for this particle
    grid_center = wp.vec3(
        wp.floor(particle_pos[0] / grid_space),
        wp.floor(particle_pos[1] / grid_space),
        wp.floor(particle_pos[2] / grid_space),
    )

    # Loop over nearby grid nodes
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                grid_idx = wp.vec3(
                    grid_center[0] + float(i),
                    grid_center[1] + float(j),
                    grid_center[2] + float(k),
                )

                # Ensure the grid index is within bounds
                if (grid_idx[0] < 0 or grid_idx[0] >= grid_size or
                    grid_idx[1] < 0 or grid_idx[1] >= grid_size or
                    grid_idx[2] < 0 or grid_idx[2] >= grid_size):
                    continue

                node_idx = int(grid_idx[0]) * grid_size*grid_size + int(grid_idx[1]) * grid_size + int(grid_idx[2])

                # Compute weight gradient
                dx = (particle_pos[0] - grid_positions[node_idx][0]*grid_space - 0.5 * grid_space) / grid_space
                dy = (particle_pos[1] - grid_positions[node_idx][1]*grid_space - 0.5 * grid_space) / grid_space
                dz = (particle_pos[2] - grid_positions[node_idx][2]*grid_space - 0.5 * grid_space) / grid_space

                weight_gradient = wp.vec3(
                    (1.0 / grid_space) * N_prime(dx) * N(dy) * N(dz),
                    (1.0 / grid_space) * N(dx) * N_prime(dy) * N(dz),
                    (1.0 / grid_space) * N(dx) * N(dy) * N_prime(dz),
                )

                # Accumulate velocity gradient
                rv_np1 += wp.outer(grid_velocities[node_idx], weight_gradient)

    # Update F_E^{n+1}_p
    F_Epn1 = (wp.identity(3, dtype=wp.float32) + timestep * rv_np1) @ particle_F_E[tid]

    # Update F^{n+1}_p
    F_np1 = (wp.identity(3, dtype=wp.float32) + timestep * rv_np1) @ particle_deformation_gradient[tid]

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

    plastic_deformation = False
    for i in range(3):
        if sigma[i] != sigma_clamped[i]:
            plastic_deformation = True

    # Update F_E and F_P
    if plastic_deformation:
        particle_polar_r[tid] = U @ wp.transpose(V)  # Rotation matrix
        particle_polar_s[tid] = V @ wp.diag(sigma_clamped) @ wp.transpose(V)  # Stretch matrix
        particle_F_E[tid] = U @ wp.diag(sigma_clamped) @ wp.transpose(V)
        particle_F_P[tid] = wp.transpose(V) @ wp.diag(1.0 / sigma_clamped) @ wp.transpose(U) @ F_np1
    else:
        particle_F_E[tid] = F_Epn1

    # Update total deformation gradient
    particle_deformation_gradient[tid] = particle_F_E[tid] @ particle_F_P[tid]


@wp.kernel
def update_particle_velocity_kernel(
    particle_positions: wp.array(dtype=wp.vec3),
    particle_velocities: wp.array(dtype=wp.vec3),
    grid_positions: wp.array(dtype=wp.vec3),
    grid_velocities: wp.array(dtype=wp.vec3),
    grid_new_velocities: wp.array(dtype=wp.vec3),
    grid_size: int,
    grid_space: float,
    alpha: float):

    tid = wp.tid()

    # Get particle position
    particle_pos = particle_positions[tid]

    # Initialize PIC and FLIP velocities
    velocity_PIC = wp.vec3(0.0, 0.0, 0.0)
    velocity_FLIP = particle_velocities[tid]

    # Compute grid center index for this particle
    grid_center = wp.vec3(
        wp.floor(particle_pos[0] / grid_space),
        wp.floor(particle_pos[1] / grid_space),
        wp.floor(particle_pos[2] / grid_space),
    )

    # Loop over nearby grid nodes
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                grid_idx = wp.vec3(
                    grid_center[0] + float(i),
                    grid_center[1] + float(j),
                    grid_center[2] + float(k),
                )

                # Ensure the grid index is within bounds
                if (grid_idx[0] < 0 or grid_idx[0] >= grid_size or
                    grid_idx[1] < 0 or grid_idx[1] >= grid_size or
                    grid_idx[2] < 0 or grid_idx[2] >= grid_size):
                    continue

                node_idx = int(grid_idx[0]) * grid_size*grid_size + int(grid_idx[1]) * grid_size + int(grid_idx[2])

                # Compute weight
                dx = (particle_pos[0] - grid_positions[node_idx][0]*grid_space - 0.5 * grid_space) / grid_space
                dy = (particle_pos[1] - grid_positions[node_idx][1]*grid_space - 0.5 * grid_space) / grid_space
                dz = (particle_pos[2] - grid_positions[node_idx][2]*grid_space - 0.5 * grid_space) / grid_space

                weight = N(dx) * N(dy) * N(dz)

                # Update PIC and FLIP velocities
                velocity_PIC += grid_new_velocities[node_idx] * weight
                velocity_FLIP += (grid_new_velocities[node_idx] - grid_velocities[node_idx]) * weight

    # Blend PIC and FLIP velocities
    particle_velocities[tid] = (1.0 - alpha) * velocity_PIC + alpha * velocity_FLIP

    # Apply small velocity threshold
    for d in range(3):
        if wp.abs(particle_velocities[tid][d]) < 1e-8:
            particle_velocities[tid][d] = 0.0

@wp.kernel
def update_positions_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    timestep: float):

    tid = wp.tid()  # Thread ID for the current particle

    positions[tid] += velocities[tid] * timestep

@wp.kernel
def update_predicted_positions_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    timestep: float,
    predicted_positions: wp.array(dtype=wp.vec3)):

    tid = wp.tid()  # Thread ID for the current particle
    predicted_positions[tid] = positions[tid] + velocities[tid] * timestep

class ParticleSystem:
    def __init__(self, num_particles, positions, velocities=None):   # TODO: velocities are initialized to 0
        self.num_particles = num_particles

        # Particle properties
        self.positions = positions

        if velocities is None:
            # Initialize velocities to zero if not provided
            self.velocities = wp.zeros(num_particles, dtype=wp.vec3, device="cuda")
        elif isinstance(velocities, wp.vec3):
            # Use the provided wp.vec3 to initialize all velocities
            self.velocities = wp.array([velocities] * num_particles, dtype=wp.vec3, device="cuda")
        elif isinstance(velocities, wp.array) and velocities.dtype == wp.vec3:
            if len(velocities) != num_particles:
                raise ValueError("The length of the velocities array must match num_particles.")
            # Directly use the provided array
            self.velocities = velocities
        else:
            raise ValueError("Velocities must be a wp.vec3 or None.")
        
        self.masses = wp.ones(num_particles, dtype=float, device="cuda")
        # self.masses = wp.full(num_particles, 0.4, dtype=float, device="cuda")
        self.initial_volumes = wp.ones(num_particles, dtype=float, device="cuda")

        # Deformation gradients
        identity_matrix = wp.mat33(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        )
        self.F_E = wp.array([identity_matrix] * num_particles, dtype=wp.mat33, device="cuda")
        self.F_P = wp.array([identity_matrix] * num_particles, dtype=wp.mat33, device="cuda")
        self.deformation_gradient = wp.array([identity_matrix] * num_particles, dtype=wp.mat33, device="cuda")
        self.densities = wp.zeros(num_particles, dtype=float, device="cuda")
        self.stresses = wp.zeros(num_particles, dtype=wp.mat33, device="cuda")

        # Material properties
        self.mu_0 = MU
        self.lambda_0 = LAMBDA
        self.alpha = ALPHA

        self.polar_r = wp.array([identity_matrix] * num_particles, dtype=wp.mat33, device="cuda")
        self.polar_s = wp.array([identity_matrix] * num_particles, dtype=wp.mat33, device="cuda")

    def initialize_particle(self, index, position, velocity, mass):
        self.positions[index] = wp.vec3(*position)
        self.velocities[index] = wp.vec3(*velocity)
        self.masses[index] = mass
        self.initial_volumes[index] = 1.0
        self.F_E[index] = wp.identity(3, dtype=wp.float32)
        self.F_P[index] = wp.identity(3, dtype=wp.float32)

    def compute_stress_tensors(self, use_cauchy=1):
        """
        Compute stress tensors for all particles.
        """
        # Step 3

        debug_flags = wp.zeros(self.num_particles, dtype=int, device="cuda")

        # Launch debug kernel
        wp.launch(
            kernel=compute_stress_tensor_debug_kernel,
            dim=self.num_particles,
            inputs=[
                self.F_E,
                self.F_P,
                self.mu_0,
                self.lambda_0,
                self.alpha,
                self.stresses,
                use_cauchy,
                # 0,
                debug_flags,
            ],
        )

        # Copy debug flags to CPU and inspect
        debug_flags_cpu = debug_flags.numpy()
        if np.any(debug_flags_cpu > 0):
            print("Debug issues detected in compute_stress_tensor_kernel:")
            for tid, flag in enumerate(debug_flags_cpu):
                if flag > 0:
                    print(f"Particle {tid} - Issue Code: {flag}")

    def update_deformation_gradients(self, grid):
        # Step 7
        wp.launch(
            kernel=update_deformation_gradient_kernel,
            dim=self.num_particles,
            inputs=[
                self.positions,
                self.F_E,
                self.F_P,
                self.deformation_gradient,
                self.polar_r,
                self.polar_s,
                grid.positions,
                grid.velocities,
                grid.size,
                grid.grid_space,
                TIMESTEP,
                CRIT_COMPRESS,
                CRIT_STRETCH
            ],
        )

    def update_velocity(self, grid, alpha=0.95):
        # Step 8 - Update particle velocities
        
        wp.launch(
            kernel=update_particle_velocity_kernel,
            dim=self.num_particles,
            inputs=[
                self.positions,
                self.velocities,
                grid.positions,
                grid.velocities,
                grid.new_velocities,
                grid.size,
                grid.grid_space,
                alpha,
            ]
        )

    def apply_collisions(self, collision_objects):
        """
        Apply collisions to particles using precomputed data from collision objects.
        """
        num_particles = self.num_particles
        num_objects = len(collision_objects)

        # Initialize 2D Warp arrays for collision data
        level_set_values_np = np.zeros((num_particles, num_objects), dtype=np.float32)
        normals_np = np.zeros((num_particles, num_objects, 3), dtype=np.float32)
        velocities_np = np.zeros((num_particles, num_objects, 3), dtype=np.float32)
        friction_coefficients_np = np.zeros(num_objects, dtype=np.float32)

        # Estimate future positions based on current velocities
        predicted_positions = wp.zeros_like(self.positions)
        # print("predicted positions length", len(predicted_positions))
        wp.launch(
            kernel=update_predicted_positions_kernel,  # Custom kernel to compute predicted positions
            dim=num_particles,
            inputs=[self.positions, self.velocities, TIMESTEP, predicted_positions],
        )

        for i, obj in enumerate(collision_objects):
            lv, n, v = obj.precompute_for_kernel(predicted_positions.numpy())
            # print("lv length", len(lv))
            level_set_values_np[:, i] = lv.numpy()
            normals_np[:, i, :] = n.numpy()
            velocities_np[:, i, :] = v.numpy()
            friction_coefficients_np[i] = obj.friction_coefficient

        # Flatten the arrays for Warp
        level_set_values = wp.array(level_set_values_np.reshape(num_particles, num_objects), dtype=float, device="cuda")
        normals = wp.array(normals_np.reshape(num_particles, num_objects, 3), dtype=wp.vec3, device="cuda")
        velocities = wp.array(velocities_np.reshape(num_particles, num_objects, 3), dtype=wp.vec3, device="cuda")
        friction_coefficients = wp.array(friction_coefficients_np, dtype=float, device="cuda")

        wp.launch(
            kernel=apply_collision_kernel,
            dim=self.num_particles,
            inputs=[
                self.velocities,
                level_set_values,
                normals,
                velocities,
                friction_coefficients,
            ],
        )

    def update_position(self):
        # Step 10 - Update particle positions
        wp.launch(
            kernel=update_positions_kernel,
            dim=self.num_particles,
            inputs=[
                self.positions,
                self.velocities,
                TIMESTEP
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