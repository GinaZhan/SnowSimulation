import numpy as np
import warp as wp

@wp.func
def collision_response(
    velocity: wp.vec3,
    object_velocity: wp.vec3,
    normal: wp.vec3,
    friction_coefficient: float) -> wp.vec3:
    
    # Relative velocity
    v_rel = velocity - object_velocity

    # Compute normal component of velocity
    vn = wp.dot(v_rel, normal)
    if vn >= 0.0:
        return velocity  # No collision

    # print(velocity)
    # Compute tangential velocity
    vt = v_rel - normal * vn
    if wp.length(vt) <= (-1.0) * friction_coefficient * vn:
        # Stick condition
        v_rel_new = wp.vec3(0.0, 0.0, 0.0)
    else:
        # Dynamic friction
        v_rel_new = vt + friction_coefficient * vn * vt / wp.length(vt)

    # Final velocity
    return v_rel_new + object_velocity

@wp.kernel
def apply_collision_kernel(
    velocities: wp.array(dtype=wp.vec3),
    level_set_values: wp.array2d(dtype=float),  # Now 2D: (num_particles, num_objects)
    object_normals: wp.array2d(dtype=wp.vec3),  # Now 2D: (num_particles, num_objects)
    object_velocities: wp.array2d(dtype=wp.vec3),  # Now 2D: (num_particles, num_objects)
    friction_coefficients: wp.array(dtype=float)):  # Still 1D: (num_objects)

    tid = wp.tid()  # Particle index
    velocity = velocities[tid]

    # Check for collisions with each object
    num_objects = level_set_values.shape[1]
    for obj_idx in range(num_objects):
        normal = object_normals[tid, obj_idx]
        object_velocity = object_velocities[tid, obj_idx]
        friction_coefficient = friction_coefficients[obj_idx]

        # Use precomputed level set values for collision detection
        if level_set_values[tid, obj_idx] <= 0.0:  # Colliding if φ ≤ 0
            velocity = collision_response(velocity, object_velocity, normal, friction_coefficient)
            break  # Handle only first collision

    # Update particle velocity
    velocities[tid] = velocity

@wp.kernel
def apply_grid_collision_kernel(
    velocities: wp.array(dtype=wp.vec3),
    level_set_values: wp.array2d(dtype=float),  # Now 2D: (num_particles, num_objects)
    object_normals: wp.array2d(dtype=wp.vec3),  # Now 2D: (num_particles, num_objects)
    object_velocities: wp.array2d(dtype=wp.vec3),  # Now 2D: (num_particles, num_objects)
    friction_coefficients: wp.array(dtype=float),
    active: wp.array(dtype=bool)):

    tid = wp.tid()  # grid node index
    velocity = velocities[tid]

    if active[tid]:
        # Check for collisions with each object
        num_objects = level_set_values.shape[1]
        for obj_idx in range(num_objects):
            normal = object_normals[tid, obj_idx]
            object_velocity = object_velocities[tid, obj_idx]
            friction_coefficient = friction_coefficients[obj_idx]

            # Use precomputed level set values for collision detection
            if level_set_values[tid, obj_idx] <= 0.0:  # Colliding if φ ≤ 0
                velocity = collision_response(velocity, object_velocity, normal, friction_coefficient)
                break  # Handle only first collision

    # Update particle velocity
    velocities[tid] = velocity

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
        return self.level_set(position) <= 0.0

    def compute_normal(self, position):
        """
        Compute the unit normal vector n = ∇φ at the position.
        n is always perpendicular to the object surface.
        n points outward from the surface.
        """
        epsilon = 1e-5

        grad_phi_x = (self.level_set(position + wp.vec3(epsilon, 0.0, 0.0)) - self.level_set(position)) / epsilon
        grad_phi_y = (self.level_set(position + wp.vec3(0.0, epsilon, 0.0)) - self.level_set(position)) / epsilon
        grad_phi_z = (self.level_set(position + wp.vec3(0.0, 0.0, epsilon)) - self.level_set(position)) / epsilon

        grad_phi = wp.vec3(grad_phi_x, grad_phi_y, grad_phi_z)
        grad_phi_norm = wp.length(grad_phi)

        # Handle edge case where the gradient norm is zero
        if grad_phi_norm == 0.0:
            return wp.vec3(0.0, 0.0, 0.0)

        return grad_phi / grad_phi_norm
    
    def precompute_for_kernel(self, input_positions, grid_space=0):
        """
        Precompute collision data for a given set of positions.
        """
        num_particles = len(input_positions)
        
        # Use NumPy arrays for modification
        level_set_values_np = np.zeros(num_particles, dtype=np.float32)
        normals_np = np.zeros((num_particles, 3), dtype=np.float32)
        velocities_np = np.zeros((num_particles, 3), dtype=np.float32)

        for i, pos in enumerate(input_positions):
            # if grid_space:
            #     pos = pos * grid_space + 0.5 * grid_space
            level_set_values_np[i] = self.level_set(pos)
            normals_np[i] = self.compute_normal(pos)
            velocities_np[i] = self.velocity_function(pos)

        # Convert to Warp arrays
        level_set_values = wp.array(level_set_values_np, dtype=float, device="cuda")
        normals = wp.array(normals_np, dtype=wp.vec3, device="cuda")
        velocities = wp.array(velocities_np, dtype=wp.vec3, device="cuda")

        return level_set_values, normals, velocities