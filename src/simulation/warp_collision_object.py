# import numpy as np
import warp as wp

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
        """
        Compute the unit normal vector n = ∇φ at the position.
        n is always perpendicular to the object surface.
        n points outward from the surface.
        """
        epsilon = 1e-5
        grad_phi = np.array([
            (self.level_set(position + np.array([epsilon, 0, 0])) - self.level_set(position)) / epsilon,
            (self.level_set(position + np.array([0, epsilon, 0])) - self.level_set(position)) / epsilon,
            (self.level_set(position + np.array([0, 0, epsilon])) - self.level_set(position)) / epsilon
        ])
        grad_phi_norm = np.linalg.norm(grad_phi)

        # at the center of a sphere
        if grad_phi_norm == 0:
            return np.zeros(3)
        
        return grad_phi / grad_phi_norm

@wp.func
def collision_response(
    velocity: wp.vec3,
    position: wp.vec3,
    object_velocity: wp.vec3,
    normal: wp.vec3,
    friction_coefficient: float) -> wp.vec3:
    
    # Relative velocity
    v_rel = velocity - object_velocity

    # Compute normal component of velocity
    vn = wp.dot(v_rel, normal)
    if vn >= 0.0:
        return velocity  # No collision

    # Compute tangential velocity
    vt = v_rel - normal * vn
    if wp.length(vt) <= (-1) * friction_coefficient * vn:
        # Stick condition
        v_rel_new = wp.vec3(0.0, 0.0, 0.0)
    else:
        # Dynamic friction
        v_rel_new = vt + friction_coefficient * vn * vt / wp.length(vt)

    # Final velocity
    return v_rel_new + object_velocity

@wp.kernel
def apply_collision_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    object_normals: wp.array(dtype=wp.vec3),
    object_velocities: wp.array(dtype=wp.vec3),
    friction_coefficients: wp.array(dtype=float),
    num_objects: int,
    timestep: float):

    tid = wp.tid()
    position = positions[tid]
    velocity = velocities[tid]

    for obj_idx in range(num_objects):
        normal = object_normals[obj_idx]
        object_velocity = object_velocities[obj_idx]
        friction_coefficient = friction_coefficients[obj_idx]

        # Simplified collision check (replace with actual logic)
        if wp.dot(position, normal) <= 0.0:  # Inside or on the object
            velocity = collision_response(velocity, position, object_velocity, normal, friction_coefficient)
            break  # Handle only one collision

    velocities[tid] = velocity