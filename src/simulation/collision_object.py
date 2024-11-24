import numpy as np

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

    def collision_response(self, velocity, position):
        """Compute the new velocity for snow after collision."""
        # Get the object's velocity and normal at the collision point
        v_co = self.velocity_function(position)
        n = self.normal(position)
        
        
        # Relative velocity in the collision object’s frame
        v_rel = velocity - v_co
        

        # The normal component is the part of the velocity vector that is parallel to the surface's normal vector (scalar)
        vn = np.dot(v_rel, n)
        if vn != 0:
            print("Normal: ", n)
            print("v_rel: ", v_rel)
            print("vn: ", vn)

        if vn >= 0:
            # No collision (objects are separating or the snow is moving away from the surface)
            return velocity
        # Otherwise, snow is going though the surface inward

        # Tangential component is the part of the velocity vector that lies parallel to the surface
        # ||vt|| > 0, slides on surface; otherwise, stick on surface
        vt = v_rel - n * vn
        if np.linalg.norm(vt) <= (-1) * self.friction_coefficient * vn:
            # Stick condition
            print("Sticky")
            v_rel_new = np.zeros(3)
        else:
            # Apply dynamic friction
            print("Dynamic")
            v_rel_new = vt + self.friction_coefficient * vn * vt / np.linalg.norm(vt)
        
        # Final velocity in world coordinates
        v_new = v_rel_new + v_co
        return v_new
