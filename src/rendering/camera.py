import numpy as np

class Camera:
    def __init__(self, position, target, up, fov, aspect, near, far):
        self.position = np.array(position, dtype=np.float32)
        self.target = np.array(target, dtype=np.float32)
        self.up = np.array(up, dtype=np.float32)
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far

    def get_view_matrix(self):
        # LookAt matrix
        f = (self.target - self.position)
        f = f / np.linalg.norm(f)
        r = np.cross(f, self.up)
        r = r / np.linalg.norm(r)
        u = np.cross(r, f)

        view = np.identity(4, dtype=np.float32)
        view[0, :3] = r
        view[1, :3] = u
        view[2, :3] = -f
        view[:3, 3] = -self.position @ np.array([r, u, -f])
        return view

    def get_projection_matrix(self):
        # Perspective projection matrix
        t = np.tan(np.radians(self.fov / 2)) * self.near
        r = t * self.aspect

        projection = np.zeros((4, 4), dtype=np.float32)
        projection[0, 0] = self.near / r
        projection[1, 1] = self.near / t
        projection[2, 2] = -(self.far + self.near) / (self.far - self.near)
        projection[2, 3] = -2 * self.far * self.near / (self.far - self.near)
        projection[3, 2] = -1
        return projection
