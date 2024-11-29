# import sys
# import os

# # Add the parent directory to the Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from src.simulation.grid import *
# from src.simulation.particles import *

# def test_compute_weight():
#     p = Particle((1, 1, 1), velocity=(0, -1, 0), mass=1)
#     node = GridNode((1, 3, 1))
#     weight = node.compute_weight(p)
#     print(weight)

# test_compute_weight()

import warp as wp

wp.init()

# @wp.kernel
# def test_kernel(arr: wp.array(dtype=wp.vec3)):
#     tid = wp.tid()
#     arr[tid] = wp.vec3(1.0, 2.0, 3.0)

# # Create an array of wp.vec3
# data = wp.array(
#     data=[wp.vec3(0.0, 0.0, 0.0) for _ in range(10)],
#     dtype=wp.vec3,
#     device="cuda",
# )

# # Launch the kernel
# wp.launch(test_kernel, dim=10, inputs=[data])

# # Retrieve the results
# result = data.numpy()
# print(result)

@wp.kernel
def modify_array(arr: wp.array(dtype=float)):
    tid = wp.tid()
    arr[tid] += 10.0

# Initialize a Warp array
arr = wp.array([1.0, 2.0, 3.0], dtype=float, device="cuda")

# Launch kernel
wp.launch(kernel=modify_array, dim=len(arr), inputs=[arr])

# Print the modified array
print(arr.numpy())

# @wp.kernel
# def simple_kernel(a: wp.array(dtype=wp.vec3),
#                   b: wp.array(dtype=wp.vec3),
#                   c: wp.array(dtype=float)):

#     # get thread index
#     tid = wp.tid()

#     # load two vec3s
#     x = a[tid]
#     y = b[tid]

#     # compute the dot product between vectors
#     r = wp.dot(x, y)

#     # write result back to memory
#     c[tid] = r