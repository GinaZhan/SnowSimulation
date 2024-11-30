# import warp as wp

# wp.init()
# # main.py

from simulation.warp_mpm_solver import MPMSolver
from simulation.warp_particles import ParticleSystem
from simulation.warp_grid import Grid
from simulation.constants import GRID_SPACE

import glfw
from rendering.opengl_render import OpenGLRenderer
import numpy as np
import warp as wp

def setup_simulation():
    num_particles = 10000

    # Initialize particle system and grid
    # particle_system = ParticleSystem(num_particles)
    grid = Grid(size=64)

    radius = 1
    # positions = wp.zeros((num_particles, 3), dtype=wp.vec3, device="cuda")
    velocities = wp.zeros(num_particles, dtype=wp.vec3, device="cuda")
    pos_list = []
    # vel_list = []

    # Initialize particle positions and velocities
    for i in range(num_particles):
        pos = np.random.uniform(-radius, radius, 3)
        while np.linalg.norm(pos) > radius:
            pos = np.random.uniform(-radius, radius, 3)
        pos[0] += 11
        pos[1] += 1.2
        pos[2] += 11
        pos_list.append(pos)

    positions = wp.array([wp.vec3(*p) for p in pos_list], dtype=wp.vec3, device="cuda")
    
    # Initialize particle system and grid
    particle_system = ParticleSystem(num_particles, positions)

    # Initialize the solver
    solver = MPMSolver(particle_system, grid)
    return solver

# Initialize GLFW
if not glfw.init():
    raise Exception("GLFW could not be initialized")

window = glfw.create_window(800, 600, "Snow Simulation", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW window could not be created")

glfw.make_context_current(window)

solver = setup_simulation()
# print("Solver Steup finished!")
# Generate dummy particle positions for testing
# particle_positions = np.random.uniform(-1, 1, (1000, 3)).astype(np.float32)

particle_positions = solver.particle_system.positions.numpy()
print(particle_positions)

# Initialize renderer
renderer = OpenGLRenderer(particle_positions)
renderer.initialize()

FIRST = True

while not glfw.window_should_close(window):
    glfw.poll_events()

    if FIRST:
        print("Run First Timestep")
        solver.run_initial_step()
        FIRST = False
    else:
    # Run one simulation timestep
        solver.run_time_step()
    # print(particle_positions)

    # Update particle positions in the shared array
    particle_positions = solver.particle_system.positions.numpy()
    print("Actual particle position: ", particle_positions)

    # Render the updated particle positions
    renderer.update_particle_positions(particle_positions)
    renderer.render()

    glfw.swap_buffers(window)

glfw.terminate()