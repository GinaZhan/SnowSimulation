# import warp as wp

# wp.init()
# # main.py

from simulation.mpm_solver import MPMSolver
from simulation.particles import ParticleSystem, Particle
from simulation.grid import Grid

import glfw
from rendering.opengl_render import OpenGLRenderer
import numpy as np

def setup_simulation():
    # Initialize particle system and grid
    particle_system = ParticleSystem()
    grid = Grid(size=64)

    # Example: Adding a single particle to the system
    # particle = Particle(position=[0.5, 0.5, 0.5], velocity=[0.0, 0.0, 0.0], mass=1.0, volume=1.0)

    num_particles = 10
    radius = 1
    for _ in range(num_particles):
        pos = np.random.uniform(-radius, radius, 3)
        while np.linalg.norm(pos) > radius:
            pos = np.random.uniform(-radius, radius, 3)
        vel = np.zeros(3)
        particle = Particle(position=pos, velocity=vel, mass=1.0)
        particle_system.add_particle(particle)

    # Initialize the solver with time step
    solver = MPMSolver(particle_system, grid)
    
    return solver

# def run_simulation():
#     solver = setup_simulation()
    
#     # Run the simulation for a number of steps
#     for step in range(100):  # Example: 100 time steps
#         solver.run_time_step()

# if __name__ == "__main__":
#     run_simulation()



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

particle_positions = np.array([p.position for p in solver.particle_system.particles], dtype=np.float32)
print(particle_positions)

# Initialize renderer
renderer = OpenGLRenderer(particle_positions)
renderer.initialize()
# print("Render Steup finished!")

# Main render loop
# while not glfw.window_should_close(window):
#     glfw.poll_events()

#     # Update particle positions here if needed (e.g., call your simulation update)
#     renderer.render()

#     glfw.swap_buffers(window)

# solver.run_initial_step()
FIRST = True

while not glfw.window_should_close(window):
    glfw.poll_events()

    if FIRST:
        solver.run_initial_step()
        FIRST = False
    else:
    # Run one simulation timestep
        solver.run_time_step()
    # print(particle_positions)

    # Update particle positions in the shared array
    for i, particle in enumerate(solver.particle_system.particles):
        particle_positions[i] = particle.position
        # print("Actual particle position:", particle.position)

    # Render the updated particle positions
    renderer.update_particle_positions(particle_positions)
    renderer.render()

    glfw.swap_buffers(window)

glfw.terminate()