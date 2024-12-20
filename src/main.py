from simulation.mpm_solver import MPMSolver
from simulation.particles import ParticleSystem
from simulation.grid import Grid
from simulation.constants import GRID_SPACE
from simulation.snow_object import SnowObject
from rendering.opengl_render import OpenGLRenderer
from rendering.utils import save_rendered_frame, save_simulation_state, load_simulation_state

import glfw
import numpy as np
import warp as wp
import os



# if True, start a new simulation; if False, reload past simulation states and continue
NEW_SIMULATION = True

# directory that stores the frames
frames_dir = "simulation_results/2_snowballs_0.001"
os.makedirs(frames_dir, exist_ok=True)


def setup_simulation():
    # Create Grid
    grid = Grid(size=100, grid_space=GRID_SPACE)    # 1.5m * 1.5m * 1.5m grid space

    # Initialize SnowObject
    snow_creator = SnowObject(particle_diameter=0.0072, target_density=400)

    # Create a snowball
    snowball = snow_creator.create_snowball(
        radius=0.05,               # Radius of the snowball
        center=[0.2, 0.75, 0.75],       # Center position
        velocity=[5, 0, 0]        # Initial velocity
    )

    snowcube = snow_creator.create_snowcube(
        side_length=0.1,         # Side length of the cube
        center=[1.3, 0.75, 0.75],      # Center position
        velocity=[-5, 0, 0]      # Initial velocity
    )

    # Combine snowballs into one particle system
    positions = wp.array(
        np.concatenate([snowball["positions"].numpy(), snowcube["positions"].numpy()]),
        dtype=wp.vec3,
        device="cuda"
    )

    velocities = wp.array(
        np.concatenate([snowball["velocities"].numpy(), snowcube["velocities"].numpy()]),
        dtype=wp.vec3,
        device="cuda"
    )
    num_particles = snowball["num_particles"] + snowcube["num_particles"]

    particle_system = ParticleSystem(num_particles=num_particles, positions=positions, velocities=velocities)

    # Initialize the solver
    solver = MPMSolver(particle_system, grid)
    return solver

# Initialize GLFW
if not glfw.init():
    raise Exception("GLFW could not be initialized")

window = glfw.create_window(800, 800, "Snow Simulation", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW window could not be created")

glfw.make_context_current(window)

solver = setup_simulation()
# print("Solver Steup finished!")
# Generate dummy particle positions for testing
# particle_positions = np.random.uniform(-1, 1, (1000, 3)).astype(np.float32)

if NEW_SIMULATION:
    solver.run_initial_step()
    frame_id = 0
else:
    frame_id = load_simulation_state(solver) + 1

particle_positions = solver.particle_system.positions.numpy()
particle_densities = solver.particle_system.densities.numpy()
print(particle_positions)
print(max(particle_densities))

# print("Grid positions: ", solver.grid.positions)
# Initialize renderer
renderer = OpenGLRenderer(particle_positions, particle_densities)
renderer.initialize()

# FIRST = True


while not glfw.window_should_close(window):
    glfw.poll_events()

    # if FIRST:
    #     print("Run First Timestep")
    #     solver.run_initial_step()
    #     FIRST = False
    # else:
    # Run one simulation timestep
        # solver.run_time_step()
    # print(particle_positions)

    solver.run_time_step()

    # Update particle positions in the shared array
    particle_positions = solver.particle_system.positions.numpy()
    renderer.update_particle_positions(particle_positions)
    renderer.render()
    save_rendered_frame(frame_id, window)
    if frame_id % 20 == 0:
        save_simulation_state(frame_id, solver)
    frame_id += 1
    glfw.swap_buffers(window)

    # if frame_id % 10 == 0:
    #     particle_positions = solver.particle_system.positions.numpy()
    #     renderer.update_particle_positions(particle_positions)
    #     renderer.render()
    #     save_rendered_frame(frame_id, window)

    # if frame_id % 20 == 0:
    #     save_simulation_state(frame_id, solver)
    # frame_id += 1

    # if frame_id % 10 == 0:
    #     glfw.swap_buffers(window)

glfw.terminate()