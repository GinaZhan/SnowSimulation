from simulation.mpm_solver import MPMSolver
from simulation.grid import Grid
from simulation.constants import GRID_SPACE
from simulation.snow_object import SnowObject, create_particle_system
from rendering.opengl_render import OpenGLRenderer
from rendering.utils import save_rendered_frame, save_simulation_state, load_simulation_state

import glfw
import os



# if True, start a new simulation; if False, reload past simulation states and continue
NEW_SIMULATION = True

# directory that stores the frames
frames_dir = "simulation_results/falling_0.001_density"
os.makedirs(frames_dir, exist_ok=True)


def setup_simulation():
    # Create Grid
    grid = Grid(size=100, grid_space=GRID_SPACE)    # 1.5m * 1.5m * 1.5m grid space

    # Initialize SnowObject
    snow_creator = SnowObject(particle_diameter=0.0072, target_density=400)

    # Create a snowball
    snowball = snow_creator.create_snowball(
        radius=0.1,               # Radius of the snowball
        center=[0.4, 1.0, 0.75],       # Center position
        velocity=[0, 0, 0]        # Initial velocity
    )

    snowcube1 = snow_creator.create_snowcube(
        side_length=0.2,         # Side length of the cube
        center=[1.0, 0.2, 0.75],      # Center position
        velocity=[0, 0, 0]      # Initial velocity
    )

    snowcube2 = snow_creator.create_snowcube(
        side_length=0.2,         # Side length of the cube
        center=[1.1, 0.7, 0.75],      # Center position
        velocity=[0, 0, 0]      # Initial velocity
    )

    snowcube3 = snow_creator.create_snowcube(
        side_length=0.2,         # Side length of the cube
        center=[1.0, 1.2, 0.75],      # Center position
        velocity=[0, 0, 0]      # Initial velocity
    )

    particle_system = create_particle_system([snowball, snowcube1, snowcube2, snowcube3])

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

if NEW_SIMULATION:
    solver.run_initial_step()
    frame_id = 0
else:
    frame_id = load_simulation_state(solver) + 1

particle_positions = solver.particle_system.positions.numpy()
particle_densities = solver.particle_system.densities.numpy()

# Initialize renderer
renderer = OpenGLRenderer(particle_positions, particle_densities)
renderer.initialize()

while not glfw.window_should_close(window):
    glfw.poll_events()

    solver.run_time_step()

    # Update particle positions in the shared array
    particle_positions = solver.particle_system.positions.numpy()
    renderer.update_particle_positions(particle_positions)
    renderer.render()
    save_rendered_frame(frames_dir, frame_id, window)
    if frame_id % 20 == 0:
        save_simulation_state(frame_id, solver)
    frame_id += 1
    glfw.swap_buffers(window)

glfw.terminate()