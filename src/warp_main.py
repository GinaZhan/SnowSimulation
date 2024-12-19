from simulation.mpm_solver import MPMSolver
from simulation.particles import ParticleSystem
from simulation.grid import Grid
from simulation.constants import GRID_SPACE
from rendering.opengl_render import OpenGLRenderer

import glfw
import numpy as np
import warp as wp
from PIL import Image
import os
import OpenGL.GL as gl
import time
import pickle

# if True, start a new simulation; if False, reload past simulation states and continue
NEW_SIMULATION = True

# directory that stores the frames
frames_dir = "simulation_results/2_snowballs_0.001"
os.makedirs(frames_dir, exist_ok=True)

def save_rendered_frame(frame_id, window):
    # Get the width and height of the window
    width, height = glfw.get_framebuffer_size(window)

    # Read pixels from the OpenGL framebuffer
    gl_data = gl.glReadPixels(0, 0, width, height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)

    # Convert to a NumPy array
    image_data = np.frombuffer(gl_data, dtype=np.uint8).reshape((height, width, 4))

    # Flip vertically because OpenGL's origin is bottom-left
    image_data = np.flip(image_data, axis=0)

    # Save the frame as an image
    frame_path = os.path.join(frames_dir, f"frame_{frame_id:04d}.png")
    image = Image.fromarray(image_data, "RGBA")
    
    for attempt in range(10):
        try:
            image.save(frame_path)
            return
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(3)
    # return
    raise RuntimeError(f"Failed to save frame{frame_id:04d} after 10 attempts.")

def save_simulation_state(frame_id, solver, file_path="simulation_state.pkl"):
    state = {
        "frame_id": frame_id,
        "num_particles": solver.particle_system.num_particles,
        "positions": solver.particle_system.positions.numpy(),
        "velocities": solver.particle_system.velocities.numpy(),
        "masses": solver.particle_system.masses.numpy(),
        "initial_volumes": solver.particle_system.initial_volumes.numpy(),
        "F_E": solver.particle_system.F_E.numpy(),
        "F_P": solver.particle_system.F_P.numpy(),
        "deformation_gradient": solver.particle_system.deformation_gradient.numpy(),
        "densities": solver.particle_system.densities.numpy(),
        "stresses": solver.particle_system.stresses.numpy(),
        # Add grid-specific state here if needed
    }
    with open(file_path, "wb") as f:
        pickle.dump(state, f)
    print(f"Simulation state saved at frame {frame_id}")

def load_simulation_state(solver, file_path="simulation_state.pkl"):
    with open(file_path, "rb") as f:
        state = pickle.load(f)
    solver.particle_system.num_particles = state["num_particles"]
    solver.particle_system.positions = wp.array(state["positions"], dtype=wp.vec3, device="cuda")
    solver.particle_system.velocities = wp.array(state["velocities"], dtype=wp.vec3, device="cuda")
    solver.particle_system.masses = wp.array(state["masses"], dtype=float, device="cuda")
    solver.particle_system.initial_volumes = wp.array(state["initial_volumes"], dtype=float, device="cuda")
    solver.particle_system.F_E = wp.array(state["F_E"], dtype=wp.mat33, device="cuda")
    solver.particle_system.F_P = wp.array(state["F_P"], dtype=wp.mat33, device="cuda")
    solver.particle_system.deformation_gradient = wp.array(state["deformation_gradient"], dtype=wp.mat33, device="cuda")
    solver.particle_system.densities = wp.array(state["densities"], dtype=float, device="cuda")
    solver.particle_system.stresses = wp.array(state["stresses"], dtype=wp.mat33, device="cuda")
    print(f"Simulation state loaded from frame {state['frame_id']}")
    return state["frame_id"]


def setup_simulation():
    # num_particles = 20000    # There should be 4-8 particles in one GridNode

    # Initialize particle system and grid
    # particle_system = ParticleSystem(num_particles)
    grid = Grid(size=64, grid_space=GRID_SPACE)
    # grid = Grid(size=64)

    radius = 0.3
    # positions = wp.zeros((num_particles, 3), dtype=wp.vec3, device="cuda")
    # velocities = wp.vec3(10.0, 0.0, 0.0)
    pos_list = []
    # vel_list = []

    # cube_num_particles = int(num_particles/3)
    # # Initialize particle positions and velocities
    # for i in range(cube_num_particles):
    #     pos = np.random.uniform(-radius, radius, 3)
    #     # # This while loop determines snowcube or snowball
    #     # while np.linalg.norm(pos) > radius:
    #     #     pos = np.random.uniform(-radius, radius, 3)
    #     pos[0] += 3.1
    #     # pos[1] += 3.15
    #     pos[1] += 1.0
    #     pos[2] += 3
    #     pos_list.append(pos)

    # for i in range(cube_num_particles):
    #     pos = np.random.uniform(-radius, radius, 3)
    #     # while np.linalg.norm(pos) > radius:
    #     #     pos = np.random.uniform(-radius, radius, 3)
    #     pos[0] += 2.7
    #     pos[1] += 3
    #     # pos[1] += 0.8
    #     pos[2] += 3
    #     pos_list.append(pos)

    # for i in range(cube_num_particles):
    #     pos = np.random.uniform(-radius, radius, 3)
    #     # while np.linalg.norm(pos) > radius:
    #     #     pos = np.random.uniform(-radius, radius, 3)
    #     pos[0] += 3
    #     pos[1] += 5
    #     # pos[1] += 0.8
    #     pos[2] += 3
    #     pos_list.append(pos)

    num_snowball = int(num_particles / 2)

    for i in range(num_snowball):
        pos = np.random.uniform(-radius, radius, 3)
        while np.linalg.norm(pos) > radius:
            pos = np.random.uniform(-radius, radius, 3)
        pos[0] += 0.5
        pos[1] += 3
        # pos[1] += 0.8
        pos[2] += 3
        pos_list.append(pos)

    for i in range(num_snowball):
        pos = np.random.uniform(-radius, radius, 3)
        while np.linalg.norm(pos) > radius:
            pos = np.random.uniform(-radius, radius, 3)
        pos[0] += 6.0
        pos[1] += 3
        # pos[1] += 0.8
        pos[2] += 3
        pos_list.append(pos)

    first_snowball = [wp.vec3(5, 0, 0) for _ in range(num_snowball)]
    second_snowball = [wp.vec3(-5, 0, 0) for _ in range(num_snowball)]

    # Concatenate the lists
    vec_list = first_snowball + second_snowball

    # Convert the list to a wp.array
    velocities = wp.array(vec_list, dtype=wp.vec3)
    

    positions = wp.array([wp.vec3(*p) for p in pos_list], dtype=wp.vec3, device="cuda")
    
    # Initialize particle system and grid
    particle_system = ParticleSystem(num_particles, positions, velocities)
    # particle_system = ParticleSystem(num_particles, positions)

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

# ffmpeg -framerate 30 -i simulation_frames/frame_%04d.png -c:v libx264 -r 30 -pix_fmt yuv420p simulation_video.mp4