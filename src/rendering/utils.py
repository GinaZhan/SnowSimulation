import numpy as np
import warp as wp
import time
import pickle
import glfw
import os
import OpenGL.GL as gl
from PIL import Image

def save_rendered_frame(frames_dir, frame_id, window):
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