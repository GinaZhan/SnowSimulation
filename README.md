# Snow Simulation Project Structure

This project aims to simulate snow behavior using the Material Point Method (MPM), GPU acceleration, and OpenGL for rendering. The structure includes key components for simulation, rendering, utilities, and tests.

## Project Structure


## Folder and File Descriptions

### 1. `src/`
This folder contains the main source code of the project.

- **`simulation/`**: Contains all the logic for the MPM snow simulation.
  - `mpm_solver.py`: Implements the Material Point Method (MPM) for simulating snow.
  - `particles.py`: Manages the particle system (Lagrangian particles).
  - `grid.py`: Implements the Eulerian grid for snow interactions with the environment.
  - `snow_material.py`: Defines the material properties of snow (e.g., plasticity, stickiness).
  
- **`rendering/`**: Handles OpenGL-based real-time rendering of the snow simulation.
  - `opengl_renderer.py`: Implements the OpenGL renderer to visualize the snow simulation.
  - `shaders/`: Contains the vertex and fragment shaders for snow rendering.
    - `particle.vert`: Vertex shader for rendering snow particles.
    - `particle.frag`: Fragment shader for snow particles.
    - `snow_frag.frag`: Fragment shader for rendering snow-specific visuals.
  - `camera.py`: Camera control for navigating the scene during rendering.

- **`utils/`**: Contains utility functions for configuration, timing, and data handling.
  - `config.py`: Stores simulation and rendering configuration parameters.
  - `data_loader.py`: Helper for loading and saving simulation data.
  - `timer.py`: Provides performance measurement tools.
  - `video_export.py`: Exports the rendered frames into a video file.

- **`main.py`**: The main entry point for running the simulation and rendering.

### 2. `tests/`
This folder contains unit tests to ensure the simulation and rendering components work correctly.

- `test_mpm_solver.py`: Tests for the MPM solver.
- `test_opengl_renderer.py`: Tests for the OpenGL renderer.
- `test_particles.py`: Tests for the particle system.

### 3. `resources/`
Optional resources such as 3D models and textures for the simulation.

- **`models/`**: Optional 3D models to simulate collisions (e.g., walls, objects).
- **`textures/`**: Optional textures for the ground, snow, and other visual elements.
- **`shaders/`**: Additional shaders for visual effects, if needed.

### 4. `video_output/`
This folder will store the generated video frames and final simulation output.

## Additional Files

- **`README.md`**: Documentation for how to run the project, set it up, and use the simulation.
- **`requirements.txt`**: List of Python dependencies required to run the project.
