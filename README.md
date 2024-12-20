# MPM Snow Simulation Project

This project implements a Material Point Method (MPM) for simulating snow behaviors, inspired by the paper ["A Material Point Method for Snow Simulation"](https://disneyanimation.com/publications/a-material-point-method-for-snow-simulation/) by Stomakhin et al. The goal is to simulate various snow interactions, including falling, piling, collisions, and different stickiness levels.

## Features Implemented

1. **Basic MPM Framework**:
   - Particle-grid transfer using APIC (Affine Particle-In-Cell).
   - Explicit time-stepping for updating particle positions and velocities.
   
2. **Snow Mechanics**:
   - Elastoplastic deformation modeled using the Cauchy stress tensor.
   - Deformation gradient split into elastic and plastic components (F = F_E * F_P).
   - Support for critical compression and stretch thresholds for plastic flow.

3. **Grid Operations**:
   - Mass and velocity transfers between particles and grid nodes.
   - Stress-based forces and grid velocity updates.

4. **Visualization**:
   - OpenGL-based rendering for visualizing snow behaviors.
   - Options for 3D rendering and parameter tuning during runtime.

## Why Use Explicit Time-Stepping?

- **Simplicity**: Explicit time-stepping is straightforward to implement and debug, making it suitable for initial development.
- **Efficiency for Small Steps**: While explicit methods require smaller time steps for stability, the simplicity of implementation outweighs the performance cost for moderate-scale simulations.
- **Compatibility**: The explicit approach works well with APIC transfers and elastoplastic updates, ensuring accurate results without introducing additional solver complexity.

## Resources Used

1. **Papers and Reference Code**:
   - ["A Material Point Method for Snow Simulation"](https://disneyanimation.com/publications/a-material-point-method-for-snow-simulation/).
   - ["Snow Implicit Math"](https://github.com/Azmisov/snow/blob/master/papers/snow_implicit_math.pdf).
   - ["SnowSim Code"](https://github.com/Azmisov/snow/tree/master/SnowSim)
   - ["Snow Code"](https://github.com/JAGJ10/Snow/tree/master)

2. **Libraries and Tools**:
   - **NVIDIA Warp**: For GPU-accelerated computations.
   - **OpenGL**: For rendering the simulation.
   - **Python**: Primary language for implementation, with dependencies for numerical and graphical processing.

## Project Structure

```
MPM_Snow_Simulation/
├── src/
│   ├── simulation/
│   │   ├── mpm_solver.py
│   │   ├── particles.py
│   │   ├── grid.py
│   │   ├── snow_material.py
│   ├── rendering/
│   │   ├── opengl_renderer.py
│   │   ├── shaders/
│   ├── utilities/
│       ├── config.py
│       ├── data_handler.py
├── tests/
│   ├── test_simulation.py
│   ├── test_rendering.py
├── README.md
├── requirements.txt
├── run_simulation.py
```

## Instructions to Compile and Test

### Prerequisites

1. **Hardware**:
   - A system with an NVIDIA GPU (GTX 1050 Ti or higher recommended).

2. **Software**:
   - Python 3.8+
   - Dependencies specified in `requirements.txt`:
     - numpy
     - pyopengl
     - nvidia-warp
     - pytest (for testing)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/username/MPM_Snow_Simulation.git
   cd MPM_Snow_Simulation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Simulation

1. Adjust snow particles in `main.py`. Three snowballs or snowcubes are provided and can be controlled by commenting it out. The total number of 
particles per grid node can be modified. The constant 'NEW_SIMULATION' controls whether to start a new simulation or reload an old state and keep
simulating from there.

2. Adjust parameters in `constants.py` for different scenarios, such as snow stickiness or timestep.

3. Three collision objects are provided in `mpm_solver.py` and can be chosen by commenting and uncommenting.

4. Run the simulation script:
   ```bash
   python main.py
   ```

5. After the frames are saved under the certain directory, use the following command to create videos:
   ```bash
   ffmpeg -framerate [framerate] -i simulation_frames/frame_%04d.png -c:v libx264 -r 30 -pix_fmt yuv420p simulation_video.mp4
   ```

## Future Work

- Adding implicit time-stepping for better stability with larger time steps.
- Enhancing visualization with more realistic snow rendering.
- Optimizing performance for larger simulations using parallelization.

## Acknowledgments

Thanks to the authors of the referenced papers and open-source contributors for their valuable resources and tools that made this project possible.

