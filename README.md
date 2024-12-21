# MPM Snow Simulation Project: Realistic Snow Dynamics

This project implements a Material Point Method (MPM) for simulating snow behaviors, inspired by the paper ["A Material Point Method for Snow Simulation"](https://disneyanimation.com/publications/a-material-point-method-for-snow-simulation/) by Stomakhin et al. The goal is to simulate various snow interactions, including falling, piling, collisions, and different stickiness levels.

## Simulation Output

The simulation output videos are saved in the `video_output` directory.

## Features Implemented

### Material Point Method (MPM) for Snow Simulation

The Material Point Method (MPM) is a hybrid approach combining particle and grid-based methods to simulate materials such as snow. Below is a step-by-step description of the MPM process for snow simulation:

1. **Rasterize Particle Data to the Grid**\
   The first step is transferring data from particles to the grid. Mass is transferred using a weighting function, ensuring that the mass at each grid node is calculated based on the contributions of nearby particles. Similarly, velocity is transferred to the grid. However, to preserve momentum conservation, normalized weights are used to compute grid velocities.

2. **Compute Particle Volumes and Densities (First Timestep Only)**\
To calculate forces accurately, the simulation requires an estimate of each particle's initial volume. The grid's density is estimated and then mapped back to the particles, allowing for volume computation. This step is critical for initializing the simulation.

3. **Compute Grid Forces**\
Grid forces are calculated based on stress tensors and the deformation gradients of particles. These forces account for the material's behavior and its interaction with the environment.

4. **Update Grid Velocities**\
Grid velocities are updated using the forces computed in the previous step. This step uses equations derived from physical principles to ensure accurate simulation dynamics.

5. **Handle Grid-Based Body Collisions**\
Grid-based collisions are handled by modifying grid velocities to prevent particles from penetrating boundaries or objects.

6. **Solve the Linear System for Time Integration**\
The simulation can use either an implicit or explicit method to update velocities. Although I initially attempted an implicit method for better stability, it failed due to unresolved issues in the implementation. As a result, the simulation currently uses the explicit method, which updates velocities directly. The implicit method code is still present in the codebase but disabled. In the future, if I have the opportunity to debug and resolve the implicit implementation, I plan to enable it for improved robustness.

7. **Update Deformation Gradient**\
The deformation gradient for each particle is updated to account for elastic and plastic deformations. This step captures material properties such as stiffness and plasticity.

8. **Update Particle Velocities**\
Particle velocities are updated using a combination of PIC (Particle-In-Cell) and FLIP (Fluid Implicit Particle) methods. A weighting parameter balances the contributions of these methods to control numerical dissipation.

9. **Handle Particle-Based Body Collisions**\
Particle-based collisions are handled to ensure that particles do not pass through boundaries or other objects. This complements the grid-based collision handling.

10. **Update Particle Positions**\
Particle positions are updated based on their velocities, completing the timestep. This step ensures the particles move according to the simulation dynamics.

11. **Visualization**\
OpenGL-based rendering for visualizing snow behaviors.

## Resources Used

1. **Papers and Reference Code**:
   - ["A Material Point Method for Snow Simulation"](https://disneyanimation.com/publications/a-material-point-method-for-snow-simulation/).
   - ["Snow Implicit Math"](https://github.com/Azmisov/snow/blob/master/papers/snow_implicit_math.pdf).
   - ["SnowSim Code"](https://github.com/Azmisov/snow/tree/master/SnowSim)
   - ["Snow Code"](https://github.com/JAGJ10/Snow/tree/master)

2. **Libraries and Tools**:
   - **NVIDIA Warp**: For GPU-accelerated computations. (https://github.com/NVIDIA/warp)
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
│   │   ├── collision_object.py
│   │   ├── constants.py
│   │   ├── snow_object.py
│   ├── rendering/
│   │   ├── opengl_renderer.py
│   │   ├── camera.py
│   │   ├── utils.py
│   │   ├── shaders/
│   │       ├── fragment_shader.glsl
│   │       ├── vertex_shader.glsl
│   ├── main.py
├── README.md
├── requirements.txt

```

## Instructions to Compile and Test

### Prerequisites

1. **Hardware**:
   - A system with an NVIDIA GPU (GTX 1050 Ti or higher recommended).

2. **Software**:
   - Python 3.10 (Python 3.8+ may suffice, but compatibility is not guaranteed)
   - Dependencies specified in `requirements.txt`.

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/GinaZhan/SnowSimulation.git
   cd SnowSimulation
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv env
   source env/bin/activate  # On Windows, use 'env\Scripts\activate'
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Simulation

1. Create snow objects and load them into a `ParticleSystem` object in `main.py` using the functions provided in `snow_object.py`. The `NEW_SIMULATION` constant determines whether to start a new simulation or reload a previous state. The current grid size is 1.5 * 1.5 * 1.5 m^3. The directory to save the frames can be modified in `main.py`.

2. (Optional) Adjust parameters in `constants.py` for different scenarios, such as snow stickiness or grid_space.

3. Three collision objects are provided in `mpm_solver.py` and can be chosen by commenting and uncommenting.

4. (Optional) In `fragment_shader.glsl`, choose whether mapping density to brightness.

5. (Optional) Camera can be adjusted in `opengl_render.py`. Currently it looks at the center of the grid.

6. Run the simulation script:
   ```
   cd src
   python main.py
   ```

7. To create a video from the saved frames, run the following command:
   ```
   ffmpeg -framerate [framerate] -i simulation_frames/frame_%04d.png -c:v libx264 -r 30 -pix_fmt yuv420p simulation_video.mp4
   ```

8. Sometimes, intermittent issues may occur where frames are not saved, causing the process to be interrupted. In that case, change the constant `NEW_SIMULATION` to False in `main.py` to reload the state and continue.

## Future Work

- Adding implicit velocity update for better stability with larger timesteps.
- Enhancing visualization with more realistic snow rendering.

## Acknowledgments

Thanks to the authors of the referenced papers and open-source contributors for their valuable resources and tools that made this project possible.