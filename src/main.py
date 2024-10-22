import warp as wp

wp.init()
# main.py

from simulation.mpm_solver import MPMSolver
from simulation.particles import ParticleSystem, Particle
from simulation.grid import Grid
from simulation.snow_material import SnowMaterial

def setup_simulation():
    # Initialize particle system and grid
    particle_system = ParticleSystem()
    grid = Grid(size=64)

    # Example: Adding a single particle to the system
    particle = Particle(position=[0.5, 0.5, 0.5], velocity=[0.0, 0.0, 0.0], mass=1.0, volume=1.0)
    particle_system.add_particle(particle)

    # Initialize the solver with time step
    solver = MPMSolver(particle_system, grid, time_step=0.01)
    
    return solver

def run_simulation():
    solver = setup_simulation()
    
    # Run the simulation for a number of steps
    for step in range(100):  # Example: 100 time steps
        solver.run_time_step()

if __name__ == "__main__":
    run_simulation()
