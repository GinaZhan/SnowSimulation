from .particles import ParticleSystem
from .grid import Grid
from .stress_forces import compute_stress_forces

import warp as wp


class MPMSolver:
    def __init__(self, particle_system: ParticleSystem, grid: Grid, time_step: float):
        self.particle_system = particle_system
        self.grid = grid
        self.time_step = time_step

        # after rasterization
        # grid.setup_particle_density_volume(p)

    def rasterize_particles_to_grid(self):
        # Transfer mass and velocity from particles to grid
        self.grid.clear()
        for particle in self.particle_system.particles:
            self.grid.transfer_mass_and_velocity(particle)

    def compute_forces(self):
        # Compute stress-based forces using the constitutive relations
        compute_stress_forces(self.particle_system, self.grid)

    def update_grid_velocities(self):
        # Update the velocity of grid nodes based on forces and time step
        for node in self.grid.nodes:
            node.update_velocity(self.time_step)

    def update_particle_velocities_and_positions(self):
        # Update particle velocities and positions based on grid data
        for particle in self.particle_system.particles:
            particle.update_velocity(self.grid, self.time_step)
            particle.update_position(self.time_step)

    def run_time_step(self):
        # Full time step of MPM: Rasterize, Compute Forces, Update Grid, Update Particles
        self.rasterize_particles_to_grid()
        self.compute_forces()
        self.update_grid_velocities()
        self.update_particle_velocities_and_positions()