from .particles import *
from .grid import Grid
# from .stress_forces import compute_stress_forces
from .collision_object import CollisionObject
from .constants import *

import numpy as np
import warp as wp


class MPMSolver:
    def __init__(self, particle_system: ParticleSystem, grid: Grid):
        self.particle_system = particle_system
        self.grid = grid

        self.collision_objects = [
            CollisionObject(
                level_set=lambda x: x[2],  # Floor at z = 0
                velocity_function=lambda x: np.zeros(3),  # Static floor
                friction_coefficient=0.5
            )
        ]
        # after rasterization
        # self.grid.setup_particle_density_volume(p)

    def rasterize_particles_to_grid(self):
        # Transfer mass and velocity from particles to grid
        self.grid.clear()
        # Step 1
        self.grid.transfer_mass_and_velocity(self.particle_system.particles)

    def compute_forces(self):
        # Compute stress-based forces using the constitutive relations
        # Step 3
        self.grid.compute_grid_forces(self.particle_system.particles)

    def update_grid_velocities(self):
        # Update the velocity of grid nodes based on forces and time step
        # Step 4
        self.grid.update_grid_velocity_star()
        # Step 5
        self.grid.apply_collisions(self.collision_objects)
        # Step 6
        self.grid.explicit_update_velocity()

    def update_particle_velocities_and_positions(self):
        # Update particle velocities and positions based on grid data
        for particle in self.particle_system.particles:
            # Step 7
            particle.update_deformation_gradient(self.grid, TIMESTEP)
            particle.update_velocity(self.grid)
            particle.apply_collision(self.collision_objects)
            particle.update_position()

    def run_time_step(self):
        # Full time step of MPM: Rasterize, Compute Forces, Update Grid, Update Particles
        self.rasterize_particles_to_grid()
        self.compute_forces()
        self.update_grid_velocities()
        self.update_particle_velocities_and_positions()

    def run_initial_step(self):
        self.rasterize_particles_to_grid()
        self.grid.setup_particle_density_volume(self.particle_system.particles)
        self.compute_forces()
        self.update_grid_velocities()
        self.update_particle_velocities_and_positions()

        # for step in range(num_steps):
        #     self.run_time_step()

        #     if render_callback:
        #         render_callback()

    def render_callback():
        pass