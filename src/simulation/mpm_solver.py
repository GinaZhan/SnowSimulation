from .particles import *
from .grid import Grid
# from .stress_forces import compute_stress_forces
from .collision_object import CollisionObject
from .constants import *

# import numpy as np
import warp as wp


class MPMSolver:
    def __init__(self, particle_system: ParticleSystem, grid: Grid):
        self.particle_system = particle_system
        self.grid = grid

        self.collision_objects = [
            # Floor
            CollisionObject(
                level_set=lambda x: x[1] - 0.5,  # Floor at y = 0.5
                velocity_function=lambda x: wp.vec3(0.0, 0.0, 0.0),  # Static floor
                friction_coefficient=0.5
            ),
            # # # Wall
            # CollisionObject(
            #     level_set=lambda x: 6.0 - x[0],  # Wall at x = 6.0
            #     velocity_function=lambda x: wp.vec3(0.0, 0.0, 0.0),  # Static wall
            #     friction_coefficient=0.5
            # ),
            # # Slide
            # CollisionObject(
            #     level_set=lambda x: x[1] - (-0.5 * x[0] + 3.0),  # Slide: y = -0.5 * x + 3
            #     velocity_function=lambda x: wp.vec3(0.0, 0.0, 0.0),  # Static slide
            #     friction_coefficient=0.3  # Low friction for sliding
            # )
        ]

    def rasterize_particles_to_grid(self):
        # Transfer mass and velocity from particles to grid
        self.grid.clear()
        # Step 1
        self.grid.transfer_mass_and_velocity(self.particle_system)
        # print("Step 1")

    def compute_forces(self):
        # Compute stress-based forces using the constitutive relations
        # Step 3
        self.grid.compute_grid_forces(self.particle_system)
        # print("Step 3")


    def update_grid_velocities(self):
        # Update the velocity of grid nodes based on forces and time step
        # Step 4
        self.grid.update_grid_velocity_star()
        # print("Step 4")
        # Step 5
        self.grid.apply_collisions(self.collision_objects)
        # print("Step 5")
        # Step 6
        self.grid.explicit_update_velocity()
        # print("Step 6")
        # self.grid.implicit_update_velocity(self.particle_system)

    def update_particle_velocities_and_positions(self):
        # Update particle velocities and positions based on grid data
        # Step 7
        self.particle_system.update_deformation_gradients(self.grid)
        # print("Step 7")
        # Step 8
        # print("Before Updating Velocities: ", self.particle_system.velocities)
        self.particle_system.update_velocity(self.grid)
        print("After Updating Velocities: ", self.particle_system.velocities)
        # print("Step 8")
        # Step 9
        self.particle_system.apply_collisions(self.collision_objects)
        # print("Before Updating Position: ", self.particle_system.positions)
        # print("Step 9")
        # Step 10
        self.particle_system.update_position()
        # print("After Updating Position: ", self.particle_system.positions)
        # print("Step 10")

    def run_time_step(self):
        # Full time step of MPM: Rasterize, Compute Forces, Update Grid, Update Particles
        self.rasterize_particles_to_grid()
        self.compute_forces()
        self.update_grid_velocities()
        self.update_particle_velocities_and_positions()

    def run_initial_step(self):
        self.rasterize_particles_to_grid()
        self.grid.setup_particle_density_volume(self.particle_system)
        self.compute_forces()
        self.update_grid_velocities()
        self.update_particle_velocities_and_positions()

        # for step in range(num_steps):
        #     self.run_time_step()

        #     if render_callback:
        #         render_callback()

    def render_callback():
        pass