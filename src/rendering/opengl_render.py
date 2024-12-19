from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
import numpy as np
from .camera import Camera
import os

class OpenGLRenderer:
    def __init__(self, particle_positions, particle_densities):
        self.particle_positions = particle_positions
        self.particle_densities = particle_densities  # Add densities
        self.max_density = np.max(particle_densities)

        # OpenGL handles
        self.shader_program = None
        self.vao = None
        self.vbo = glGenBuffers(1)  # VBO for positions + densities
        self.camera = None

        # print(particle_positions.shape)


    def initialize(self):
        # Compile shaders
        vertex_shader = self.load_shader("shaders/vertex_shader.glsl", GL_VERTEX_SHADER)
        fragment_shader = self.load_shader("shaders/fragment_shader.glsl", GL_FRAGMENT_SHADER)
        self.shader_program = compileProgram(vertex_shader, fragment_shader)

        # Combine positions and densities
        particle_data = np.hstack((self.particle_positions, self.particle_densities.reshape(-1, 1)))

        # Set up VAO and VBO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, particle_data.nbytes, particle_data, GL_STATIC_DRAW)

        # Configure vertex attributes
        stride = 4 * 4  # 4 floats per vertex (3 for position + 1 for density)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))  # Position
        glEnableVertexAttribArray(0)

        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))  # Density
        glEnableVertexAttribArray(1)

        # Unbind VAO and VBO
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)  # Additive blending

        # Enable depth testing
        glDisable(GL_DEPTH_TEST)

        # Initialize camera
        self.camera = Camera(
            position=[3.0, 3.0, 20.0],
            target=[3.0, 3.0, 3.0],
            up=[0.0, 1.0, 0.0],
            fov=20.0,
            aspect=800 / 600,
            near=0.1,
            far=100.0
        )


    def load_shader(self, file_path, shader_type):
        # Get the directory of this script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(current_dir, file_path)

        # Load the shader source from the correct path
        with open(full_path, 'r') as f:
            shader_source = f.read()

        return compileShader(shader_source, shader_type)
    
    def update_particle_positions(self, particle_positions):
        self.particle_positions = np.array(particle_positions, dtype=np.float32)
        particle_data = np.hstack((self.particle_positions, self.particle_densities.reshape(-1, 1)))  # Combine positions and densities

        # Check buffer size and reallocate if necessary
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        current_buffer_size = glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE)
        if particle_data.nbytes > current_buffer_size:
            print("Reallocating buffer due to size mismatch")
            glBufferData(GL_ARRAY_BUFFER, particle_data.nbytes, None, GL_DYNAMIC_DRAW)

        # Update buffer data
        glBufferSubData(GL_ARRAY_BUFFER, 0, particle_data.nbytes, particle_data.ravel())
        glBindBuffer(GL_ARRAY_BUFFER, 0)


    def render(self):
        glClearColor(0.0, 0.3, 0.6, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Use the shader program
        glUseProgram(self.shader_program)

        # Set up view and projection matrices
        view = self.camera.get_view_matrix()
        # print("View Matrix:\n", view)
        projection = self.camera.get_projection_matrix()
        # print("Projection Matrix:\n", projection)
        model = np.identity(4, dtype=np.float32)
        # print("Model Matrix:\n", model)
        glUniformMatrix4fv(glGetUniformLocation(self.shader_program, "model"), 1, GL_TRUE, model)
        glUniformMatrix4fv(glGetUniformLocation(self.shader_program, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(self.shader_program, "projection"), 1, GL_TRUE, projection)

        # print("Max Density: ", self.max_density)
        glUniform1f(glGetUniformLocation(self.shader_program, "maxdensity"), self.max_density)

        # Set point size
        # glPointSize(5)  # Adjust the size of the particles as desired

        # Draw particles
        glBindVertexArray(self.vao)
        print("Render position: ", self.particle_positions)
        glDrawArrays(GL_POINTS, 0, len(self.particle_positions))
        glBindVertexArray(0)
