from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
import numpy as np
from .camera import Camera
import os

class OpenGLRenderer:
    def __init__(self, particle_positions):
        self.particle_positions = particle_positions
        self.shader_program = None
        self.vao = None
        self.vbo = glGenBuffers(1)
        self.camera = None

    def initialize(self):
        # Compile shaders
        vertex_shader = self.load_shader("shaders/vertex_shader.glsl", GL_VERTEX_SHADER)
        fragment_shader = self.load_shader("shaders/fragment_shader.glsl", GL_FRAGMENT_SHADER)
        self.shader_program = compileProgram(vertex_shader, fragment_shader)

        # Set up VAO and VBO for particles
        self.vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, self.particle_positions.nbytes, self.particle_positions, GL_STATIC_DRAW)

        # Configure vertex attributes
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        # Enable depth testing for 3D rendering
        glEnable(GL_DEPTH_TEST)

        # Set up camera
        self.camera = Camera(
            position=[0.0, 0.0, 20.0],
            target=[0.0, 0.0, 0.0],
            up=[0.0, 1.0, 0.0],
            fov=45.0,
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
        
        # Check buffer size and reallocate if necessary
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        current_buffer_size = glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE)
        if self.particle_positions.nbytes > current_buffer_size:
            print("Reallocating buffer due to size mismatch")
            glBufferData(GL_ARRAY_BUFFER, self.particle_positions.nbytes, None, GL_DYNAMIC_DRAW)
        
        # Update buffer data
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.particle_positions.nbytes, self.particle_positions.ravel())
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def render(self):
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

        # Draw particles
        glBindVertexArray(self.vao)
        print("Render position: ", self.particle_positions)
        glDrawArrays(GL_POINTS, 0, len(self.particle_positions))
        glBindVertexArray(0)
