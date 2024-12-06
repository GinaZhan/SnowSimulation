#version 330 core

layout(location = 0) in vec3 position;  // Particle position
layout(location = 1) in float density;  // Particle density

out float fragDensity;  // Pass density to the fragment shader

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float maxdensity;

void main() {
    fragDensity = density / maxdensity;  // Pass the density to the fragment shader
    // fragDensity = maxdensity;  // Pass the density to the fragment shader
    gl_Position = projection * view * model * vec4(position, 1.0);
}
