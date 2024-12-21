#version 330 core

in float fragDensity;  // Density from the vertex shader
out vec4 FragColor;    // Final fragment color

void main() {
    // Map density to brightness (adjust scale as needed)
    float brightness = fragDensity;
    FragColor = vec4(brightness, brightness, brightness, 1.0);

    // Do not adjust brightness using density
    // FragColor = vec4(1.0, 1.0, 1.0, 1.0);
}
