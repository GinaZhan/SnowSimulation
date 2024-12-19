#version 330 core

in float fragDensity;  // Density from the vertex shader
out vec4 FragColor;    // Final fragment color

void main() {
    // Map density to brightness (adjust scale as needed)
    float brightness = fragDensity;
    // float finalBrightness = min(brightness, 1.0);
    // if (brightness < 1e-3) {
    //     brightness = 0.1;
    // }
    FragColor = vec4(brightness, brightness, brightness, 1.0);

    // FragColor = vec4(1.0, 1.0, 1.0, brightness);  // Use brightness for grayscale color
}
