#version 330 core

in vec3 fragPosition;
out vec4 fragColor;

// void main() {
//     fragColor = vec4(1.0, 1.0, 1.0, 1.0); // White color for snow
// }

// void main() {
//     if (fragPosition.y < 5.0) {
//         fragColor = vec4(0.0, 0.0, 1.0, 1.0); // Blue for y < 5
//     } else if (fragPosition.y < 10.0) {
//         fragColor = vec4(0.0, 1.0, 0.0, 1.0); // Green for 5 <= y < 10
//     } else {
//         fragColor = vec4(1.0, 1.0, 1.0, 1.0); // White for y >= 10
//     }
// }

void main() {
    fragColor = vec4(1.0, 1.0, 1.0, 0.5);
}