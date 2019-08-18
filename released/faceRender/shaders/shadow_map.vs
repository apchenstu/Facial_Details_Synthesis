#version 330 core
layout (location = 0) in vec3 pos;

uniform mat4 light_pv;
uniform mat4 model;

void main()
{
    gl_Position = light_pv * model * vec4(pos, 1.0);
}