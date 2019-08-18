#version 330 core
layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 txt_cord;

out vec2 txt_cord_out;


void main()
{
    txt_cord_out = txt_cord;
    gl_Position = vec4(pos, 1.0);
}