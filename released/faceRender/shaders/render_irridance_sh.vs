#version 330 core
layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 tex_cord;

out VS_OUT {
    vec3 normal;
} vs_out;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

void main()
{
    vs_out.normal = normal;
    gl_Position = vec4(tex_cord.x*2 -1, (1-tex_cord.y)*2 -1, 0.0,  1.0);
}
