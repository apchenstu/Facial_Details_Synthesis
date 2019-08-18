#version 330 core
layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 tex_cord;

out VS_OUT {
    vec2 tex_cord;
} vs_out;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

void main()
{

    vs_out.tex_cord = tex_cord;

    gl_Position = projection * view * model * vec4(pos, 1.0);

}
