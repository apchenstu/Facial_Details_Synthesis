#version 330 core
layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 tex_cord;
layout (location = 3) in vec3 tgt;

out VS_OUT {
    vec2 tex_cord;
    vec3 normal;
    vec3 tgt;
    vec3 pos;
} vs_out;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;


void main(void)
{
    vs_out.tex_cord = tex_cord;
    vs_out.normal = vec3(normalize(model*vec4(normal,0.0)));
    vs_out.tgt = vec3(normalize(model*vec4(tgt,0.0)));
    vs_out.pos = vec3(model*vec4(pos,0.0));

    gl_Position = projection * view * model * vec4(pos, 1.0);
}
