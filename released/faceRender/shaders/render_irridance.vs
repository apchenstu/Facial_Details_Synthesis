#version 330 core
layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 tex_cord;

out VS_OUT {
    vec3 frag_pos;
    vec3 normal;
    vec2 tex_cord;
    vec4 fragpos_lightspace;
} vs_out;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
uniform mat4 light_pv;

void main()
{
    vs_out.frag_pos = vec3(model * vec4(pos, 1.0));
    vs_out.normal = transpose(inverse(mat3(model))) * normal;
    vs_out.tex_cord = tex_cord;
    vs_out.fragpos_lightspace = light_pv * vec4(vs_out.frag_pos, 1.0);
    gl_Position = vec4(tex_cord.x*2 -1, (1-tex_cord.y)*2 -1, 0.0,  1.0);
}
