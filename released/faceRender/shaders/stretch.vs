#version 330 core
layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 tex_cord;


uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
uniform mat4 light_pv;


out vec3 local_pos;
out vec3 world_pos;



void main()
{

    local_pos = pos;
    world_pos = vec3(model * vec4(pos, 1.0));
    gl_Position = vec4(tex_cord.x*2 -1, (1-tex_cord.y)*2 -1, 0.0,  1.0);

}
