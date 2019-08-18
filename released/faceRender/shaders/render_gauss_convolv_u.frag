#version 330 core
#extension GL_ARB_explicit_uniform_location : enable
out vec4 frag_color;

in VS_OUT {
    vec2 tex_cord;
} fs_in;



uniform sampler2D irridance_map;
uniform sampler2D stretch_map;

uniform float gauss_width;//in mm
uniform float world_scale;

float curve[] = float[13](.00298, .009245, .027835, .065591, .120978, .174666, .197412, .174666, .120978, .065591, .027835, .009245, .00298);

void main()
{    

    float cur_u = fs_in.tex_cord.x;
    float cur_v = 1 - fs_in.tex_cord.y;
    float unscale_stretch = 5;//must match the scale in stretch.frag
    vec2 stretch = texture(stretch_map, vec2(cur_u, cur_v)).rg * unscale_stretch;
    vec2 net_width = stretch * world_scale  * (gauss_width / 1000);



    float u = cur_u - 3 * net_width.x;
    vec4 sum = vec4(0);
    for( int i = 0; i < 13; i++ ){
        sum += texture(irridance_map, vec2(u, cur_v)) * curve[i];
        u += .5 * net_width.x;
    }

    frag_color = sum;

}

