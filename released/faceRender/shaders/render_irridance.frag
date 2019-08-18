#version 330 core
out vec4 frag_color;

in VS_OUT {
    vec3 frag_pos;
    vec3 normal;
    vec2 tex_cord;
    vec4 fragpos_lightspace;
} fs_in;

uniform vec3 light_pos;

uniform sampler2D shadow_map;
uniform sampler2D diffuse_texture;

float ShadowCalculation(vec4 fragpos_lightspace)
{
    
    // 1. without pcf
    // perform perspective divide
    vec3 proj_coords = fragpos_lightspace.xyz / fragpos_lightspace.w;
    // transform to [0,1] range
    proj_coords = proj_coords * 0.5 + 0.5;
    // get closest depth value from light's perspective (using [0,1] range fragPosLight as coords)
    float closest_depth = texture(shadow_map, proj_coords.xy).r; 
    // get depth of current fragment from light's perspective
    float current_depth = proj_coords.z;
    // check whether current frag pos is in shadow
    float shadow = current_depth - 0.005 > closest_depth  ? 1.0 : 0.0;
    return shadow;

    /*
    // 2. with pcf
    // perform perspective divide
    vec3 proj_coords = fragpos_lightspace.xyz / fragpos_lightspace.w;
    // transform to [0,1] range
    proj_coords = proj_coords * 0.5 + 0.5;
    // get depth of current fragment from light's perspective
    float current_depth = proj_coords.z;
    float shadow = 0.0;
    vec2 texel_size = 1.0 / textureSize(shadow_map, 0);
    int w_n = 3;
    for(int x = -w_n; x <= w_n; ++x)
    {
        for(int y = -w_n; y <= w_n; ++y)
        {
            float pcf_depth = texture(shadow_map, proj_coords.xy + vec2(x, y) * texel_size).r; 
            shadow += current_depth - 0.005 > pcf_depth ? 1.0 : 0.0;        
        }    
    }
    shadow /= (2*w_n + 1)*(2*w_n + 1);
    return shadow;*/

}


void main()
{    

    vec3 n = normalize(fs_in.normal);
    vec3 l = normalize(light_pos - fs_in.frag_pos);

    // calculate irridance
    vec3 tex = texture(diffuse_texture, vec2(fs_in.tex_cord.x, 1- fs_in.tex_cord.y)).xyz;
    vec3 light_color = vec3(0.99);
    float shadow = ShadowCalculation(fs_in.fragpos_lightspace);
    frag_color = vec4(((1.0 - shadow) * light_color * max(0.0, dot(n, l))), 1);

}
