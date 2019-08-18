#version 330 core
out vec4 frag_color;

in VS_OUT {
    vec2 tex_cord;
} fs_in;

uniform sampler2D diffuse_texture;

// blurred texture
uniform sampler2D tex_blur0;
uniform sampler2D tex_blur1;
uniform sampler2D tex_blur2;
uniform sampler2D tex_blur3;
uniform sampler2D tex_blur4;
uniform sampler2D tex_blur5;



// coefficients to different blurred texture
uniform vec4 coeff0;
uniform vec4 coeff1;
uniform vec4 coeff2;
uniform vec4 coeff3;
uniform vec4 coeff4;
uniform vec4 coeff5;


/////////////////////////////////////////////////////////////////////////////
// Computes the bssrdf term based on the gaussian textures
vec4 bssrdf(vec2 tex_coord)
{
    vec4 final_interp = coeff0 * texture (tex_blur0, tex_coord);
    final_interp += coeff1 * texture (tex_blur1, tex_coord);
    final_interp += coeff2 * texture (tex_blur2, tex_coord); 
    final_interp += coeff3 * texture (tex_blur3, tex_coord); 
    final_interp += coeff4 * texture (tex_blur4, tex_coord); 
    final_interp += coeff5 * texture (tex_blur5, tex_coord);

    // renormalize to white diffuse light
    final_interp /= (coeff0 + coeff1 + coeff2 + coeff3 + coeff4 + coeff5); 

    return final_interp;
}


void main()
{    

    vec2 uv = vec2(fs_in.tex_cord.x, 1-fs_in.tex_cord.y);
   
    vec3 blr_cmbin_irrd = bssrdf(uv).rgb;

    vec3 diffuse_clr = texture(diffuse_texture, uv).rgb;
    vec3 light_clr = vec3(1.0, 1.0, 1.0);

    // ambient 
    float amb = .2;
    vec3 color = amb*diffuse_clr;

    // scattering
    color += blr_cmbin_irrd*diffuse_clr;


    // apply gamma correction
    float gamma = 1.;
    frag_color.rgb = pow(color, vec3(1.0/gamma));

}