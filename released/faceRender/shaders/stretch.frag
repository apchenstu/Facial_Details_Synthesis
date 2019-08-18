#version 330 core

in vec3 local_pos;
in vec3 world_pos;

out vec4 frag_color;

uniform vec2 dim;

void main(void)
{
	vec3 du = dFdx( world_pos );
	vec3 dv = dFdy( world_pos );
	float scale = .2;//make output fit in range 0->1, must unscale this when using
	vec2 stretch = vec2(length( du ), length( dv )) * dim ;
	
	//outputs 1 / dist in world space that each pixel covers
	frag_color = vec4(scale* 1 / stretch, 0, 1); // A two-component texture
}