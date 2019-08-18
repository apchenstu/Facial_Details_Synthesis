
#version 330 core

/*
out vec3 color;
uniform int channel;

in vec2 out_originalTex;
in vec3 out_normal;

uniform sampler2D ourTexture;

void main(){
	color.rgb = texture(ourTexture, vec2(out_originalTex.x,1-out_originalTex.y)).rgb;
}
*/

out vec3 color;

in vec3 ecPosition;
in vec3 tnorm;

float amb = 0.2;
float spec = 0.2;
vec3 Surfcolor = vec3(0.8, 0.8, 0.8);
vec3 LightPosition = vec3(0, 0, 1000);
vec3 EyePosition = vec3(0, 50, 1000);

void main () {
	vec3 N = normalize(tnorm);
	vec3 V = normalize(EyePosition - ecPosition);
	
	vec3 L =  normalize(LightPosition - ecPosition);
	vec3 R = normalize(2*N-L);

	vec3 ambient = Surfcolor * amb;
	vec3 diffuse = Surfcolor * (1. - amb) * max(dot(L, N), 0.0);
	vec3 specular = spec*vec3(1.0, 1.0, 1.0) * pow(max(dot(R, V), 0.0),3.0);

	color =  ambient + diffuse +specular;
	//color =  specular;
}