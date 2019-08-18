
#version 330 core
out vec3 color;
uniform int channel;

in vec2 out_originalTex;
in vec3 out_normal;

uniform sampler2D ourTexture;

void main(){
	color.rgb = texture(ourTexture, out_originalTex).rgb;
}


