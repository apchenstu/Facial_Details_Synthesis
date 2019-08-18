
#version 330 core
out vec3 color;
uniform int channel;

in vec3 out_posiction;

void main(){

	if(0==channel)
		color.r = out_posiction.r;
	else if(1==channel)
		color.r = out_posiction.g;
	else
		color.r = out_posiction.b;
}