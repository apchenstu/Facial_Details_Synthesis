
#version 330 core
out vec3 color;
uniform int channel;

in vec3 out_normal;

void main(){

   	if(0==channel)
		color.r = out_normal.r;
	else if(1==channel)
		color.r = out_normal.g;
	else
		color.r = out_normal.b;

}