
#version 330 core
out vec3 color;

in float depth;

void main(){

   	int res = 65536;
	res = int(res*depth);
    color= vec3(res & 255,(res>>8)&255,(res>>16)&255)/255.0;

}