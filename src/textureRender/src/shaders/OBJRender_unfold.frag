
#version 330 core
out vec3 color;

in vec2 uv;
in vec2 out_originalTex;
in vec3 out_normal;
in float depth;

uniform sampler2D rawImg;
uniform sampler2D depthTexture;

void main(){

	vec3 depth_encode = texture(depthTexture, out_originalTex).rgb;
	float depth_recover =  (depth_encode.r + depth_encode.g*256 + depth_encode.b*256*256)*255.0/65536.0;
/*
	if(abs(depth_recover-depth)<0.001)
		color = texture(rawImg, vec2(out_originalTex.x,1.0 - out_originalTex.y)).rgb;
	else
		color = vec3(0.0,0,0);
*/
	color = texture(rawImg, vec2(out_originalTex.x,1.0 - out_originalTex.y)).rgb;
}


