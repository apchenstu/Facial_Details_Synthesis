#version 330 core

layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec2 in_TexCoord ;
layout(location = 2) in vec3 in_Normal ;

uniform mat4 MVP;
uniform int H;
uniform int W;

out vec2 out_texcoord;
out vec3 out_normal;
out vec3 out_posiction;
out vec2 out_originalTex;


void main()
{
	gl_Position = vec4(in_TexCoord.x*2-1,1-in_TexCoord.y*2,0,1);
//	gl_Position = vec4(1-in_TexCoord.y*2,1-in_TexCoord.x*2,0,1);
   out_posiction = vertexPosition_modelspace;
   out_normal = normalize(in_Normal);

  
   vec4 imageSpace = MVP * vec4(vertexPosition_modelspace,1);
   out_originalTex = vec2(imageSpace.x/W,imageSpace.y/H);
   out_originalTex.y = 1.0 - out_originalTex.y;
   //out_originalTex = in_TexCoord;
}