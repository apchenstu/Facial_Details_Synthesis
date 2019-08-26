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
out float depth;
out vec2 uv;

void main()
{
   out_posiction = vertexPosition_modelspace;
   out_normal = normalize(in_Normal);

   vec4 imageSpace = MVP * vec4(vertexPosition_modelspace,1);
   out_originalTex = vec2(imageSpace.x/W/imageSpace.z,imageSpace.y/H/imageSpace.z);
   out_originalTex.y = 1.0 - out_originalTex.y;
   depth = abs(imageSpace.z) - MVP[2][3] + 0.2;
   gl_Position = vec4(in_TexCoord.x*2-1,1-in_TexCoord.y*2,0,1);
   uv = gl_Position.xy;
}