#version 330 core

layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec2 in_TexCoord ;
layout(location = 2) in vec3 in_Normal ;

uniform mat3 R;
uniform mat4 MV;
uniform mat4 MVP;

uniform int H;
uniform int W;

out vec2 out_texcoord;
out vec3 out_normal;
out vec3 out_posiction;
out vec2 out_originalTex;

vec3 Ambient = vec3(0.15, 0.15, 0.15);
vec3 EyePosition = vec3(0, 0, 400);
float Kd = 1;
vec3 LightColor = vec3(1, 1, 1);
vec3 LightPosition = vec3(0, 50, 200);
vec3 Specular = vec3(0.5, 0.5, 0.5);


//out vec3  DiffuseColor;
//out vec3  SpecularColor;
out vec3 ecPosition;
out vec3 tnorm;


void main()
{
    ecPosition = R*vertexPosition_modelspace;
    tnorm      = normalize(R*in_Normal);
	/*
    vec3 lightVec   = normalize(LightPosition - ecPosition);
    vec3 viewVec    = normalize(EyePosition - ecPosition);
    vec3 Hvec       = normalize(viewVec + lightVec);

    float spec = abs(dot(Hvec, tnorm));
    spec = pow(spec, 16.0);

    DiffuseColor    = LightColor * vec3 (Kd * abs(dot(lightVec, tnorm)));
    DiffuseColor    = clamp(Ambient + DiffuseColor, 0.0, 1.0);
    SpecularColor   = clamp((LightColor * Specular * spec), 0.0, 1.0);

    out_originalTex = in_TexCoord;
	*/
	
    vec4 imageSpace = MVP * vec4(vertexPosition_modelspace,1);
    out_originalTex = vec2(imageSpace.x/W,imageSpace.y/H);
    gl_Position =  vec4(out_originalTex.xy*2-1,-ecPosition.z/1000,1);
}

/*
void main()
{
	//gl_Position = vec4(in_TexCoord.x*2-1,1-in_TexCoord.y*2,0,1);
   out_posiction = vertexPosition_modelspace;
   out_normal = normalize(in_Normal);

  
   vec4 imageSpace = MVP * vec4(vertexPosition_modelspace,1);
   out_originalTex = vec2(imageSpace.x/W,imageSpace.y/H);
   gl_Position =  vec4(out_originalTex.xy*2-1,0,1);


}
*/