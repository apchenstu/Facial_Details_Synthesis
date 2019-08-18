#version 330 core
out vec4 frag_color;

in VS_OUT {
    vec2 tex_cord;
    vec3 normal;
    vec3 tgt;
    vec3 pos;
} fs_in;

uniform sampler2D hm;

uniform int flag;

uniform vec3  LightPosition;
uniform vec3  LightColor;
uniform vec3  EyePosition;
uniform vec3  Specular;
uniform vec3  Ambient;
uniform float Kd;
uniform vec3  SurfaceColor;

void main (void)
{

    // compute normal 
    vec3 tnorm;
    if(flag ==1 ){
		vec3 N = normalize(fs_in.normal);
		vec3 T = normalize(fs_in.tgt- dot(fs_in.normal, fs_in.tgt)*fs_in.normal);
		vec3 B = cross(N,T);
		mat3 TBN= mat3(T,B,N);
		vec2 uv = vec2(fs_in.tex_cord.x, 1-fs_in.tex_cord.y);
		vec3 map_normal = texture2D(hm, uv).xyz;
		tnorm = 2.0*map_normal-vec3(1.0,1.0,1.0);
		tnorm = normalize(TBN*tnorm);
    }else{
    	tnorm = fs_in.normal;
    }



    vec3 lightVec   = normalize(LightPosition - fs_in.pos);
    vec3 viewVec    = normalize(EyePosition - fs_in.pos);
    vec3 Hvec       = normalize(viewVec + lightVec);

    float spec = abs(dot(Hvec, tnorm));
    spec = pow(spec, 16.0);

    vec3 DiffuseColor    = LightColor * vec3 (Kd * abs(dot(lightVec, tnorm)));
    DiffuseColor    = clamp(Ambient + DiffuseColor, 0.0, 1.0);
    vec3 SpecularColor   = clamp((LightColor * Specular * spec), 0.0, 1.0);

    vec3 finalColor = SurfaceColor * DiffuseColor + SpecularColor;
    frag_color = vec4 (finalColor, 1.0);

}
