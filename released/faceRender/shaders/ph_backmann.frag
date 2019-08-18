#version 330 core

in vec2 txt_cord_out; 
out vec4 frag_color;

float PHBeckmann( float ndoth, float m )
{
  float alpha = acos( ndoth );
  float ta = tan( alpha );
  float val = 1.0/(m*m*pow(ndoth,4.0))*exp(-(ta*ta)/(m*m));
  return val;
}



void main()
{   
    //gl_FragDepth = gl_FragCoord.z;
    // normalize to [0, 1]
    float spec = 0.5 * pow( PHBeckmann( txt_cord_out.x, txt_cord_out.y ), 0.1 );
    frag_color = vec4(spec, spec, spec, 1.0);
    //frag_color = vec4(0., 0., 0., 1.0);
}
