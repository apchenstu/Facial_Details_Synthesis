#version 330 core
out vec4 frag_color;

in VS_OUT {
    vec3 normal;
} fs_in;

uniform float v[27];
uniform float c_val[7];


void main()
{    

    vec3 n = normalize(fs_in.normal);

    float x = n.x;
    float y = n.y;
    float z = n.z;
    float sh[9];

    sh[0] = c_val[0];
    sh[1] = c_val[1]*y;
    sh[2] = c_val[1]*z;
    sh[3] = c_val[1]*x;
    sh[4] = c_val[2]*(x*x - y*y);
    sh[5] = c_val[3]*(x*z);
    sh[6] = c_val[4]*(3*z*z -1);
    sh[7] = c_val[5]*(y*z);
    sh[8] = c_val[6]*(x*y);

    float irridance_x = 0.f;
    float irridance_y = 0.f;
    float irridance_z = 0.f;
    for(int i=0; i<9; i++){
        irridance_x += sh[i]*v[i];
        irridance_y += sh[i]*v[9 + i];
        irridance_z += sh[i]*v[18 +i];
    }

    // calculate irridance
    frag_color = vec4(irridance_x, irridance_y, irridance_z , 1);

}
