#define _CRT_SECURE_NO_WARNINGS
#include "util.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize.h>

using namespace glm;

void OBJRender::Scaleshape(std::vector<tinyobj::shape_t> &shapes, float Scale)
{
	float maxY = FLT_MIN;

	if (Scale > 0)
		scale = 1.0 / Scale;
	else
		scale = 1.0 / maxY;

	for (size_t t = 0; t < shapes.size(); ++t) {
		for (size_t v = 0; v < shapes[t].mesh.positions.size(); ++v) {
			shapes[t].mesh.positions[v] *= scale;
		}
	}
}

Vector3 OBJRender::CalcObjCenter(tinyobj::shape_t &myOBJ)
{
	
	Vector3 ObjCenter;

	ObjCenter.x = 0;
	ObjCenter.y = 0;
	ObjCenter.z = 0;

	int numVertices = myOBJ.mesh.positions.size();
	vector<float> x, y, z;
	x.resize(numVertices/3); y.resize(numVertices/3); z.resize(numVertices/3);

	for (int i = 0; i < numVertices/3; i++)
	{
			x[i] = myOBJ.mesh.positions[3*i];
			y[i] = myOBJ.mesh.positions[3*i  + 1];
			z[i] = myOBJ.mesh.positions[3*i + 2];
	}
	sort(x.begin(), x.end()); sort(y.begin(), y.end()); sort(z.begin(), z.end());
	ObjCenter.x = x[numVertices/6];
	ObjCenter.y = y[numVertices/6];
	ObjCenter.z = z[numVertices/6];

	return (ObjCenter);
}

void OBJRender::GenNormal(tinyobj::shape_t &shape,float angle) {


	const float pi = 3.14159;
	/* calculate the cosine of the angle (in degrees) */
	float cos_angle = cos(angle * pi / 180.0);


	/* allocate space for new normals */
	shape.mesh.normals.resize(shape.mesh.positions.size());
	for (size_t i = 0; i < shape.mesh.positions.size(); i++)
		shape.mesh.normals[i] = 0;
	std::vector<int> count(shape.mesh.positions.size() / 3, 0);

	Vector3 u, v,n;
	for (size_t i = 0; i < shape.mesh.indices.size(); i+=3) {
		unsigned int indexX = 3 * shape.mesh.indices[i];
		unsigned int indexY = 3 * shape.mesh.indices[i+1];
		unsigned int indexZ = 3 * shape.mesh.indices[i+2];

		u[0] = shape.mesh.positions[indexY + 0] -
			shape.mesh.positions[indexX + 0];
		u[1] = shape.mesh.positions[indexY + 1] -
			shape.mesh.positions[indexX + 1];
		u[2] = shape.mesh.positions[indexY + 2] -
			shape.mesh.positions[indexX + 2];

		v[0] = shape.mesh.positions[indexZ + 0] -
			shape.mesh.positions[indexX + 0];
		v[1] = shape.mesh.positions[indexZ + 1] -
			shape.mesh.positions[indexX + 1];
		v[2] = shape.mesh.positions[indexZ + 2] -
			shape.mesh.positions[indexX + 2];

		n = u.cross(v); n.normalize();
		count[int(indexX / 3)] += 1; count[int(indexY / 3)] += 1; count[int(indexZ / 3)] += 1;
		shape.mesh.normals[indexX] += n.x; shape.mesh.normals[indexX +1] += n.y; shape.mesh.normals[indexX + 2] += n.z;
		shape.mesh.normals[indexY] += n.x; shape.mesh.normals[indexY + 1] += n.y; shape.mesh.normals[indexY + 2] += n.z;
		shape.mesh.normals[indexZ] += n.x; shape.mesh.normals[indexZ + 1] += n.y; shape.mesh.normals[indexZ + 2] += n.z;
	}
	for (size_t i = 0; i < shape.mesh.positions.size(); i++)
		shape.mesh.normals[i] /= count[int(i / 3)];
}

void OBJRender::saveOBJ(std::vector<tinyobj::shape_t> &shapes,std::string filename)
{
	FILE *fp;
	std::string tempName = filename.substr(filename.find_last_of("\\")+1, filename.find_last_of(".")- filename.find_last_of("\\")-1);
	fp = fopen(filename.c_str(), "w");
	fprintf(fp, ("mtllib ./"+ tempName +".mtl\n").c_str());
	fprintf(fp, "usemtl material_0\n");
	for (size_t i = 0; i < shapes[0].mesh.positions.size(); i+=3)
		fprintf(fp, "v %f %f %f\n", shapes[0].mesh.positions[i], shapes[0].mesh.positions[i+1], shapes[0].mesh.positions[i+2]);
	
	if (shapes[0].mesh.normals.size() == 0)
		GenNormal(shapes[0],90);

	for (size_t i = 0; i < shapes[0].mesh.normals.size(); i += 3)
		fprintf(fp, "vn %f %f %f\n", shapes[0].mesh.normals[i], shapes[0].mesh.normals[i+1], shapes[0].mesh.normals[i+2]);

	for (size_t i = 0; i < shapes[0].mesh.texcoords.size(); i += 2)
		fprintf(fp, "vt %f %f\n", shapes[0].mesh.texcoords[i], shapes[0].mesh.texcoords[i + 1]);

	if (shapes[0].mesh.texcoords.size() == 0) {
		for (size_t i = 0; i < shapes[0].mesh.indices.size(); i += 3)
			fprintf(fp, "f %d/%d %d/%d %d/%d\n", shapes[0].mesh.indices[i] + 1, shapes[0].mesh.indices[i] + 1, shapes[0].mesh.indices[i + 1] + 1, shapes[0].mesh.indices[i + 1] + 1, shapes[0].mesh.indices[i + 2] + 1, shapes[0].mesh.indices[i + 2] + 1);
	}
	else {
		for (size_t i = 0; i < shapes[0].mesh.indices.size(); i += 3)
			fprintf(fp, "f %d/%d/%d %d/%d/%d %d/%d/%d\n",shapes[0].mesh.indices[i] + 1, shapes[0].mesh.indices[i] + 1, shapes[0].mesh.indices[i] + 1, shapes[0].mesh.indices[i + 1] + 1, shapes[0].mesh.indices[i + 1] + 1, shapes[0].mesh.indices[i + 1] + 1, shapes[0].mesh.indices[i + 2] + 1, shapes[0].mesh.indices[i + 2] + 1, shapes[0].mesh.indices[i + 2] + 1);
	}
	fclose(fp);

}


OBJRender::OBJRender(std::string filename, std::string shaderPath, std::string mode,int w, int h)
{
	tarW = w;
	tarH = h;
	std::string err;
	tinyobj::LoadObj(shapes, materials, err, filename.c_str());

	if (!glfwInit())
	{
		printf("Failed to initialize GLFW\n");
		exit(-1);
	}


	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);	// Want OpenGL 3.3
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);	// For MacOSX
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	window = glfwCreateWindow(tarW, tarH, "OpenGL", NULL, NULL);
	if (!window) {
	//	LOG4CPLUS_ERROR(mlogger, "Failed to open GLFW window.  If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
		glfwTerminate();
		exit(-1);
	}

	glfwMakeContextCurrent(window);
	glfwHideWindow(window);
	gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
	
	texUVs = VertexArray = vertexbuffers = elementbuffers = NULL;
	loadModel(shapes, VertexArray, vertexbuffers, texUVs, elementbuffers);

	std::string ShaderFragment;
	std::string ShaderVertex = shaderPath+"/OBJRender.vs";
	if ("0" == mode)
		ShaderFragment = shaderPath + "/OBJRender_normal.frag";
	else if ("1" == mode)
		ShaderFragment = shaderPath + "/OBJRender_posiction.frag";
	else if("2"==mode)
		ShaderFragment = shaderPath + "/OBJRender_mask.frag";
	else if ("3" == mode)
		ShaderFragment = shaderPath + "/OBJRender_unfold.frag";
	else if ("4" == mode) {
		ShaderVertex = shaderPath + "/OBJRender_project.vs";
		ShaderFragment = shaderPath + "/OBJRender_project.frag";
	}

	programTexture = LoadShaders(ShaderVertex.c_str(), ShaderFragment.c_str());
	if ("3" == mode) {
		ShaderVertex = shaderPath + "/OBJRender_depth.vs";
		ShaderFragment = shaderPath + "/OBJRender_depth.frag";
		programDepth = LoadShaders(ShaderVertex.c_str(), ShaderFragment.c_str());

		MVPMatrixID_depth = glGetUniformLocation(programDepth, "MVP");
		ROWSID_depth = glGetUniformLocation(programDepth, "H");
		COLSID_depth = glGetUniformLocation(programDepth, "W");
	}

	RotateID = glGetUniformLocation(programTexture, "R");
	MVPMatrixID = glGetUniformLocation(programTexture, "MVP");
	//MVMatrixID = glGetUniformLocation(programTexture, "MV");
	MVID = glGetUniformLocation(programTexture, "channel");
	ROWSID = glGetUniformLocation(programTexture, "H");
	COLSID = glGetUniformLocation(programTexture, "W");

	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

	glEnable(GL_DEPTH_TEST);
	glClearColor(0, 0, 0, 1);
	glClear(GL_COLOR_BUFFER_BIT);

	glGenFramebuffers(2, FramebufferName);
	glGenTextures(2, renderedTexture);
	glGenRenderbuffers(2, depthrenderbuffer);

	glGenRenderbuffers(2, renderedNorms);

	for (int i = 0; i < 2; ++i)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName[i]);
		glBindTexture(GL_TEXTURE_2D, renderedTexture[i]);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tarW, tarH, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

		glBindRenderbuffer(GL_RENDERBUFFER, depthrenderbuffer[i]);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, tarW, tarH);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthrenderbuffer[i]);

		glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, renderedTexture[i], 0);

	}

}

void OBJRender::Resize(int w, int h)
{
	glDeleteTextures(2, renderedTexture);
	glDeleteFramebuffers(2,FramebufferName);

	glGenTextures(2, renderedTexture);
	glGenFramebuffers(2, FramebufferName);

	for (int i = 0; i < 2; ++i)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName[i]);
		glBindTexture(GL_TEXTURE_2D, renderedTexture[i]);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

		glBindRenderbuffer(GL_RENDERBUFFER, depthrenderbuffer[i]);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, w, h);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthrenderbuffer[i]);

		glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, renderedTexture[i], 0);

	}
}

OBJRender::~OBJRender()
{
	glDeleteBuffers(shapes.size(), vertexbuffers);
	glDeleteBuffers(shapes.size(), elementbuffers);

	glfwDestroyWindow(window);
	//glfwTerminate();
}


unsigned char* OBJRender::loadImage(string imagePacth,int &width,int &height,int &nrChannels) {
	
	unsigned char *data = stbi_load(imagePacth.c_str(), &width, &height, &nrChannels, 0);
	row = height; col = width;
	int width_new = width;int height_new = height;
	width_new = pow(2,int(log(width)/log(2)+1));
	height_new = pow(2, int(log(height) / log(2) + 1));
	unsigned char *data_new = (unsigned char *)malloc(sizeof(unsigned char)*height_new*width_new * nrChannels);
	stbir_resize_uint8(data, width, height, 0,data_new, width_new, height_new, 0, nrChannels);
	width = width_new; height = height_new;
	return data_new;

}

void OBJRender::loadTexture(string imagePacth) {
	glGenTextures(1, &OBJTexture);
	int width, height, nrChannels;
	unsigned char *data = loadImage(imagePacth, width, height, nrChannels);

	glBindTexture(GL_TEXTURE_2D, OBJTexture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	if (nrChannels == 3) {
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
	}
	else if (nrChannels == 1) {
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, data);
	}
	else {
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
	}
	glGenerateMipmap(GL_TEXTURE_2D);
	stbi_image_free(data);

	
}

void OBJRender::loadModel(std::vector<tinyobj::shape_t> &shapes, GLuint *& VertexArray, GLuint *&vertexbuffers, GLuint *&texUVs, GLuint *&elementbuffers)
{
	int n = shapes.size();

	//	if (VertexArray) delete[] VertexArray;
	VertexArray = new GLuint[1];

	//	if (normalbuffers) delete[] normalbuffers;
	normalbuffers = new GLuint[n];

	//	if (vertexbuffers) delete[] vertexbuffers;
	vertexbuffers = new GLuint[n];

	//	if (texUVs) delete[] texUVs;
	texUVs = new GLuint[n];


	//	if (elementbuffers) delete[] elementbuffers;
	elementbuffers = new GLuint[n];

	glGenVertexArrays(1, VertexArray);
	glGenBuffers(n, vertexbuffers);
	glGenBuffers(n, elementbuffers);
	glGenBuffers(n, normalbuffers);
	glGenBuffers(n, texUVs);
	glBindVertexArray(VertexArray[0]);

	for (int i = 0; i < n; ++i)
	{

		//printf("%d\n", i);


		float * vertices = shapes[i].mesh.positions.data();
		int numVertices = shapes[i].mesh.positions.size();
		int numIndex = shapes[i].mesh.indices.size();
		unsigned int * indices = shapes[i].mesh.indices.data();
		float * texcoord = shapes[i].mesh.texcoords.data();
		int numTexCoord = shapes[i].mesh.texcoords.size();

		int numNormals = shapes[i].mesh.normals.size();
		float * Norms = shapes[i].mesh.normals.data();


		//printf("Item %d ,  %d faces.   %d %d  %d\n", i, numIndex / 3, numNormals, numVertices, numTexCoord);

		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffers[i]);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float)*numVertices, vertices, GL_STATIC_DRAW);


		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffers[i]);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, numIndex * sizeof(unsigned int), indices, GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, normalbuffers[i]);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, numNormals * sizeof(float), Norms, GL_STATIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, texUVs[i]);
		glBufferData(GL_ARRAY_BUFFER, numTexCoord * sizeof(float), texcoord, GL_STATIC_DRAW);
	}
//	LOG4CPLUS_INFO(mlogger, "=======================\n");
}



int OBJRender::renderDepth() {
	glViewport(0, 0, tarW, tarH);
	glUseProgram(programDepth);

	//glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	glEnable(GL_DEPTH_TEST);

	glClearColor(0, 0, 0, 1);

	glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName[0]);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUniformMatrix4fv(MVPMatrixID_depth, 1, 1, &MVPMatrix[0][0]);
	glUniform1i(ROWSID_depth, row);
	glUniform1i(COLSID_depth, col);

	glBindVertexArray(VertexArray[0]);

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);

	glEnable(GL_TEXTURE);
	glActiveTexture(GL_TEXTURE0);glBindTexture(GL_TEXTURE_2D, OBJTexture);
	inference();
	glBindVertexArray(0);

	glBindTexture(GL_TEXTURE_2D, renderedTexture[0]);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glUseProgram(0);

	return 0;
}

int OBJRender::render()
{
	glViewport(0, 0, tarW, tarH);
	glUseProgram(programTexture);

	//glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	glEnable(GL_DEPTH_TEST);

	glClearColor(0, 0, 0, 1);

	glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName[1]);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	glUniformMatrix4fv(MVPMatrixID, 1, 1, &MVPMatrix[0][0]);
	glUniform1i(MVID,channel);
	glUniform1i(ROWSID, row);
	glUniform1i(COLSID, col);

	glBindVertexArray(VertexArray[0]);

	glUniform1i(glGetUniformLocation(programTexture, "rawImg"), 0);
	glUniform1i(glGetUniformLocation(programTexture, "depthTexture"), 1);
	glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, OBJTexture);
	glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, renderedTexture[0]);

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);

	//glEnable(GL_TEXTURE);

	inference();
	glFlush();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glUseProgram(0);

	return 0;
}

void OBJRender::SetMatrices(glm::mat4 _ModelViewMatrix, glm::mat4 _ProjectionMatrix)
{
	RotateMatrix = glm::mat3(_ModelViewMatrix[0], _ModelViewMatrix[1], _ModelViewMatrix[2]);
	ModelViewMatrix = _ModelViewMatrix;
	ProjectionMatrix = _ProjectionMatrix;
}

void OBJRender::GetTextureData(GLuint tex, unsigned char * depthmap)
{
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glUseProgram(0);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, depthmap);
}

GLuint OBJRender::LoadShaders(const char * vertex_file_path, const char * fragment_file_path)
{
	// Create the shaders
	GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
	GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

	// Read the Vertex Shader code from the file
	std::string VertexShaderCode;
	std::ifstream VertexShaderStream(vertex_file_path, std::ios::in);
	if (VertexShaderStream.is_open())
	{
		std::string Line = "";
		while (getline(VertexShaderStream, Line))
			VertexShaderCode += "\n" + Line;
		VertexShaderStream.close();
	}

	// Read the Fragment Shader code from the file
	std::string FragmentShaderCode;
	std::ifstream FragmentShaderStream(fragment_file_path, std::ios::in);
	if (FragmentShaderStream.is_open()) {
		std::string Line = "";
		while (getline(FragmentShaderStream, Line))
			FragmentShaderCode += "\n" + Line;
		FragmentShaderStream.close();
	}

	GLint Result = GL_FALSE;
	int InfoLogLength;

	// Compile Vertex Shader
	//printf("Compiling shader : %s\n", vertex_file_path);
	char const * VertexSourcePointer = VertexShaderCode.c_str();
	glShaderSource(VertexShaderID, 1, &VertexSourcePointer, NULL);
	glCompileShader(VertexShaderID);

	// Check Vertex Shader
	glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	std::vector<char> VertexShaderErrorMessage(InfoLogLength+1);
	glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
	//fprintf(stdout, "%s\n", &VertexShaderErrorMessage[0]);

	// Compile Fragment Shader
	//printf("Compiling shader : %s\n", fragment_file_path);
	char const * FragmentSourcePointer = FragmentShaderCode.c_str();
	glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer, NULL);
	glCompileShader(FragmentShaderID);

	// Check Fragment Shader
	glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	std::vector<char> FragmentShaderErrorMessage(InfoLogLength+1);
	glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
	//fprintf(stdout, "%s\n", &FragmentShaderErrorMessage[0]);

	// Link the program
	//fprintf(stdout, "Linking programn");
	GLuint ProgramID = glCreateProgram();
	glAttachShader(ProgramID, VertexShaderID);
	glAttachShader(ProgramID, FragmentShaderID);
	glLinkProgram(ProgramID);

	// Check the program
	glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
	glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	std::vector<char> ProgramErrorMessage(std::max(InfoLogLength, int(1)));
	glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
	//fprintf(stdout, "%s\n", &ProgramErrorMessage[0]);

	glDeleteShader(VertexShaderID);
	glDeleteShader(FragmentShaderID);

	return ProgramID;
}

void OBJRender::inference() {

	for (int i = 0; i < (int)shapes.size(); ++i)
	{

		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffers[i]);
		glVertexAttribPointer(
			0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*)0            // array buffer offset
		);




		glBindBuffer(GL_ARRAY_BUFFER, texUVs[i]);
		glVertexAttribPointer(
			1,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
			2,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*)0            // array buffer offset
		);


		glBindBuffer(GL_ARRAY_BUFFER, normalbuffers[i]);
		glVertexAttribPointer(
			2,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*)0            // array buffer offset
		);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffers[i]);
		// Draw the triangle !
		//	glDrawArrays(GL_TRIANGLES, 0, numVertices); // Starting from vertex 0; 3 vertices total -> 1 triangle

		glDrawElements(
			GL_TRIANGLES,      // mode
			shapes[i].mesh.indices.size(),    // count
			GL_UNSIGNED_INT,   // type
			(void*)0           // element array buffer offset
		);
	}
}
