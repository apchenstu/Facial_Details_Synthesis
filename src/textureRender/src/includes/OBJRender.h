
#pragma once

#ifndef RENDER_H
#define RENDER_H
#include "Util.h"
#include "tinyobj.h"
#include <vector>
#include <string>
#include <thread>

class Camera;
class OBJRender
{
public:
	Vector3 center;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	GLuint * VertexArray;
	GLuint *vertexbuffers;
	GLuint *normalbuffers;
	GLuint *texUVs;
	GLuint *elementbuffers;
	GLuint OBJTexture;
	GLFWwindow* window;
	GLuint programTexture, programDepth;
	GLuint MVPMatrixID, MVMatrixID, RotateID, MVID, COLSID, ROWSID;
	GLuint COLSID_depth,ROWSID_depth, MVPMatrixID_depth;
	glm::mat3 RotateMatrix;
	glm::mat4 ModelViewMatrix;
	glm::mat4 ProjectionMatrix;
	glm::mat4 MVMatrix,MVPMatrix;
	GLuint FramebufferName[2];
	GLuint renderedTexture[2];
	GLuint renderedNorms[2];
	GLuint depthrenderbuffer[2];
	void loadModel(std::vector<tinyobj::shape_t> &shapes, GLuint *& VertexArray, GLuint *&vertexbuffers, GLuint *&texUVs, GLuint *&elementbuffers);
	void inference();
	int renderDepth();
	int tarW, tarH;
	string model;
	string imagePath;

	float scale;
	int channel,row,col;
	void Scaleshape(std::vector<tinyobj::shape_t>& shapes, float Scale = 0);
	void saveOBJ(std::vector<tinyobj::shape_t> &shapes, std::string filename);
	Vector3 CalcObjCenter(tinyobj::shape_t &shapes);
	void GenNormal(tinyobj::shape_t &shape, float angle);

	OBJRender(std::string filename, std::string shaderPath, std::string mode, int w = 1500, int h = 1000);
	~OBJRender();

	unsigned char * loadImage(string imagePacth, int &width, int &height, int &nrChannels);

	int render();
	void Resize(int W, int H);
	void SetMatrices(glm::mat4 _ModelViewMatrix, glm::mat4 _ProjectionMatrix);

	void GetTextureData(GLuint tex, unsigned char * depthmap);
	void OBJRender::loadTexture(string imagePacth);
	static GLuint LoadShaders(const char * vertex_file_path, const char * fragment_file_path);
};


//static OBJRender *objRender;
//static std::thread * glThread;


#endif