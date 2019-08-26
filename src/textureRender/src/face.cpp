#define _CRT_SECURE_NO_WARNINGS
#include "Util.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
OBJRender *mobjRender;
UINT16 render_w = 1024; UINT16 render_h = 1024;


int main(int argv, char **argc)
{
	std::string objName = argc[1];
	std::string mode = argc[2];
	std::string savefile = argc[3];
	std::string channel = argc[4];
	std::string shaderPath = argc[7];

	mobjRender = new OBJRender(objName, shaderPath, mode, render_w, render_h);
	mobjRender->channel = stoi(channel);

	if ("3" == mode || "4"==mode) {
		mobjRender->model = mode;
		mobjRender->loadTexture( argc[5]);

		ifstream mvpFile(argc[6]);
		glm::mat3 R; glm::mat4 MVP = glm::mat4(1); glm::mat4 MV = MVP;
		for (size_t row = 0; row < 3; row++)
			for (size_t col = 0; col < 4; col++) {
				mvpFile >> MVP[row][col];
			}
		mobjRender->MVPMatrix = MVP;
		mobjRender->RotateMatrix = glm::mat3(MV);
	}

	mobjRender->renderDepth();
	mobjRender->render();
	glFinish();

	unsigned char* result = (unsigned char *)malloc(sizeof(unsigned char)*render_w*render_h * 3);
	mobjRender->GetTextureData(mobjRender->renderedTexture[1], result);
	stbi_write_png(savefile.c_str(), render_w, render_h, 3, result, 0);

	return 0;
}