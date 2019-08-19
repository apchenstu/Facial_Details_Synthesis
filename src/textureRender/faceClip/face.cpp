#define _CRT_SECURE_NO_WARNINGS
#include "header.h"
OBJRender *mobjRender;
UINT16 render_w = 4096; UINT16 render_h = 4096;


int main(int argv, char **argc)
{
	std::string objName = argc[1];
	std::string mode = argc[2];
	std::string savefile = argc[3];
	std::string channel = argc[4];

	
	mobjRender = new OBJRender(objName.c_str(), mode,render_w, render_h);
	mobjRender->channel = stoi(channel);

	if ("3" == mode || "4"==mode) {
		mobjRender->model = mode;
		mobjRender->loadTexture( argc[5]);

		ifstream mvFile(string(argc[6]) + ".modelview.txt");
		ifstream mvpFile(string(argc[6])+ ".affine_from_ortho.txt");
		glm::mat3 R; glm::mat4 MVP = glm::mat4(1); glm::mat4 MV = MVP;
		for (uint row = 0; row < 3; row++)
			for (uint col = 0; col < 4; col++) {
				mvpFile >> MVP[row][col];
				mvFile >> MV[row][col];
			}
		mobjRender->MVPMatrix = MVP;
		mobjRender->MVMatrix = MV;
		mobjRender->RotateMatrix = glm::mat3(MV);
	}

	mobjRender->render();
	glFinish();

	
	//cv::Mat img = cv::Mat::zeros(cv::Size(render_w, render_h), CV_8UC3);
	//mobjRender->GetTextureData(mobjRender->renderedTexture[0], img.data);

	//cv::imwrite(savefile, img);

	return 0;
}