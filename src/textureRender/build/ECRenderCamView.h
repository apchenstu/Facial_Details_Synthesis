#pragma once
#include "Util.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


class RenderCamView
{
public:
	RenderCamView() {};
	RenderCamView() restrict(amp);

	void setParameter(Vector3 &_location, Vector3 &_look, Vector3 &_up, float _FovW, float _FovH, int _imgW, int _imgH);

	glm::mat4 GetModelViewMatrix(float lambda = 1);
	glm::mat4 GetProjectionMatrix();

	Vector3 location;
	Vector3 lookat;
	Vector3 up;
	Vector3 K;
	float FovW, FovH;
	int imgW, imgH;
	Vector3 dw, dh;
};

