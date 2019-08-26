#include "ECRenderCamView.h"

RenderCamView::RenderCamView() restrict(amp)
{

}


void RenderCamView::setParameter(Vector3 &_location, Vector3 &_look, Vector3 &_up, float _FovW, float _FovH, int _imgW, int _imgH)
{

	location = _location;
	lookat = _look.normalize();
	up = _up;
	FovW = _FovW * 2 * acos(-1) / 360.0;
	FovH = _FovH * 2 * acos(-1) / 360.0;
	imgW = _imgW;
	imgH = _imgH;

	Vector3 w = lookat.cross(up).normalize();
	Vector3 h = lookat.cross(w).normalize();

	w = w * tan(FovW / 2.0);
	h = h * tan(FovH / 2.0);

	K = lookat - w - h;
	dw = w * 2 / (float)imgW;
	dh = h * 2 / (float)imgH;
//	LOG4CPLUS_INFO(mlogger, "RenderCamView parameters have been set.");

}

glm::mat4 RenderCamView::GetModelViewMatrix(float lambda)
{
	glm::vec3 location = glm::vec3(this->location[0] * lambda, this->location[1] * lambda, this->location[2] * lambda);
	//std::cout << location.x << " " << location.y << " " << location.z << std::endl;
	glm::vec3 front = glm::vec3(this->lookat[0], this->lookat[1], this->lookat[2]);
	glm::vec3 _up = glm::vec3(this->up[0], this->up[1], this->up[2]);


	//glm::mat4 t(0.0f);
	//t[0][0] = t[1][1] = t[3][3] = 1.0;
	//t[2][2] = -1.0;

	return glm::lookAt(location, location + front, _up);
}

glm::mat4 RenderCamView::GetProjectionMatrix()
{
	float glnear = 1.0;
	glm::mat4 res = glm::perspective(
		FovH,			// Field of View (in degrees)
		(GLfloat)3840 / (GLfloat)2121,		// Aspect ratio (should be widnow width / height)
		glnear,				// Near clipping plane
		5000.0f				// Far clipping plane
		);

	res[0][0] = 2 * glnear / (tan(FovW / 2) * 2);

	return res;
}

