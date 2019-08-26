///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
//
// ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
//
// BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  
// IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
//
// License can be found in OpenFace-license.txt
//
//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace 2.0: Facial Behavior Analysis Toolkit
//       Tadas Baltrušaitis, Amir Zadeh, Yao Chong Lim, and Louis-Philippe Morency
//       in IEEE International Conference on Automatic Face and Gesture Recognition, 2018  
//
//       Convolutional experts constrained local model for facial landmark detection.
//       A. Zadeh, T. Baltrušaitis, and Louis-Philippe Morency,
//       in Computer Vision and Pattern Recognition Workshops, 2017.    
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltrušaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-specific normalisation for automatic Action Unit detection
//       Tadas Baltrušaitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
///////////////////////////////////////////////////////////////////////////////

#include "stdafx.h"

#include "PAW.h"

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

#include "LandmarkDetectorUtils.h"

using namespace LandmarkDetector;

// Copy constructor
PAW::PAW(const PAW& other) : destination_landmarks(other.destination_landmarks.clone()), source_landmarks(other.source_landmarks.clone()), triangulation(other.triangulation.clone()),
triangle_id(other.triangle_id.clone()), pixel_mask(other.pixel_mask.clone()), coefficients(other.coefficients.clone()), alpha(other.alpha.clone()), beta(other.beta.clone()), map_x(other.map_x.clone()), map_y(other.map_y.clone())
{
	this->number_of_pixels = other.number_of_pixels;
	this->min_x = other.min_x;
	this->min_y = other.min_y;
}

// A constructor from destination shape and triangulation
PAW::PAW(const cv::Mat_<float>& destination_shape, const cv::Mat_<int>& triangulation)
{
	// Initialise some variables directly
	this->destination_landmarks = destination_shape;
	this->triangulation = triangulation;

	int num_points = destination_shape.rows / 2;

	int num_tris = triangulation.rows;

	// Pre-compute the rest
	alpha = cv::Mat_<float>(num_tris, 3);
	beta = cv::Mat_<float>(num_tris, 3);

	cv::Mat_<float> xs = destination_shape(cv::Rect(0, 0, 1, num_points));
	cv::Mat_<float> ys = destination_shape(cv::Rect(0, num_points, 1, num_points));

	// Create a vector representation of the control points
	std::vector<std::vector<float>> destination_points;

	for (int tri = 0; tri < num_tris; ++tri)
	{
		int j = triangulation.at<int>(tri, 0);
		int k = triangulation.at<int>(tri, 1);
		int l = triangulation.at<int>(tri, 2);

		float c1 = ys.at<float>(l) - ys.at<float>(j);
		float c2 = xs.at<float>(l) - xs.at<float>(j);
		float c4 = ys.at<float>(k) - ys.at<float>(j);
		float c3 = xs.at<float>(k) - xs.at<float>(j);

		float c5 = c3*c1 - c2*c4;

		alpha.at<float>(tri, 0) = (ys.at<float>(j) * c2 - xs.at<float>(j) * c1) / c5;
		alpha.at<float>(tri, 1) = c1 / c5;
		alpha.at<float>(tri, 2) = -c2 / c5;

		beta.at<float>(tri, 0) = (xs.at<float>(j) * c4 - ys.at<float>(j) * c3) / c5;
		beta.at<float>(tri, 1) = -c4 / c5;
		beta.at<float>(tri, 2) = c3 / c5;

		// Add points corresponding to triangles as optimisation
		std::vector<float> triangle_points(10);

		triangle_points[0] = xs.at<float>(j);
		triangle_points[1] = ys.at<float>(j);
		triangle_points[2] = xs.at<float>(k);
		triangle_points[3] = ys.at<float>(k);
		triangle_points[4] = xs.at<float>(l);
		triangle_points[5] = ys.at<float>(l);

		cv::Vec3f xs_three(triangle_points[0], triangle_points[2], triangle_points[4]);
		cv::Vec3f ys_three(triangle_points[1], triangle_points[3], triangle_points[5]);

		double min_x, max_x, min_y, max_y;
		cv::minMaxIdx(xs_three, &min_x, &max_x);
		cv::minMaxIdx(ys_three, &min_y, &max_y);

		triangle_points[6] = (float)max_x;
		triangle_points[7] = (float)max_y;

		triangle_points[8] = (float)min_x;
		triangle_points[9] = (float)min_y;

		destination_points.push_back(triangle_points);

	}

	double max_x;
	double max_y;
	double min_x_d;
	double min_y_d;

	minMaxLoc(xs, &min_x_d, &max_x);
	minMaxLoc(ys, &min_y_d, &max_y);

	min_x = min_x_d;
	min_y = min_y_d;

	int w = (int)(max_x - min_x + 1.5);
	int h = (int)(max_y - min_y + 1.5);

	// Round the min_x and min_y for simplicity?

	pixel_mask = cv::Mat_<uchar>(h, w, (uchar)0);
	triangle_id = cv::Mat_<int>(h, w, -1);

	int curr_tri = -1;

	for (int y = 0; y < pixel_mask.rows; y++)
	{
		for (int x = 0; x < pixel_mask.cols; x++)
		{
			curr_tri = findTriangle(cv::Point_<float>(x + min_x, y + min_y), destination_points, curr_tri);
			// If there is a triangle at this location
			if (curr_tri != -1)
			{
				triangle_id.at<int>(y, x) = curr_tri;
				pixel_mask.at<uchar>(y, x) = 1;
			}
		}
	}

	// Preallocate maps and coefficients
	coefficients.create(num_tris, 6);
	map_x.create(pixel_mask.rows, pixel_mask.cols);
	map_y.create(pixel_mask.rows, pixel_mask.cols);


}

// Manually define min and max values
PAW::PAW(const cv::Mat_<float>& destination_shape, const cv::Mat_<int>& triangulation, float in_min_x, float in_min_y, float in_max_x, float in_max_y)
{
	// Initialise some variables directly
	this->destination_landmarks = destination_shape;
	this->triangulation = triangulation;

	int num_points = destination_shape.rows / 2;

	int num_tris = triangulation.rows;

	// Pre-compute the rest
	alpha = cv::Mat_<float>(num_tris, 3);
	beta = cv::Mat_<float>(num_tris, 3);

	cv::Mat_<float> xs = destination_shape(cv::Rect(0, 0, 1, num_points));
	cv::Mat_<float> ys = destination_shape(cv::Rect(0, num_points, 1, num_points));

	// Create a vector representation of the control points
	std::vector<std::vector<float>> destination_points;

	for (int tri = 0; tri < num_tris; ++tri)
	{
		int j = triangulation.at<int>(tri, 0);
		int k = triangulation.at<int>(tri, 1);
		int l = triangulation.at<int>(tri, 2);

		float c1 = ys.at<float>(l) - ys.at<float>(j);
		float c2 = xs.at<float>(l) - xs.at<float>(j);
		float c4 = ys.at<float>(k) - ys.at<float>(j);
		float c3 = xs.at<float>(k) - xs.at<float>(j);

		float c5 = c3*c1 - c2*c4;

		alpha.at<float>(tri, 0) = (ys.at<float>(j) * c2 - xs.at<float>(j) * c1) / c5;
		alpha.at<float>(tri, 1) = c1 / c5;
		alpha.at<float>(tri, 2) = -c2 / c5;

		beta.at<float>(tri, 0) = (xs.at<float>(j) * c4 - ys.at<float>(j) * c3) / c5;
		beta.at<float>(tri, 1) = -c4 / c5;
		beta.at<float>(tri, 2) = c3 / c5;

		// Add points corresponding to triangles as optimisation
		std::vector<float> triangle_points(10);

		triangle_points[0] = xs.at<float>(j);
		triangle_points[1] = ys.at<float>(j);
		triangle_points[2] = xs.at<float>(k);
		triangle_points[3] = ys.at<float>(k);
		triangle_points[4] = xs.at<float>(l);
		triangle_points[5] = ys.at<float>(l);

		cv::Vec3f xs_three(triangle_points[0], triangle_points[2], triangle_points[4]);
		cv::Vec3f ys_three(triangle_points[1], triangle_points[3], triangle_points[5]);

		double min_x, max_x, min_y, max_y;
		cv::minMaxIdx(xs_three, &min_x, &max_x);
		cv::minMaxIdx(ys_three, &min_y, &max_y);

		triangle_points[6] = (float)max_x;
		triangle_points[7] = (float)max_y;

		triangle_points[8] = (float)min_x;
		triangle_points[9] = (float)min_y;

		destination_points.push_back(triangle_points);

	}

	float max_x;
	float max_y;

	min_x = in_min_x;
	min_y = in_min_y;

	max_x = in_max_x;
	max_y = in_max_y;

	int w = (int)(max_x - min_x + 1.5);
	int h = (int)(max_y - min_y + 1.5);

	// Round the min_x and min_y for simplicity?

	pixel_mask = cv::Mat_<uchar>(h, w, (uchar)0);
	triangle_id = cv::Mat_<int>(h, w, -1);

	int curr_tri = -1;

	for (int y = 0; y < pixel_mask.rows; y++)
	{
		for (int x = 0; x < pixel_mask.cols; x++)
		{
			curr_tri = findTriangle(cv::Point_<float>(x + min_x, y + min_y), destination_points, curr_tri);
			// If there is a triangle at this location
			if (curr_tri != -1)
			{
				triangle_id.at<int>(y, x) = curr_tri;
				pixel_mask.at<uchar>(y, x) = 1;
			}
		}
	}

	// Preallocate maps and coefficients
	coefficients.create(num_tris, 6);
	map_x.create(pixel_mask.rows, pixel_mask.cols);
	map_y.create(pixel_mask.rows, pixel_mask.cols);

}

//===========================================================================
void PAW::Read(std::ifstream& stream)
{

	stream.read((char*)&number_of_pixels, 4);
	double min_x_d, min_y_d;
	stream.read((char*)&min_x_d, 8);
	stream.read((char*)&min_y_d, 8);
	min_x = (float)min_x_d;
	min_y = (float)min_y_d;

	cv::Mat_<double> destination_landmarks_d;
	ReadMatBin(stream, destination_landmarks_d);
	destination_landmarks_d.convertTo(destination_landmarks, CV_32F);

	ReadMatBin(stream, triangulation);

	ReadMatBin(stream, triangle_id);

	cv::Mat tmpMask;
	ReadMatBin(stream, tmpMask);
	tmpMask.convertTo(pixel_mask, CV_8U);

	cv::Mat_<double> alpha_d;
	ReadMatBin(stream, alpha_d);
	alpha_d.convertTo(alpha, CV_32F);

	cv::Mat_<double> beta_d;
	ReadMatBin(stream, beta_d);
	beta_d.convertTo(beta, CV_32F);

	map_x.create(pixel_mask.rows, pixel_mask.cols);
	map_y.create(pixel_mask.rows, pixel_mask.cols);

	coefficients.create(this->NumberOfTriangles(), 6);

	source_landmarks = destination_landmarks;
}

//=============================================================================
// cropping from the source image to the destination image using the shape in s, used to determine if shape fitting converged successfully
void PAW::Warp(const cv::Mat& image_to_warp, cv::Mat& destination_image, const cv::Mat_<float>& landmarks_to_warp)
{

	// set the current shape
	source_landmarks = landmarks_to_warp.clone();

	// prepare the mapping coefficients using the current shape
	this->CalcCoeff();

	// Do the actual mapping computation (where to warp from)
	this->WarpRegion(map_x, map_y);

	// Do the actual warp (with bi-linear interpolation)
	remap(image_to_warp, destination_image, map_x, map_y, cv::INTER_LINEAR);

}


//=============================================================================
// Calculate the warping coefficients
void PAW::CalcCoeff()
{
	int p = this->NumberOfLandmarks();

	for (int l = 0; l < this->NumberOfTriangles(); l++)
	{

		int i = triangulation.at<int>(l, 0);
		int j = triangulation.at<int>(l, 1);
		int k = triangulation.at<int>(l, 2);

		float c1 = source_landmarks.at<float>(i, 0);
		float c2 = source_landmarks.at<float>(j, 0) - c1;
		float c3 = source_landmarks.at<float>(k, 0) - c1;
		float c4 = source_landmarks.at<float>(i + p, 0);
		float c5 = source_landmarks.at<float>(j + p, 0) - c4;
		float c6 = source_landmarks.at<float>(k + p, 0) - c4;

		// Get a pointer to the coefficient we will be precomputing
		float *coeff = coefficients.ptr<float>(l);

		// Extract the relevant alphas and betas
		float *c_alpha = alpha.ptr<float>(l);
		float *c_beta = beta.ptr<float>(l);

		coeff[0] = c1 + c2 * c_alpha[0] + c3 * c_beta[0];
		coeff[1] = c2 * c_alpha[1] + c3 * c_beta[1];
		coeff[2] = c2 * c_alpha[2] + c3 * c_beta[2];
		coeff[3] = c4 + c5 * c_alpha[0] + c6 * c_beta[0];
		coeff[4] = c5 * c_alpha[1] + c6 * c_beta[1];
		coeff[5] = c5 * c_alpha[2] + c6 * c_beta[2];
	}
}

//======================================================================
// Compute the mapping coefficients
void PAW::WarpRegion(cv::Mat_<float>& mapx, cv::Mat_<float>& mapy)
{

	cv::MatIterator_<float> xp = mapx.begin();
	cv::MatIterator_<float> yp = mapy.begin();
	cv::MatIterator_<uchar> mp = pixel_mask.begin();
	cv::MatIterator_<int>   tp = triangle_id.begin();

	// The coefficients corresponding to the current triangle
	float * a;

	// Current triangle being processed	
	int k = -1;

	for (int y = 0; y < pixel_mask.rows; y++)
	{
		float yi = float(y) + min_y;

		for (int x = 0; x < pixel_mask.cols; x++)
		{
			float xi = float(x) + min_x;

			if (*mp == 0)
			{
				*xp = -1;
				*yp = -1;
			}
			else
			{
				// triangle corresponding to the current pixel
				int j = *tp;

				// If it is different from the previous triangle point to new coefficients
				// This will always be the case in the first iteration, hence a will not point to nothing
				if (j != k)
				{
					// Update the coefficient pointer if a new triangle is being processed
					a = coefficients.ptr<float>(j);
					k = j;
				}

				//ap is now the pointer to the coefficients
				float *ap = a;

				//look at the first coefficient (and increment). first coefficient is an x offset
				float xo = *ap++;
				//second coefficient is an x scale as a function of x
				xo += *ap++ * xi;
				//third coefficient ap(2) is an x scale as a function of y
				*xp = float(xo + *ap++ * yi);

				//then fourth coefficient ap(3) is a y offset
				float yo = *ap++;
				//fifth coeff adds coeff[4]*x to y
				yo += *ap++ * xi;
				//final coeff adds coeff[5]*y to y
				*yp = float(yo + *ap++ * yi);

			}
			mp++; tp++; xp++; yp++;
		}
	}
}

// ============================================================
// Helper functions to determine which point a triangle lies in
// ============================================================

// Is the point (x0,y0) on same side as a half-plane defined by (x1,y1), (x2, y2), and (x3, y3)
bool PAW::sameSide(float x0, float y0, float x1, float y1, float x2, float y2, float x3, float y3)
{

	float x = (x3 - x2)*(y0 - y2) - (x0 - x2)*(y3 - y2);
	float y = (x3 - x2)*(y1 - y2) - (x1 - x2)*(y3 - y2);

	return x*y >= 0;

}

// if point (x0, y0) is on same side for all three half-planes it is in a triangle
bool PAW::pointInTriangle(float x0, float y0, float x1, float y1, float x2, float y2, float x3, float y3)
{
	bool same_1 = sameSide(x0, y0, x1, y1, x2, y2, x3, y3);
	bool same_2 = sameSide(x0, y0, x2, y2, x1, y1, x3, y3);
	bool same_3 = sameSide(x0, y0, x3, y3, x1, y1, x2, y2);

	return same_1 && same_2 && same_3;

}

// Find if a given point lies in the triangles
int PAW::findTriangle(const cv::Point_<float>& point, const std::vector<std::vector<float>>& control_points, int guess)
{

	int num_tris = control_points.size();

	int tri = -1;

	float x0 = point.x;
	float y0 = point.y;

	// Allow a guess for speed (so as not to go through all triangles)
	if (guess != -1)
	{

		bool in_triangle = pointInTriangle(x0, y0, control_points[guess][0], control_points[guess][1], control_points[guess][2], control_points[guess][3], control_points[guess][4], control_points[guess][5]);
		if (in_triangle)
		{
			return guess;
		}
	}


	for (int i = 0; i < num_tris; ++i)
	{

		float max_x = control_points[i][6];
		float max_y = control_points[i][7];

		float min_x = control_points[i][8];
		float min_y = control_points[i][9];

		// Skip the check if the point is outside the bounding box of the triangle

		if (max_x < x0 || min_x > x0 || max_y < y0 || min_y > y0)
		{
			continue;
		}

		bool in_triangle = pointInTriangle(x0, y0,
			control_points[i][0], control_points[i][1],
			control_points[i][2], control_points[i][3],
			control_points[i][4], control_points[i][5]);

		if (in_triangle)
		{
			tri = i;
			break;
		}
	}
	return tri;
}
