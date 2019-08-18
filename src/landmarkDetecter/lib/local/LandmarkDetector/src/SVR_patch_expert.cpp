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

#include "SVR_patch_expert.h"

// OpenCV include
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

#include "LandmarkDetectorUtils.h"

using namespace LandmarkDetector;

//===========================================================================
// Computing the image gradient
void Grad(const cv::Mat& im, cv::Mat& grad)
{
	
	/*float filter[3] = {1, 0, -1};
	float dfilter[1] = {1};	
	cv::Mat filterX = cv::Mat(1,3,CV_32F, filter).clone();
	cv::Mat filterY = cv::Mat(1,1,CV_32F, dfilter).clone();
			
	cv::Mat gradX;
	cv::Mat gradY;
	cv::sepFilter2D(im, gradX, CV_32F, filterY, filterX, cv::Point(-1,-1), 0);
	cv::sepFilter2D(im, gradY, CV_32F, filterX.t(), filterY, cv::Point(-1,-1), 0);
	cv::pow(gradX,2, gradX);
	cv::pow(gradY,2, gradY);
	grad = gradX + gradY;
			
	grad.row(0).setTo(0);
	grad.col(0).setTo(0);
	grad.col(grad.cols-1).setTo(0);
	grad.row(grad.rows-1).setTo(0);		*/

	// A quicker alternative
	int x,y,h = im.rows,w = im.cols;
	float vx,vy;

	// Initialise the gradient
	grad.create(im.size(), CV_32F);
	grad.setTo(0.0f);

	cv::MatIterator_<float> gp  = grad.begin<float>() + w+1;
	cv::MatConstIterator_<float> px1 = im.begin<float>()   + w+2;
	cv::MatConstIterator_<float> px2 = im.begin<float>()   + w;
	cv::MatConstIterator_<float> py1 = im.begin<float>()   + 2*w+1;
	cv::MatConstIterator_<float> py2 = im.begin<float>()   + 1;

	for(y = 1; y < h-1; y++)
	{ 
		for(x = 1; x < w-1; x++)
		{
			vx = *px1++ - *px2++;
			vy = *py1++ - *py2++;
			*gp++ = vx*vx + vy*vy;
		}
		px1 += 2;
		px2 += 2;
		py1 += 2;
		py2 += 2;
		gp += 2;
	}

}

// A copy constructor
SVR_patch_expert::SVR_patch_expert(const SVR_patch_expert& other) : weights(other.weights.clone())
{
	this->type = other.type;
	this->scaling = other.scaling;
	this->bias = other.bias;
	this->confidence = other.confidence;

	for (std::map<int, cv::Mat_<double> >::const_iterator it = other.weights_dfts.begin(); it != other.weights_dfts.end(); it++)
	{
		// Make sure the matrix is copied.
		this->weights_dfts.insert(std::pair<int, cv::Mat>(it->first, it->second.clone()));
	}
}

//===========================================================================
void SVR_patch_expert::Read(std::ifstream &stream)
{

	// A sanity check when reading patch experts
	int read_type;
	stream >> read_type;
	assert(read_type == 2);
  
	stream >> type >> confidence >> scaling >> bias;
	LandmarkDetector::ReadMat(stream, weights); 
	
	// OpenCV and Matlab matrix cardinality is different, hence the transpose
	weights = weights.t();

}

//===========================================================================
void SVR_patch_expert::Response(const cv::Mat_<float>& area_of_interest, cv::Mat_<float>& response)
{

	int response_height = area_of_interest.rows - weights.rows + 1;
	int response_width = area_of_interest.cols - weights.cols + 1;
	
	// the patch area on which we will calculate reponses
	cv::Mat_<float> normalised_area_of_interest;
  
	if(response.rows != response_height || response.cols != response_width)
	{
		response.create(response_height, response_width);
	}

	// If type is raw just normalise mean and standard deviation
	if(type == 0)
	{
		// Perform normalisation across whole patch
		cv::Scalar mean;
		cv::Scalar std;

		cv::meanStdDev(area_of_interest, mean, std);
		// Avoid division by zero
		if(std[0] == 0)
		{
			std[0] = 1;
		}
		normalised_area_of_interest = (area_of_interest - mean[0]) / std[0];
	}
	// If type is gradient, perform the image gradient computation
	else if(type == 1)
	{
		Grad(area_of_interest, normalised_area_of_interest);
	}
  	else
	{
		printf("ERROR(%s,%d): Unsupported patch type %d!\n", __FILE__,__LINE__, type);
		abort();
	}
	
	cv::Mat_<float> svr_response;

	// The empty matrix as we don't pass precomputed dft's of image
	cv::Mat_<double> empty_matrix_0(0,0,0.0);
	cv::Mat_<float> empty_matrix_1(0,0,0.0);
	cv::Mat_<float> empty_matrix_2(0,0,0.0);

	// Efficient calc of patch expert SVR response across the area of interest
	matchTemplate_m(normalised_area_of_interest, empty_matrix_0, empty_matrix_1, empty_matrix_2, weights, weights_dfts, svr_response, cv::TM_CCOEFF_NORMED);
	
	response.create(svr_response.size());
	cv::MatIterator_<float> p = response.begin();

	cv::MatIterator_<float> q1 = svr_response.begin(); // respone for each pixel
	cv::MatIterator_<float> q2 = svr_response.end();

	while(q1 != q2)
	{
		// the SVR response passed into logistic regressor
		*p++ = 1.0/(1.0 + exp( -(*q1++ * scaling + bias )));
	}

}

void SVR_patch_expert::ResponseDepth(const cv::Mat_<float>& area_of_interest, cv::Mat_<float> &response)
{

	// How big the response map will be
	int response_height = area_of_interest.rows - weights.rows + 1;
	int response_width = area_of_interest.cols - weights.cols + 1;
	
	// the patch area on which we will calculate reponses
	cv::Mat_<float> normalised_area_of_interest;
  
	if(response.rows != response_height || response.cols != response_width)
	{
		response.create(response_height, response_width);
	}

	if(type == 0)
	{
		// Perform normalisation across whole patch
		cv::Scalar mean;
		cv::Scalar std;
		
		// ignore missing values
		cv::Mat_<uchar> mask = area_of_interest > 0;
		cv::meanStdDev(area_of_interest, mean, std, mask);

		// if all values the same don't divide by 0
		if(std[0] == 0)
		{
			std[0] = 1;
		}

		normalised_area_of_interest = (area_of_interest - mean[0]) / std[0];

		// Set the invalid pixels to 0
		normalised_area_of_interest.setTo(0, mask == 0);
	}
	else
	{
		printf("ERROR(%s,%d): Unsupported patch type %d!\n", __FILE__,__LINE__,type);
		abort();
	}
  
	cv::Mat_<float> svr_response;
		
	// The empty matrix as we don't pass precomputed dft's of image
	cv::Mat_<double> empty_matrix_0(0,0,0.0);
	cv::Mat_<float> empty_matrix_1(0,0,0.0);
	cv::Mat_<float> empty_matrix_2(0,0,0.0);

	// Efficient calc of patch expert response across the area of interest

	matchTemplate_m(normalised_area_of_interest, empty_matrix_0, empty_matrix_1, empty_matrix_2, weights, weights_dfts, svr_response, cv::TM_CCOEFF);
	
	response.create(svr_response.size());
	cv::MatIterator_<float> p = response.begin();

	cv::MatIterator_<float> q1 = svr_response.begin(); // respone for each pixel
	cv::MatIterator_<float> q2 = svr_response.end();

	while(q1 != q2)
	{
		// the SVR response passed through a logistic regressor
		*p++ = 1.0/(1.0 + exp( -(*q1++ * scaling + bias )));
	}	
}

// Copy constructor				
Multi_SVR_patch_expert::Multi_SVR_patch_expert(const Multi_SVR_patch_expert& other) : svr_patch_experts(other.svr_patch_experts)
{
	this->width = other.width;
	this->height = other.height;
}

//===========================================================================
void Multi_SVR_patch_expert::Read(std::ifstream &stream)
{
	// A sanity check when reading patch experts
	int type;
	stream >> type;
	assert(type == 3);

	// The number of patch experts for this view (with different modalities)
	int number_modalities;

	stream >> width >> height >> number_modalities;
	
	svr_patch_experts.resize(number_modalities);
	for(int i = 0; i < number_modalities; i++)
		svr_patch_experts[i].Read(stream);

}
//===========================================================================
void Multi_SVR_patch_expert::Response(const cv::Mat_<float> &area_of_interest, cv::Mat_<float> &response)
{
	
	int response_height = area_of_interest.rows - height + 1;
	int response_width = area_of_interest.cols - width + 1;

	if(response.rows != response_height || response.cols != response_width)
	{
		response.create(response_height, response_width);
	}

	// For the purposes of the experiment only use the response of normal intensity, for fair comparison

	if(svr_patch_experts.size() == 1)
	{
		svr_patch_experts[0].Response(area_of_interest, response);		
	}
	else
	{
		// responses from multiple patch experts these can be gradients, LBPs etc.
		response.setTo(1.0);
		
		cv::Mat_<float> modality_resp(response_height, response_width);

		for(size_t i = 0; i < svr_patch_experts.size(); i++)
		{			
			svr_patch_experts[i].Response(area_of_interest, modality_resp);			
			response = response.mul(modality_resp);	
		}	
		
	}

}

void Multi_SVR_patch_expert::ResponseDepth(const cv::Mat_<float>& area_of_interest, cv::Mat_<float>& response)
{
	int response_height = area_of_interest.rows - height + 1;
	int response_width = area_of_interest.cols - width + 1;

	if(response.rows != response_height || response.cols != response_width)
	{
		response.create(response_height, response_width);
	}
	
	// With depth patch experts only do raw data modality
	svr_patch_experts[0].ResponseDepth(area_of_interest, response);
}
//===========================================================================
