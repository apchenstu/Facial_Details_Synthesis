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

#ifndef LANDMARK_DETECTOR_UTILS_H
#define LANDMARK_DETECTOR_UTILS_H

// OpenCV includes
#include <opencv2/core/core.hpp>

#include "LandmarkDetectorModel.h"

#include "FaceDetectorMTCNN.h"

namespace LandmarkDetector
{
	//===========================================================================	
	// Defining a set of useful utility functions to be used within CLNF

	//===========================================================================
	// Fast patch expert response computation (linear model across a ROI) using normalised cross-correlation
	//===========================================================================
	// This is a modified version of openCV code that allows for precomputed dfts of templates and for precomputed dfts of an image
	// _img is the input img, _img_dft it's dft (optional), _integral_img the images integral image (optional), squared integral image (optional), 
	// templ is the template we are convolving with, templ_dfts it's dfts at varying windows sizes (optional),  _result - the output, method the type of convolution
	void matchTemplate_m(const cv::Mat_<float>& input_img, cv::Mat_<double>& img_dft, cv::Mat& _integral_img, cv::Mat& _integral_img_sq, 
		const cv::Mat_<float>&  templ, std::map<int, cv::Mat_<double> >& templ_dfts, cv::Mat_<float>& result, int method);

	// Useful utility for grabing a bounding box around a set of 2D landmarks (as a 1D 2n x 1 vector of xs followed by doubles or as an n x 2 vector)
	void ExtractBoundingBox(const cv::Mat_<float>& landmarks, float &min_x, float &max_x, float &min_y, float &max_y);

	std::vector<cv::Point2f> CalculateVisibleLandmarks(const cv::Mat_<float>& shape2D, const cv::Mat_<int>& visibilities);
	std::vector<cv::Point2f> CalculateVisibleLandmarks(const CLNF& clnf_model);
	std::vector<cv::Point2f> CalculateVisibleEyeLandmarks(const CLNF& clnf_model);

	std::vector<cv::Point2f> CalculateAllLandmarks(const cv::Mat_<float>& shape2D);
	std::vector<cv::Point2f> CalculateAllLandmarks(const CLNF& clnf_model);
	std::vector<cv::Point2f> CalculateAllEyeLandmarks(const CLNF& clnf_model);
	std::vector<cv::Point3f> Calculate3DEyeLandmarks(const CLNF& clnf_model, float fx, float fy, float cx, float cy);

	//============================================================================
	// Face detection helpers
	//============================================================================

	// Face detection using Haar cascade classifier
	bool DetectFaces(std::vector<cv::Rect_<float> >& o_regions, const cv::Mat_<uchar>& intensity, float min_width = -1, cv::Rect_<float> roi = cv::Rect_<float>(0.0, 0.0, 1.0, 1.0));
	bool DetectFaces(std::vector<cv::Rect_<float> >& o_regions, const cv::Mat_<uchar>& intensity, cv::CascadeClassifier& classifier, float min_width = -1, cv::Rect_<float> roi = cv::Rect_<float>(0.0, 0.0, 1.0, 1.0));
	// The preference point allows for disambiguation if multiple faces are present (pick the closest one), if it is not set the biggest face is chosen
	bool DetectSingleFace(cv::Rect_<float>& o_region, const cv::Mat_<uchar>& intensity, cv::CascadeClassifier& classifier, const cv::Point preference = cv::Point(-1, -1), float min_width = -1, cv::Rect_<float> roi = cv::Rect_<float>(0.0, 0.0, 1.0, 1.0));

	// Face detection using HOG-SVM classifier
	bool DetectFacesHOG(std::vector<cv::Rect_<float> >& o_regions, const cv::Mat_<uchar>& intensity, std::vector<float>& confidences, float min_width = -1, cv::Rect_<float> roi = cv::Rect_<float>(0.0, 0.0, 1.0, 1.0));
	bool DetectFacesHOG(std::vector<cv::Rect_<float> >& o_regions, const cv::Mat_<uchar>& intensity, dlib::frontal_face_detector& classifier, std::vector<float>& confidences, float min_width = -1, cv::Rect_<float> roi = cv::Rect_<float>(0.0, 0.0, 1.0, 1.0));
	// The preference point allows for disambiguation if multiple faces are present (pick the closest one), if it is not set the biggest face is chosen
	bool DetectSingleFaceHOG(cv::Rect_<float>& o_region, const cv::Mat_<uchar>& intensity, dlib::frontal_face_detector& classifier, float& confidence, const cv::Point preference = cv::Point(-1, -1), float min_width = -1, cv::Rect_<float> roi = cv::Rect_<float>(0.0, 0.0, 1.0, 1.0));

	// Face detection using Multi-task Convolutional Neural Network
	bool DetectFacesMTCNN(std::vector<cv::Rect_<float> >& o_regions, const cv::Mat& image, LandmarkDetector::FaceDetectorMTCNN& detector, std::vector<float>& confidences);
	// The preference point allows for disambiguation if multiple faces are present (pick the closest one), if it is not set the biggest face is chosen
	bool DetectSingleFaceMTCNN(cv::Rect_<float>& o_region, const cv::Mat& image, LandmarkDetector::FaceDetectorMTCNN& detector, float& confidence, const cv::Point preference = cv::Point(-1, -1));

	//============================================================================
	// Matrix reading functionality
	//============================================================================

	// Reading a matrix written in a binary format
	void ReadMatBin(std::ifstream& stream, cv::Mat &output_mat);

	// Reading in a matrix from a stream
	void ReadMat(std::ifstream& stream, cv::Mat& output_matrix);

	// Skipping comments (lines starting with # symbol)
	void SkipComments(std::ifstream& stream);


}
#endif // LANDMARK_DETECTOR_UTILS_H
