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

//  Header for all external CLM/CLNF/CE-CLM methods of interest to the user
//
//
#ifndef LANDMARK_DETECTOR_FUNC_H
#define LANDMARK_DETECTOR_FUNC_H

// OpenCV includes
#include <opencv2/core/core.hpp>

#include <LandmarkDetectorParameters.h>
#include <LandmarkDetectorUtils.h>
#include <LandmarkDetectorModel.h>

namespace LandmarkDetector
{

	//================================================================================================================
	// Landmark detection in videos, need to provide an image and model parameters (default values work well)
	// Optionally can provide a bounding box from which to start tracking
	// Can also optionally pass a grayscale image if it has already been computed to speed things up a bit
	//================================================================================================================
	bool DetectLandmarksInVideo(const cv::Mat &rgb_image, CLNF& clnf_model, FaceModelParameters& params, cv::Mat &grayscale_image);
	bool DetectLandmarksInVideo(const cv::Mat &rgb_image, const cv::Rect_<double> bounding_box, CLNF& clnf_model, FaceModelParameters& params, cv::Mat &grayscale_image);

	//================================================================================================================
	// Landmark detection in image, need to provide an image and optionally CLNF model together with parameters (default values work well)
	// Optionally can provide a bounding box in which detection is performed (this is useful if multiple faces are to be detected in images)
	// Can also optionally pass a grayscale image if it has already been computed to speed things up a bit
	//================================================================================================================
	bool DetectLandmarksInImage(const cv::Mat &rgb_image, CLNF& clnf_model, FaceModelParameters& params, cv::Mat &grayscale_image);
	// Providing a bounding box
	bool DetectLandmarksInImage(const cv::Mat &rgb_image, const cv::Rect_<double> bounding_box, CLNF& clnf_model, FaceModelParameters& params, cv::Mat &grayscale_image);

	//================================================================
	// Helper function for getting head pose from CLNF parameters

	// Return the current estimate of the head pose in world coordinates with camera at origin (0,0,0)
	// The format returned is [Tx, Ty, Tz, Eul_x, Eul_y, Eul_z]
	cv::Vec6f GetPose(const CLNF& clnf_model, float fx, float fy, float cx, float cy);

	// Return the current estimate of the head pose in world coordinates with camera at origin (0,0,0), but with rotation representing if the head is looking at the camera
	// The format returned is [Tx, Ty, Tz, Eul_x, Eul_y, Eul_z]
	cv::Vec6f GetPoseWRTCamera(const CLNF& clnf_model, float fx, float fy, float cx, float cy);
}
#endif // LANDMARK_DETECTOR_FUNC_H
