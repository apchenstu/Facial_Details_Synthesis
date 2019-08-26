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

#ifndef LANDMARK_DETECTION_VALIDATOR_H
#define LANDMARK_DETECTION_VALIDATOR_H

// OpenCV includes
#include <opencv2/core/core.hpp>

// System includes
#include <vector>

// Local includes
#include "PAW.h"

namespace LandmarkDetector
{
//===========================================================================
//
// Checking if landmark detection was successful using a CNN
// Using multiple validators trained add different views
// The regressor outputs 1 for ideal alignment and 0 for worst alignment
//===========================================================================
class DetectionValidator
{
		
public:    
	
	// The orientations of each of the landmark detection validator
	std::vector<cv::Vec3d> orientations;

	// Piecewise affine warps to the reference shape (per orientation)
	std::vector<PAW>     paws;

	//==========================================
	// Convolutional Neural Network

	// CNN layers for each view
	// view -> layer
	std::vector<std::vector<std::vector<std::vector<cv::Mat_<float> > > > > cnn_convolutional_layers;
	std::vector<std::vector<cv::Mat_<float> > > cnn_convolutional_layers_weights;
	std::vector<std::vector<cv::Mat_<float> > > cnn_convolutional_layers_im2col_precomp;

	std::vector< std::vector<int> > cnn_subsampling_layers;
	std::vector< std::vector<cv::Mat_<float> > > cnn_fully_connected_layers_weights;
	std::vector< std::vector<cv::Mat_<float>  > > cnn_fully_connected_layers_biases;
	// NEW CNN: 0 - convolutional, 1 - max pooling (2x2 stride 2), 2 - fully connected, 3 - relu, 4 - sigmoid
	std::vector<std::vector<int> > cnn_layer_types;
	
	//==========================================

	// Normalisation for face validation
	std::vector<cv::Mat_<float> > mean_images;
	std::vector<cv::Mat_<float> > standard_deviations;

	// Default constructor
	DetectionValidator(){;}

	// Copy constructor
	DetectionValidator(const DetectionValidator& other);

	// Given an image, orientation and detected landmarks output the result of the appropriate regressor
	float Check(const cv::Vec3d& orientation, const cv::Mat_<uchar>& intensity_img, cv::Mat_<float>& detected_landmarks);

	// Reading in the model
	void Read(std::string location);
			
	// Getting the closest view center based on orientation
	int GetViewId(const cv::Vec3d& orientation) const;

private:

	// The actual regressor application on the image

	// Convolutional Neural Network
	double CheckCNN(const cv::Mat_<float>& warped_img, int view_id);

	// A normalisation helper
	void NormaliseWarpedToVector(const cv::Mat_<float>& warped_img, cv::Mat_<float>& feature_vec, int view_id);

};

}
#endif // LANDMARK_DETECTION_VALIDATOR_H
