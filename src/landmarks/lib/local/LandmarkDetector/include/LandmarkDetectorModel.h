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

#ifndef LANDMARK_DETECTOR_MODEL_H
#define LANDMARK_DETECTOR_MODEL_H

// OpenCV dependencies
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect.hpp>

// dlib dependencies for face detection
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>

#include "PDM.h"
#include "Patch_experts.h"
#include "LandmarkDetectionValidator.h"
#include "LandmarkDetectorParameters.h"
#include "FaceDetectorMTCNN.h"

namespace LandmarkDetector
{

// A main class containing all the modules required for landmark detection
// Face shape model
// Patch experts
// Optimization techniques
class CLNF{

public:

	//===========================================================================
	// Member variables that contain the model description

	// The linear 3D Point Distribution Model
    PDM					pdm;
	// The set of patch experts
	Patch_experts		patch_experts;

	// The local and global parameters describing the current model instance (current landmark detections)

	// Local parameters describing the non-rigid shape
	cv::Mat_<float>    params_local;

	// Global parameters describing the rigid shape [scale, euler_x, euler_y, euler_z, tx, ty]
	cv::Vec6f           params_global;

	// A collection of hierarchical CLNF models that can be used for refinement
	std::vector<CLNF>								hierarchical_models;
	std::vector<std::string>						hierarchical_model_names;
	std::vector<std::vector<std::pair<int,int>>>	hierarchical_mapping;
	std::vector<FaceModelParameters>				hierarchical_params;

	//==================== Helpers for face detection and landmark detection validation =========================================

	// TODO these should be static, and loading should be made easier

	// Haar cascade classifier for face detection
	cv::CascadeClassifier   face_detector_HAAR;
	std::string             haar_face_detector_location;
	
	// A HOG SVM-struct based face detector
	dlib::frontal_face_detector face_detector_HOG;

	FaceDetectorMTCNN		face_detector_MTCNN;
	std::string             mtcnn_face_detector_location;

	// Validate if the detected landmarks are correct using an SVR regressor
	DetectionValidator	landmark_validator; 

	// Indicating if landmark detection succeeded (based on SVR validator)
	bool				detection_success; 

	// Indicating if the tracking has been initialised (for video based tracking)
	bool				tracking_initialised;

	// The actual output of the regressor (-1 is perfect detection 1 is worst detection)
	float				detection_certainty;

	// Indicator if eye model is there for eye detection
	bool				eye_model;

	// the triangulation per each view (for drawing purposes only)
	std::vector<cv::Mat_<int> >	triangulations;
	
	//===========================================================================
	// Member variables that retain the state of the tracking (reflecting the state of the lastly tracked (detected) image

	// Lastly detect 2D model shape [x1,x2,...xn,y1,...yn]
	cv::Mat_<float>			detected_landmarks;
	
	// The landmark detection likelihoods (combined and per patch expert)
	float					model_likelihood;
	cv::Mat_<float>			landmark_likelihoods;
	
	// Keeping track of how many frames the tracker has failed in so far when tracking in videos
	// This is useful for knowing when to initialise and reinitialise tracking
	int failures_in_a_row;

	// A template of a face that last succeeded with tracking (useful for large motions in video)
	cv::Mat_<uchar> face_template;

	// Useful when resetting or initialising the model closer to a specific location (when multiple faces are present)
	cv::Point_<double> preference_det;

	// Tracking which view was used last
	int view_used;

	// See if the model was read in correctly
	bool loaded_successfully;

	// A default constructor
	CLNF();

	// Constructor from a model file
	CLNF(std::string fname);
	
	// Copy constructor (makes a deep copy of the detector)
	CLNF(const CLNF& other);

	// Assignment operator for lvalues (makes a deep copy of the detector)
	CLNF & operator= (const CLNF& other);

	// Empty Destructor	as the memory of every object will be managed by the corresponding libraries (no pointers)
	~CLNF(){}

	// Move constructor
	CLNF(const CLNF&& other);

	// Assignment operator for rvalues
	CLNF & operator= (const CLNF&& other);

	// Does the actual work - landmark detection
	bool DetectLandmarks(const cv::Mat_<uchar> &image, FaceModelParameters& params);
	
	// Gets the shape of the current detected landmarks in camera space (given camera calibration)
	// Can only be called after a call to DetectLandmarksInVideo or DetectLandmarksInImage
	cv::Mat_<float> GetShape(float fx, float fy, float cx, float cy) const;

	// A utility bounding box function
	cv::Rect_<float> GetBoundingBox() const;

	// Get the currently non-self occluded landmarks
	cv::Mat_<int> GetVisibilities() const;

	// Reset the model (useful if we want to completelly reinitialise, or we want to track another video)
	void Reset();

	// Reset the model, choosing the face nearest (x,y) where x and y are between 0 and 1.
	void Reset(double x, double y);

	// Reading the model in
	void Read(std::string name);
	
private:

	// Helper reading function
	bool Read_CLNF(std::string clnf_location);

	// the speedup of RLMS using precalculated KDE responses (described in Saragih 2011 RLMS paper)
	std::map<int, cv::Mat_<float> >		kde_resp_precalc;

	// The model fitting: patch response computation and optimisation steps
    bool Fit(const cv::Mat_<float>& intensity_image, const std::vector<int>& window_sizes, const FaceModelParameters& parameters);

	// Mean shift computation that uses precalculated kernel density estimators (the one actually used)
	void NonVectorisedMeanShift_precalc_kde(cv::Mat_<float>& out_mean_shifts, const std::vector<cv::Mat_<float> >& patch_expert_responses, 
		const cv::Mat_<float> &dxs, const cv::Mat_<float> &dys, int resp_size, float a, int scale, int view_id, 
		std::map<int, cv::Mat_<float> >& mean_shifts);

	// The actual model optimisation (update step), returns the model likelihood
    float NU_RLMS(cv::Vec6f& final_global, cv::Mat_<float>& final_local, const std::vector<cv::Mat_<float> >& patch_expert_responses, 
				  const cv::Vec6f& initial_global, const cv::Mat_<float>& initial_local,
		          const cv::Mat_<float>& base_shape, const cv::Matx22f& sim_img_to_ref, 
				  const cv::Matx22f& sim_ref_to_img, int resp_size, int view_idx, bool rigid, int scale, 
		          cv::Mat_<float>& landmark_lhoods, const FaceModelParameters& parameters, bool compute_lhood);

	// Generating the weight matrix for the Weighted least squares
	void GetWeightMatrix(cv::Mat_<float>& WeightMatrix, int scale, int view_id, const FaceModelParameters& parameters);

  };
  //===========================================================================
}
#endif // LANDMARK_DETECTOR_MODEL_H
