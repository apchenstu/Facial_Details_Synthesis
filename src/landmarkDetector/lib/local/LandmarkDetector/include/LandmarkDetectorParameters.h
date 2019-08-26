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

//  Parameters of the CE-CLM, CLNF, and CLM trackers
#ifndef LANDMARK_DETECTOR_PARAM_H
#define LANDMARK_DETECTOR_PARAM_H

#include <vector>

namespace LandmarkDetector
{

struct FaceModelParameters
{

	// A number of RLMS or NU-RLMS iterations
	int num_optimisation_iteration;
	
	// Should pose be limited to 180 degrees frontal
	bool limit_pose;
	
	// Should face validation be done
	bool validate_detections;

	// Landmark detection validator boundary for correct detection, the regressor output 1 (perfect alignment) 0 (bad alignment), 
	float validation_boundary;

	// Used when tracking is going well
	std::vector<int> window_sizes_small;

	// Used when initialising or tracking fails
	std::vector<int> window_sizes_init;
	
	// Used for the current frame
	std::vector<int> window_sizes_current;
	
	// How big is the tracking template that helps with large motions
	float face_template_scale;	
	bool use_face_template;

	// Where to load the model from
	std::string model_location;
	
	// this is used for the smooting of response maps (KDE sigma)
	float sigma;

	float reg_factor;	// weight put to regularisation
	float weight_factor; // factor for weighted least squares

	// should multiple views be considered during reinit
	bool multi_view;
	
	// Based on model location, this affects the parameter settings
	enum LandmarkDetector { CLM_DETECTOR, CLNF_DETECTOR, CECLM_DETECTOR };
	LandmarkDetector curr_landmark_detector;

	// How often should face detection be used to attempt reinitialisation, every n frames (set to negative not to reinit)
	int reinit_video_every;

	// Determining which face detector to use for (re)initialisation, HAAR is quicker but provides more false positives and is not goot for in-the-wild conditions
	// Also HAAR detector can detect smaller faces while HOG SVM is only capable of detecting faces at least 70px across
	// MTCNN detector is much more accurate that the other two, and is even suitable for profile faces, but it is somewhat slower
	enum FaceDetector{HAAR_DETECTOR, HOG_SVM_DETECTOR, MTCNN_DETECTOR};

	std::string haar_face_detector_location;
	std::string mtcnn_face_detector_location;
	FaceDetector curr_face_detector;

	// Should the model be refined hierarchically (if available)
	bool refine_hierarchical;

	// Should the parameters be refined for different scales
	bool refine_parameters;

	FaceModelParameters();

	FaceModelParameters(std::vector<std::string> &arguments);

	private:
		void init();
		void check_model_path(const std::string& root = "/");

};

}

#endif // LANDMARK_DETECTOR_PARAM_H
