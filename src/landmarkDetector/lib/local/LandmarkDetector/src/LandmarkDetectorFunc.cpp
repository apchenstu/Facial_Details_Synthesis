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

#include "LandmarkDetectorFunc.h"
#include "RotationHelpers.h"
#include "ImageManipulationHelpers.h"

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

// System includes
#include <vector>
#include <numeric>

using namespace LandmarkDetector;

// Getting a head pose estimate from the currently detected landmarks, with appropriate correction due to the PDM assuming an orthographic camera
// which is only correct close to the centre of the image
// This method returns a corrected pose estimate with respect to world coordinates with camera at origin (0,0,0)
// The format returned is [Tx, Ty, Tz, Eul_x, Eul_y, Eul_z]
cv::Vec6f LandmarkDetector::GetPose(const CLNF& clnf_model, float fx, float fy, float cx, float cy)
{
	if (!clnf_model.detected_landmarks.empty() && clnf_model.params_global[0] != 0)
	{
		// This is used as an initial estimate for the iterative PnP algorithm
		float Z = fx / clnf_model.params_global[0];

		float X = ((clnf_model.params_global[4] - cx) * (1.0 / fx)) * Z;
		float Y = ((clnf_model.params_global[5] - cy) * (1.0 / fy)) * Z;

		// Correction for orientation

		// 2D points
		cv::Mat_<float> landmarks_2D = clnf_model.detected_landmarks;

		landmarks_2D = landmarks_2D.reshape(1, 2).t();

		// 3D points
		cv::Mat_<float> landmarks_3D;
		clnf_model.pdm.CalcShape3D(landmarks_3D, clnf_model.params_local);
		
		landmarks_3D = landmarks_3D.reshape(1, 3).t();

		// Solving the PNP model

		// The camera matrix
		cv::Matx33f camera_matrix(fx, 0, cx, 0, fy, cy, 0, 0, 1);

		cv::Vec3f vec_trans(X, Y, Z);
		cv::Vec3f vec_rot(clnf_model.params_global[1], clnf_model.params_global[2], clnf_model.params_global[3]);

		cv::solvePnP(landmarks_3D, landmarks_2D, camera_matrix, cv::Mat(), vec_rot, vec_trans, true);

		cv::Vec3f euler = Utilities::AxisAngle2Euler(vec_rot);

		return cv::Vec6f(vec_trans[0], vec_trans[1], vec_trans[2], euler[0], euler[1], euler[2]);
	}
	else
	{
		return cv::Vec6f(0, 0, 0, 0, 0, 0);
	}
}

// Getting a head pose estimate from the currently detected landmarks, with appropriate correction due to perspective projection
// This method returns a corrected pose estimate with respect to a point camera (NOTE not the world coordinates), which is useful to find out if the person is looking at a camera
// The format returned is [Tx, Ty, Tz, Eul_x, Eul_y, Eul_z]
cv::Vec6f LandmarkDetector::GetPoseWRTCamera(const CLNF& clnf_model, float fx, float fy, float cx, float cy)
{
	if (!clnf_model.detected_landmarks.empty() && clnf_model.params_global[0] != 0)
	{

		float Z = fx / clnf_model.params_global[0];

		float X = ((clnf_model.params_global[4] - cx) * (1.0 / fx)) * Z;
		float Y = ((clnf_model.params_global[5] - cy) * (1.0 / fy)) * Z;

		// Correction for orientation

		// 3D points
		cv::Mat_<float> landmarks_3D;
		clnf_model.pdm.CalcShape3D(landmarks_3D, clnf_model.params_local);

		landmarks_3D = landmarks_3D.reshape(1, 3).t();

		// 2D points
		cv::Mat_<float> landmarks_2D = clnf_model.detected_landmarks;

		landmarks_2D = landmarks_2D.reshape(1, 2).t();

		// Solving the PNP model

		// The camera matrix
		cv::Matx33f camera_matrix(fx, 0, cx, 0, fy, cy, 0, 0, 1);

		cv::Vec3f vec_trans(X, Y, Z);
		cv::Vec3f vec_rot(clnf_model.params_global[1], clnf_model.params_global[2], clnf_model.params_global[3]);

		cv::solvePnP(landmarks_3D, landmarks_2D, camera_matrix, cv::Mat(), vec_rot, vec_trans, true);

		// Here we correct for the camera orientation, for this need to determine the angle the camera makes with the head pose
		float z_x = cv::sqrt(vec_trans[0] * vec_trans[0] + vec_trans[2] * vec_trans[2]);
		float eul_x = atan2(vec_trans[1], z_x);

		float z_y = cv::sqrt(vec_trans[1] * vec_trans[1] + vec_trans[2] * vec_trans[2]);
		float eul_y = -atan2(vec_trans[0], z_y);

		cv::Matx33f camera_rotation = Utilities::Euler2RotationMatrix(cv::Vec3f(eul_x, eul_y, 0));
		cv::Matx33f head_rotation = Utilities::AxisAngle2RotationMatrix(vec_rot);

		cv::Matx33f corrected_rotation = camera_rotation * head_rotation;

		cv::Vec3f euler_corrected = Utilities::RotationMatrix2Euler(corrected_rotation);

		return cv::Vec6f(vec_trans[0], vec_trans[1], vec_trans[2], euler_corrected[0], euler_corrected[1], euler_corrected[2]);
	}
	else
	{
		return cv::Vec6f(0, 0, 0, 0, 0, 0);
	}
}

// If landmark detection in video succeeded create a template for use in simple tracking
void UpdateTemplate(const cv::Mat_<uchar> &grayscale_image, CLNF& clnf_model)
{
	cv::Rect_<float> bounding_box;
	clnf_model.pdm.CalcBoundingBox(bounding_box, clnf_model.params_global, clnf_model.params_local);
	
	// Make sure the box is not out of bounds
	cv::Rect_<int> bbox_tmp((int)bounding_box.x, (int)bounding_box.y, (int)bounding_box.width, (int)bounding_box.height);
	bounding_box = bbox_tmp & cv::Rect(0, 0, grayscale_image.cols, grayscale_image.rows);

	clnf_model.face_template = grayscale_image(bounding_box).clone();
}

// This method uses basic template matching in order to allow for better tracking of fast moving faces
void CorrectGlobalParametersVideo(const cv::Mat_<uchar> &grayscale_image, CLNF& clnf_model, const FaceModelParameters& params)
{
	cv::Rect_<float> init_box;
	clnf_model.pdm.CalcBoundingBox(init_box, clnf_model.params_global, clnf_model.params_local);

	cv::Rect roi(init_box.x - init_box.width/2, init_box.y - init_box.height/2, init_box.width * 2, init_box.height * 2);
	roi = roi & cv::Rect(0, 0, grayscale_image.cols, grayscale_image.rows);

	int off_x = roi.x;
	int off_y = roi.y;

	float scaling = params.face_template_scale / clnf_model.params_global[0];
	cv::Mat_<uchar> image;
	if(scaling < 1)
	{
		cv::resize(clnf_model.face_template, clnf_model.face_template, cv::Size(), scaling, scaling);
		cv::resize(grayscale_image(roi), image, cv::Size(), scaling, scaling);
	}
	else
	{
		scaling = 1;
		image = grayscale_image(roi).clone();
	}
		
	// Resizing the template			
	cv::Mat corr_out;
	cv::matchTemplate(image, clnf_model.face_template, corr_out, cv::TM_CCOEFF_NORMED);

	// Actually matching it
	//double min, max;
	int max_loc[2];

	cv::minMaxIdx(corr_out, NULL, NULL, NULL, max_loc);

	cv::Rect_<float> out_bbox(max_loc[1]/scaling + off_x, max_loc[0]/scaling + off_y, clnf_model.face_template.rows / scaling, clnf_model.face_template.cols / scaling);

	float shift_x = out_bbox.x - init_box.x;
	float shift_y = out_bbox.y - init_box.y;
			
	clnf_model.params_global[4] = clnf_model.params_global[4] + shift_x;
	clnf_model.params_global[5] = clnf_model.params_global[5] + shift_y;
	
}

bool LandmarkDetector::DetectLandmarksInVideo(const cv::Mat &rgb_image, CLNF& clnf_model, FaceModelParameters& params, cv::Mat& grayscale_image)
{
	// First need to decide if the landmarks should be "detected" or "tracked"
	// Detected means running face detection and a larger search area, tracked means initialising from previous step
	// and using a smaller search area

	if(grayscale_image.empty())
	{
		Utilities::ConvertToGrayscale_8bit(rgb_image, grayscale_image);
	}

	// Indicating that this is a first detection in video sequence or after restart
	bool initial_detection = !clnf_model.tracking_initialised;

	// Only do it if there was a face detection at all
	if(clnf_model.tracking_initialised)
	{

		// The area of interest search size will depend if the previous track was successful
		if(!clnf_model.detection_success)
		{
			params.window_sizes_current = params.window_sizes_init;
		}
		else
		{
			params.window_sizes_current = params.window_sizes_small;
		}

		// Before the expensive landmark detection step apply a quick template tracking approach
		if(params.use_face_template && !clnf_model.face_template.empty() && clnf_model.detection_success)
		{
			CorrectGlobalParametersVideo(grayscale_image, clnf_model, params);
		}

		bool track_success = clnf_model.DetectLandmarks(grayscale_image, params);
		
		if(!track_success)
		{
			// Make a record that tracking failed
			clnf_model.failures_in_a_row++;
		}
		else
		{
			// indicate that tracking is a success
			clnf_model.failures_in_a_row = -1;		
			
			if(params.use_face_template)
			{
				UpdateTemplate(grayscale_image, clnf_model);
			}
		}
	}

	// This is used for both detection (if it the tracking has not been initialised yet) or if the tracking failed (however we do this every n frames, for speed)
	// This also has the effect of an attempt to reinitialise just after the tracking has failed, which is useful during large motions
	if((!clnf_model.tracking_initialised && (clnf_model.failures_in_a_row + 1) % (params.reinit_video_every * 6) == 0) 
		|| (clnf_model.tracking_initialised && !clnf_model.detection_success && params.reinit_video_every > 0 && clnf_model.failures_in_a_row % params.reinit_video_every == 0))
	{

		cv::Rect_<float> bounding_box;
		
		// If the face detector has not been initialised and we're using it, then read it in
		if(clnf_model.face_detector_HAAR.empty() && params.curr_face_detector == params.HAAR_DETECTOR)
		{
			clnf_model.face_detector_HAAR.load(params.haar_face_detector_location);
			clnf_model.haar_face_detector_location = params.haar_face_detector_location;
		}
		if (clnf_model.face_detector_MTCNN.empty() && params.curr_face_detector == params.MTCNN_DETECTOR)
		{
			clnf_model.face_detector_MTCNN.Read(params.mtcnn_face_detector_location);
			clnf_model.mtcnn_face_detector_location = params.mtcnn_face_detector_location;

			// If the model is still empty default to HOG
			if (clnf_model.face_detector_MTCNN.empty())
			{
				std::cout << "INFO: defaulting to HOG-SVM face detector" << std::endl;
				params.curr_face_detector = LandmarkDetector::FaceModelParameters::HOG_SVM_DETECTOR;
			}

		}

		cv::Point preference_det(-1, -1);
		if(clnf_model.preference_det.x != -1 && clnf_model.preference_det.y != -1)
		{
			preference_det.x = clnf_model.preference_det.x * grayscale_image.cols;
			preference_det.y = clnf_model.preference_det.y * grayscale_image.rows;
			clnf_model.preference_det = cv::Point(-1, -1);
		}

		bool face_detection_success;
		if(params.curr_face_detector == FaceModelParameters::HOG_SVM_DETECTOR)
		{
			float confidence;
			face_detection_success = LandmarkDetector::DetectSingleFaceHOG(bounding_box, grayscale_image, clnf_model.face_detector_HOG, confidence, preference_det);
		}
		else if(params.curr_face_detector == FaceModelParameters::HAAR_DETECTOR)
		{
			face_detection_success = LandmarkDetector::DetectSingleFace(bounding_box, grayscale_image, clnf_model.face_detector_HAAR, preference_det);
		}
		else if (params.curr_face_detector == FaceModelParameters::MTCNN_DETECTOR)
		{
			float confidence;
			face_detection_success = LandmarkDetector::DetectSingleFaceMTCNN(bounding_box, rgb_image, clnf_model.face_detector_MTCNN, confidence, preference_det);
		}

		// Attempt to detect landmarks using the detected face (if unseccessful the detection will be ignored)
		if(face_detection_success)
		{
			// Indicate that tracking has started as a face was detected
			clnf_model.tracking_initialised = true;
						
			// Keep track of old model values so that they can be restored if redetection fails
			cv::Vec6f params_global_init = clnf_model.params_global;
			cv::Mat_<float> params_local_init = clnf_model.params_local.clone();
			float likelihood_init = clnf_model.model_likelihood;
			cv::Mat_<float> detected_landmarks_init = clnf_model.detected_landmarks.clone();
			cv::Mat_<float> landmark_likelihoods_init = clnf_model.landmark_likelihoods.clone();

			// Use the detected bounding box and empty local parameters
			clnf_model.params_local.setTo(0);
			clnf_model.pdm.CalcParams(clnf_model.params_global, bounding_box, clnf_model.params_local);		

			// Make sure the search size is large
			params.window_sizes_current = params.window_sizes_init;

			// TODO rem (should the multi-hyp version be only for CEN and not CLNF?), otherwise poss too slow, and poss not accurate
			//bool landmark_detection_success = clnf_model.DetectLandmarks(grayscale_image, params);

			// Do the actual landmark detection (and keep it only if successful)
			// Perform multi-hypothesis detection here (as face detector can pick up multiple of them)
			params.multi_view = true;
			bool landmark_detection_success = DetectLandmarksInImage(rgb_image, bounding_box, clnf_model, params, grayscale_image);
			params.multi_view = false;


			// If landmark reinitialisation unsucessful continue from previous estimates
			// if it's initial detection however, do not care if it was successful as the validator might be wrong, so continue trackig
			// regardless
			if(!initial_detection && !landmark_detection_success)
			{

				// Restore previous estimates
				clnf_model.params_global = params_global_init;
				clnf_model.params_local = params_local_init.clone();
				clnf_model.pdm.CalcShape2D(clnf_model.detected_landmarks, clnf_model.params_local, clnf_model.params_global);
				clnf_model.model_likelihood = likelihood_init;
				clnf_model.detected_landmarks = detected_landmarks_init.clone();
				clnf_model.landmark_likelihoods = landmark_likelihoods_init.clone();

				return false;
			}
			else
			{
				clnf_model.failures_in_a_row = -1;			
				
				if(params.use_face_template)
				{
					UpdateTemplate(grayscale_image, clnf_model);
				}

				return true;
			}
		}
	}

	// if the model has not been initialised yet class it as a failure
	if(!clnf_model.tracking_initialised)
	{
		clnf_model.failures_in_a_row++;
	}

	// un-initialise the tracking
	if(	clnf_model.failures_in_a_row > 100)
	{
		clnf_model.tracking_initialised = false;
	}

	return clnf_model.detection_success;
	
}

bool LandmarkDetector::DetectLandmarksInVideo(const cv::Mat &rgb_image, const cv::Rect_<double> bounding_box, CLNF& clnf_model, FaceModelParameters& params, cv::Mat &grayscale_image)
{
	if(bounding_box.width > 0)
	{
		// calculate the local and global parameters from the generated 2D shape (mapping from the 2D to 3D because camera params are unknown)
		clnf_model.params_local.setTo(0);
		clnf_model.pdm.CalcParams(clnf_model.params_global, bounding_box, clnf_model.params_local);		

		// indicate that face was detected so initialisation is not necessary
		clnf_model.tracking_initialised = true;
	}

	return DetectLandmarksInVideo(rgb_image, clnf_model, params, grayscale_image);

}

//================================================================================================================
// Landmark detection in image, need to provide an image and optionally CLNF model together with parameters (default values work well)
// Optionally can provide a bounding box in which detection is performed (this is useful if multiple faces are to be detected in images)
//================================================================================================================

bool DetectLandmarksInImageMultiHypBasic(const cv::Mat_<uchar> &grayscale_image, std::vector<cv::Vec3d> rotation_hypotheses, 
	const cv::Rect_<double> bounding_box, CLNF& clnf_model, FaceModelParameters& params)
{

	// Use the initialisation size for the landmark detection
	params.window_sizes_current = params.window_sizes_init;

	// Store the current best estimate
	float best_likelihood;
	float best_detection_certainty;
	cv::Vec6f best_global_parameters;
	cv::Mat_<float> best_local_parameters;
	cv::Mat_<float> best_detected_landmarks;
	cv::Mat_<float> best_landmark_likelihoods;
	bool best_success;

	// The hierarchical model parameters
	std::vector<float> best_likelihood_h(clnf_model.hierarchical_models.size());
	std::vector<cv::Vec6f> best_global_parameters_h(clnf_model.hierarchical_models.size());
	std::vector<cv::Mat_<float>> best_local_parameters_h(clnf_model.hierarchical_models.size());
	std::vector<cv::Mat_<float>> best_detected_landmarks_h(clnf_model.hierarchical_models.size());
	std::vector<cv::Mat_<float>> best_landmark_likelihoods_h(clnf_model.hierarchical_models.size());

	for (size_t hypothesis = 0; hypothesis < rotation_hypotheses.size(); ++hypothesis)
	{
		// Reset the potentially set clnf_model parameters
		clnf_model.params_local.setTo(0.0);

		for (size_t part = 0; part < clnf_model.hierarchical_models.size(); ++part)
		{
			clnf_model.hierarchical_models[part].params_local.setTo(0.0);
		}

		// calculate the local and global parameters from the generated 2D shape (mapping from the 2D to 3D because camera params are unknown)
		clnf_model.pdm.CalcParams(clnf_model.params_global, bounding_box, clnf_model.params_local, rotation_hypotheses[hypothesis]);
	
		bool success = clnf_model.DetectLandmarks(grayscale_image, params);	

		if (hypothesis == 0 || best_likelihood < clnf_model.model_likelihood)
		{
			best_likelihood = clnf_model.model_likelihood;
			best_global_parameters = clnf_model.params_global;
			best_local_parameters = clnf_model.params_local.clone();
			best_detected_landmarks = clnf_model.detected_landmarks.clone();
			best_landmark_likelihoods = clnf_model.landmark_likelihoods.clone();
			best_detection_certainty = clnf_model.detection_certainty;
			best_success = success;
			
			for (size_t part = 0; part < clnf_model.hierarchical_models.size(); ++part)
			{
				best_likelihood_h[part] = clnf_model.hierarchical_models[part].model_likelihood;
				best_global_parameters_h[part] = clnf_model.hierarchical_models[part].params_global;
				best_local_parameters_h[part] = clnf_model.hierarchical_models[part].params_local.clone();
				best_detected_landmarks_h[part] = clnf_model.hierarchical_models[part].detected_landmarks.clone();
				best_landmark_likelihoods_h[part] = clnf_model.hierarchical_models[part].landmark_likelihoods.clone();
			}
		}

	}

	// Store the best estimates in the clnf_model
	clnf_model.model_likelihood = best_likelihood;
	clnf_model.params_global = best_global_parameters;
	clnf_model.params_local = best_local_parameters.clone();
	clnf_model.detected_landmarks = best_detected_landmarks.clone();
	clnf_model.detection_success = best_success;
	clnf_model.landmark_likelihoods = best_landmark_likelihoods.clone();
	clnf_model.detection_certainty = best_detection_certainty;

	for (size_t part = 0; part < clnf_model.hierarchical_models.size(); ++part)
	{
		clnf_model.hierarchical_models[part].params_global = best_global_parameters_h[part];
		clnf_model.hierarchical_models[part].params_local = best_local_parameters_h[part].clone();
		clnf_model.hierarchical_models[part].detected_landmarks = best_detected_landmarks_h[part].clone();
		clnf_model.hierarchical_models[part].landmark_likelihoods = best_landmark_likelihoods_h[part].clone();
	}

	return best_success;


}

// Helper index sorting function
template <typename T> std::vector<size_t> sort_indexes(const std::vector<T> &v) {

	// initialize original index locations
	std::vector<size_t> idx(v.size());
	std::iota(idx.begin(), idx.end(), 0);

	// sort indexes based on comparing values in v
	std::sort(idx.begin(), idx.end(),
		[&v](size_t i1, size_t i2) {return v[i1] > v[i2]; });

	return idx;
}

bool DetectLandmarksInImageMultiHypEarlyTerm(const cv::Mat_<uchar> &grayscale_image, std::vector<cv::Vec3d> rotation_hypotheses, 
	const cv::Rect_<double> bounding_box, CLNF& clnf_model, FaceModelParameters& params)
{
	FaceModelParameters old_params(params);
	
	// Use the initialisation size for the landmark detection
	params.window_sizes_current = params.window_sizes_init;

	bool early_term = false;

	// Setup the parameters accordingly
	// Only do the first iteration
	for (size_t i = 1; i < params.window_sizes_current.size(); ++i)
	{
		params.window_sizes_current[i] = 0;
	}
	params.refine_hierarchical = false;
	params.validate_detections = false;

	bool success = false;

	// Keeping track of converges
	std::vector<float> likelihoods;
	std::vector<cv::Vec6f> global_parameters;
	std::vector<cv::Mat_<float>> local_parameters;

	for (size_t hypothesis = 0; hypothesis < rotation_hypotheses.size(); ++hypothesis)
	{
		// Reset the potentially set clnf_model parameters
		clnf_model.params_local.setTo(0.0);

		for (size_t part = 0; part < clnf_model.hierarchical_models.size(); ++part)
		{
			clnf_model.hierarchical_models[part].params_local.setTo(0.0);
		}

		// calculate the local and global parameters from the generated 2D shape (mapping from the 2D to 3D because camera params are unknown)
		clnf_model.pdm.CalcParams(clnf_model.params_global, bounding_box, clnf_model.params_local, rotation_hypotheses[hypothesis]);

		// Perform landmark detection in first scale
		clnf_model.DetectLandmarks(grayscale_image, params);

		float lhood = clnf_model.model_likelihood * clnf_model.patch_experts.early_term_weights[clnf_model.view_used] + clnf_model.patch_experts.early_term_biases[clnf_model.view_used];

		// If likelihood higher than cutoff continue on this model
		if (lhood > clnf_model.patch_experts.early_term_cutoffs[clnf_model.view_used])
		{
			params.refine_hierarchical = old_params.refine_hierarchical;
			params.window_sizes_current = params.window_sizes_init;
			params.window_sizes_current[0] = 0;
			params.validate_detections = old_params.validate_detections;
			success = clnf_model.DetectLandmarks(grayscale_image, params);
			early_term = true;
			break;
		}
		else
		{
			likelihoods.push_back(lhood);
			global_parameters.push_back(clnf_model.params_global);
			local_parameters.push_back(clnf_model.params_local);
		}
	}


	if (!early_term)
	{

		// Store the current best estimate
		float best_likelihood;
		cv::Vec6f best_global_parameters;
		cv::Mat_<float> best_local_parameters;
		cv::Mat_<float> best_detected_landmarks;
		cv::Mat_<float> best_landmark_likelihoods;
		bool best_success;

		// The hierarchical model parameters
		std::vector<float> best_likelihood_h(clnf_model.hierarchical_models.size());
		std::vector<cv::Vec6f> best_global_parameters_h(clnf_model.hierarchical_models.size());
		std::vector<cv::Mat_<float>> best_local_parameters_h(clnf_model.hierarchical_models.size());
		std::vector<cv::Mat_<float>> best_detected_landmarks_h(clnf_model.hierarchical_models.size());
		std::vector<cv::Mat_<float>> best_landmark_likelihoods_h(clnf_model.hierarchical_models.size());

		// Sort the likelihoods and pick the best top 3 models
		std::vector<size_t> indices = sort_indexes(likelihoods);

		// Pick 3 best hypotheses and complete them
		size_t max = indices.size() >= 3 ? 3 : indices.size();

		params.refine_hierarchical = old_params.refine_hierarchical;
		params.window_sizes_current = params.window_sizes_init;
		params.window_sizes_current[0] = 0;
		params.validate_detections = old_params.validate_detections;


		for (size_t i = 0; i < max; ++i)
		{
			// Reset the potentially set clnf_model parameters
			clnf_model.params_local = local_parameters[indices[i]];
			clnf_model.params_global = global_parameters[indices[i]];
			for (size_t part = 0; part < clnf_model.hierarchical_models.size(); ++part)
			{
				clnf_model.hierarchical_models[part].params_local.setTo(0.0);
			}
	
			// Perform landmark detection in first scale
			success = clnf_model.DetectLandmarks(grayscale_image, params);

			if (i == 0 || best_likelihood < clnf_model.model_likelihood)
			{
				best_likelihood = clnf_model.model_likelihood;
				best_global_parameters = clnf_model.params_global;
				best_local_parameters = clnf_model.params_local.clone();
				best_detected_landmarks = clnf_model.detected_landmarks.clone();
				best_landmark_likelihoods = clnf_model.landmark_likelihoods.clone();
				best_success = success;

				for (size_t part = 0; part < clnf_model.hierarchical_models.size(); ++part)
				{
					best_likelihood_h[part] = clnf_model.hierarchical_models[part].model_likelihood;
					best_global_parameters_h[part] = clnf_model.hierarchical_models[part].params_global;
					best_local_parameters_h[part] = clnf_model.hierarchical_models[part].params_local.clone();
					best_detected_landmarks_h[part] = clnf_model.hierarchical_models[part].detected_landmarks.clone();
					best_landmark_likelihoods_h[part] = clnf_model.hierarchical_models[part].landmark_likelihoods.clone();
				}
			}

		}

		// Store the best estimates in the clnf_model
		clnf_model.model_likelihood = best_likelihood;
		clnf_model.params_global = best_global_parameters;
		clnf_model.params_local = best_local_parameters.clone();
		clnf_model.detected_landmarks = best_detected_landmarks.clone();
		clnf_model.detection_success = best_success;
		clnf_model.landmark_likelihoods = best_landmark_likelihoods.clone();

		for (size_t part = 0; part < clnf_model.hierarchical_models.size(); ++part)
		{
			clnf_model.hierarchical_models[part].params_global = best_global_parameters_h[part];
			clnf_model.hierarchical_models[part].params_local = best_local_parameters_h[part].clone();
			clnf_model.hierarchical_models[part].detected_landmarks = best_detected_landmarks_h[part].clone();
			clnf_model.hierarchical_models[part].landmark_likelihoods = best_landmark_likelihoods_h[part].clone();
		}

	}

	params = old_params;

	return success;

}


// This is the one where the actual work gets done, other DetectLandmarksInImage calls lead to this one
bool LandmarkDetector::DetectLandmarksInImage(const cv::Mat &rgb_image, const cv::Rect_<double> bounding_box, CLNF& clnf_model, FaceModelParameters& params, cv::Mat &grayscale_image)
{

	if (grayscale_image.empty())
	{
		Utilities::ConvertToGrayscale_8bit(rgb_image, grayscale_image);
	}

	// Can have multiple hypotheses
	std::vector<cv::Vec3d> rotation_hypotheses;

	if(params.multi_view)
	{
		// Try out different orientation initialisations
		// It is possible to add other orientation hypotheses easilly by just pushing to this vector
		rotation_hypotheses.push_back(cv::Vec3d(0,0,0));
		rotation_hypotheses.push_back(cv::Vec3d(0, -0.5236, 0));
		rotation_hypotheses.push_back(cv::Vec3d(0, 0.5236,0));
		rotation_hypotheses.push_back(cv::Vec3d(0, -0.96, 0));
		rotation_hypotheses.push_back(cv::Vec3d(0, 0.96, 0));
		rotation_hypotheses.push_back(cv::Vec3d(0, 0, 0.5236));
		rotation_hypotheses.push_back(cv::Vec3d(0, 0, -0.5236));
		rotation_hypotheses.push_back(cv::Vec3d(0, -1.57, 0));
		rotation_hypotheses.push_back(cv::Vec3d(0, 1.57, 0));
		rotation_hypotheses.push_back(cv::Vec3d(0, -1.22, 0.698));
		rotation_hypotheses.push_back(cv::Vec3d(0, 1.22, -0.698));
	}
	else
	{
		// Assume the face is close to frontal
		rotation_hypotheses.push_back(cv::Vec3d(0,0,0));
	}
	
	bool success;

	// Either use basic multi-hypothesis testing or clever testing if early termination parameters are present
	if(clnf_model.patch_experts.early_term_biases.size() == 0)
	{
		success = DetectLandmarksInImageMultiHypBasic(grayscale_image, rotation_hypotheses, bounding_box, clnf_model, params);
	}
	else
	{
		success = DetectLandmarksInImageMultiHypEarlyTerm(grayscale_image, rotation_hypotheses, bounding_box, clnf_model, params);
	}
	return success;
}

bool LandmarkDetector::DetectLandmarksInImage(const cv::Mat &rgb_image, CLNF& clnf_model, FaceModelParameters& params, cv::Mat &grayscale_image)
{
	if (grayscale_image.empty())
	{
		Utilities::ConvertToGrayscale_8bit(rgb_image, grayscale_image);
	}

	cv::Rect_<float> bounding_box;

	// If the face detector has not been initialised read it in
	if(clnf_model.face_detector_HAAR.empty() && params.curr_face_detector == FaceModelParameters::HAAR_DETECTOR)
	{
		clnf_model.face_detector_HAAR.load(params.haar_face_detector_location);
		clnf_model.haar_face_detector_location = params.haar_face_detector_location;
	}
	
	if (clnf_model.face_detector_MTCNN.empty() && params.curr_face_detector == FaceModelParameters::MTCNN_DETECTOR)
	{
		clnf_model.face_detector_MTCNN.Read(params.mtcnn_face_detector_location);
		clnf_model.mtcnn_face_detector_location = params.mtcnn_face_detector_location;

		// If the model is still empty default to HOG
		if (clnf_model.face_detector_MTCNN.empty())
		{
			std::cout << "INFO: defaulting to HOG-SVM face detector" << std::endl;
			params.curr_face_detector = LandmarkDetector::FaceModelParameters::HOG_SVM_DETECTOR;
		}

	}

	// Detect the face first
	if(params.curr_face_detector == FaceModelParameters::HOG_SVM_DETECTOR)
	{
		float confidence;
		LandmarkDetector::DetectSingleFaceHOG(bounding_box, grayscale_image, clnf_model.face_detector_HOG, confidence);
	}
	else if(params.curr_face_detector == FaceModelParameters::HAAR_DETECTOR)
	{
		LandmarkDetector::DetectSingleFace(bounding_box, rgb_image, clnf_model.face_detector_HAAR);
	}
	else if (params.curr_face_detector == FaceModelParameters::MTCNN_DETECTOR)
	{
		float confidence;
		LandmarkDetector::DetectSingleFaceMTCNN(bounding_box, rgb_image, clnf_model.face_detector_MTCNN, confidence);
	}

	if(bounding_box.width == 0)
	{
		return false;
	}
	else
	{
		return DetectLandmarksInImage(rgb_image, bounding_box, clnf_model, params, grayscale_image);
	}
}
