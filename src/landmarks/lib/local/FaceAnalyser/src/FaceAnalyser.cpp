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
#include <stdafx_fa.h>

#include "FaceAnalyser.h"

// Local includes
#include "Face_utils.h"

using namespace FaceAnalysis;

// Constructor from a model file (or a default one if not provided
FaceAnalyser::FaceAnalyser(const FaceAnalysis::FaceAnalyserParameters& face_analyser_params)
{
	this->Read(face_analyser_params.getModelLoc());
		
	align_mask = face_analyser_params.getAlignMask();
	align_scale_out = face_analyser_params.getSimScaleOut();
	align_width_out = face_analyser_params.getSimSizeOut();
	align_height_out = face_analyser_params.getSimSizeOut();

	align_scale_au = face_analyser_params.sim_scale_au;
	align_width_au = face_analyser_params.sim_size_au;
	align_height_au = face_analyser_params.sim_size_au;

	// Initialise the histograms that will represent bins from 0 - 1 (as HoG values are only stored as those)
	num_bins_hog = 1000;
	max_val_hog = 1;
	min_val_hog = -0.005;

	// The geometry histogram ranges from -60 to 60
	num_bins_geom = 10000;
	max_val_geom = 60;
	min_val_geom = -60;
		
	// Keep track for how many frames have been tracked so far
	frames_tracking = 0;

	// If the model used is dynamic (person callibration and video correction)
	dynamic = face_analyser_params.getDynamic();

	out_grayscale = face_analyser_params.grayscale;

	if(face_analyser_params.getOrientationBins().empty())
	{
		// Just using frontal currently
		head_orientations.push_back(cv::Vec3d(0,0,0));
	}
	else
	{
		head_orientations = face_analyser_params.getOrientationBins();
	}
	hog_hist_sum.resize(head_orientations.size());
	face_image_hist_sum.resize(head_orientations.size());
	hog_desc_hist.resize(head_orientations.size());
	geom_hist_sum = 0;
	face_image_hist.resize(head_orientations.size());

	au_prediction_correction_count.resize(head_orientations.size(), 0);
	au_prediction_correction_histogram.resize(head_orientations.size());
	dyn_scaling.resize(head_orientations.size());

}

// Utility for getting the names of returned AUs (presence)
std::vector<std::string> FaceAnalyser::GetAUClassNames() const
{
	std::vector<std::string> au_class_names_all;
	std::vector<std::string> au_class_names_stat = AU_SVM_static_appearance_lin.GetAUNames();
	std::vector<std::string> au_class_names_dyn = AU_SVM_dynamic_appearance_lin.GetAUNames();

	for (size_t i = 0; i < au_class_names_stat.size(); ++i)
	{
		au_class_names_all.push_back(au_class_names_stat[i]);
	}
	for (size_t i = 0; i < au_class_names_dyn.size(); ++i)
	{
		au_class_names_all.push_back(au_class_names_dyn[i]);
	}

	return au_class_names_all;
}

// Utility for getting the names of returned AUs (intensity)
std::vector<std::string> FaceAnalyser::GetAURegNames() const
{
	std::vector<std::string> au_reg_names_all;
	std::vector<std::string> au_reg_names_stat = AU_SVR_static_appearance_lin_regressors.GetAUNames();
	std::vector<std::string> au_reg_names_dyn = AU_SVR_dynamic_appearance_lin_regressors.GetAUNames();

	for (size_t i = 0; i < au_reg_names_stat.size(); ++i)
	{
		au_reg_names_all.push_back(au_reg_names_stat[i]);
	}
	for (size_t i = 0; i < au_reg_names_dyn.size(); ++i)
	{
		au_reg_names_all.push_back(au_reg_names_dyn[i]);
	}

	return au_reg_names_all;
}

std::vector<bool> FaceAnalyser::GetDynamicAUClass() const
{
	std::vector<bool> au_dynamic_class;
	std::vector<std::string> au_class_names_stat = AU_SVM_static_appearance_lin.GetAUNames();
	std::vector<std::string> au_class_names_dyn = AU_SVM_dynamic_appearance_lin.GetAUNames();

	for (size_t i = 0; i < au_class_names_stat.size(); ++i)
	{
		au_dynamic_class.push_back(false);
	}
	for (size_t i = 0; i < au_class_names_dyn.size(); ++i)
	{
		au_dynamic_class.push_back(true);
	}

	return au_dynamic_class;
}

std::vector<std::pair<std::string, bool>> FaceAnalyser::GetDynamicAUReg() const
{
	std::vector<std::pair<std::string, bool>> au_dynamic_reg;
	std::vector<std::string> au_reg_names_stat = AU_SVR_static_appearance_lin_regressors.GetAUNames();
	std::vector<std::string> au_reg_names_dyn = AU_SVR_dynamic_appearance_lin_regressors.GetAUNames();

	for (size_t i = 0; i < au_reg_names_stat.size(); ++i)
	{
		au_dynamic_reg.push_back(std::pair<std::string, bool>(au_reg_names_stat[i], false));
	}
	for (size_t i = 0; i < au_reg_names_dyn.size(); ++i)
	{
		au_dynamic_reg.push_back(std::pair<std::string, bool>(au_reg_names_dyn[i], true));
	}

	return au_dynamic_reg;
}

cv::Mat_<int> FaceAnalyser::GetTriangulation()
{
	return triangulation.clone();
}

void FaceAnalyser::GetLatestHOG(cv::Mat_<double>& hog_descriptor, int& num_rows, int& num_cols)
{
	hog_descriptor = this->hog_desc_frame.clone();

	if(!hog_desc_frame.empty())
	{
		num_rows = this->num_hog_rows;
		num_cols = this->num_hog_cols;
	}
	else
	{
		num_rows = 0;
		num_cols = 0;
	}
}

void FaceAnalyser::GetLatestAlignedFace(cv::Mat& image)
{
	image = this->aligned_face_for_output.clone();
}

void FaceAnalyser::GetLatestNeutralHOG(cv::Mat_<double>& hog_descriptor, int& num_rows, int& num_cols)
{
	hog_descriptor = this->hog_desc_median;
	if(!hog_desc_median.empty())
	{
		num_rows = this->num_hog_rows;
		num_cols = this->num_hog_cols;
	}
	else
	{
		num_rows = 0;
		num_cols = 0;
	}
}

// Getting the closest view center based on orientation
int GetViewId(const std::vector<cv::Vec3d> orientations_all, const cv::Vec3d& orientation)
{
	int id = 0;

	double dbest = -1.0;

	for(size_t i = 0; i < orientations_all.size(); i++)
	{
	
		// Distance to current view
		double d = cv::norm(orientation, orientations_all[i]);

		if(i == 0 || d < dbest)
		{
			dbest = d;
			id = (int) i;
		}
	}
	return id;
	
}

void FaceAnalyser::PredictStaticAUsAndComputeFeatures(const cv::Mat& frame, const cv::Mat_<float>& detected_landmarks)
{
	
	// Extract shape parameters from the detected landmarks
	cv::Vec6f params_global;
	cv::Mat_<float> params_local;
	pdm.CalcParams(params_global, params_local, detected_landmarks);

	// The aligned face requirement for AUs
	AlignFaceMask(aligned_face_for_au, frame, detected_landmarks, params_global, pdm, triangulation, true, align_scale_au, align_width_au, align_height_au);

	// If the aligned face for AU matches the output requested one, just reuse it, else compute it
	if (align_scale_out == align_scale_au && align_width_out == align_width_au && align_height_out == align_height_au && align_mask)
	{
		aligned_face_for_output = aligned_face_for_au.clone();
	}
	else
	{
		if (align_mask)
		{
			AlignFaceMask(aligned_face_for_output, frame, detected_landmarks, params_global, pdm, triangulation, true, align_scale_out, align_width_out, align_height_out);
		}
		else
		{
			AlignFace(aligned_face_for_output, frame, detected_landmarks, params_global, pdm, true, align_scale_out, align_width_out, align_height_out);
		}
	}

	// Extract HOG descriptor from the frame and convert it to a useable format
	cv::Mat_<double> hog_descriptor;
	Extract_FHOG_descriptor(hog_descriptor, aligned_face_for_au, this->num_hog_rows, this->num_hog_cols);

	// Store the descriptor
	hog_desc_frame = hog_descriptor;

	cv::Vec3d curr_orient(params_global[1], params_global[2], params_global[3]);
	int orientation_to_use = GetViewId(this->head_orientations, curr_orient);
	
	// Geom descriptor and its median, TODO these should be floats?
	params_local = params_local.t();
	params_local.convertTo(geom_descriptor_frame, CV_64F);
	
	cv::Mat_<double> princ_comp_d;
	pdm.princ_comp.convertTo(princ_comp_d, CV_64F);

	// Stack with the actual feature point locations (without mean)
	cv::Mat_<double> locs = princ_comp_d * geom_descriptor_frame.t();

	cv::hconcat(locs.t(), geom_descriptor_frame.clone(), geom_descriptor_frame);
	
	// First convert the face image to double representation as a row vector, TODO rem
	//cv::Mat_<uchar> aligned_face_cols(1, aligned_face_for_au.cols * aligned_face_for_au.rows * aligned_face_for_au.channels(), aligned_face_for_au.data, 1);
	//cv::Mat_<double> aligned_face_cols_double;
	//aligned_face_cols.convertTo(aligned_face_cols_double, CV_64F);
	
	// Perform AU prediction	
	auto AU_predictions_intensity = PredictCurrentAUs(orientation_to_use);
	auto AU_predictions_occurence = PredictCurrentAUsClass(orientation_to_use);

	// Make sure intensity is within range (0-5)
	for (size_t au = 0; au < AU_predictions_intensity.size(); ++au)
	{
		if (AU_predictions_intensity[au].second < 0)
			AU_predictions_intensity[au].second = 0;

		if (AU_predictions_intensity[au].second > 5)
			AU_predictions_intensity[au].second = 5;
	}
	
	AU_predictions_reg = AU_predictions_intensity;
	AU_predictions_class = AU_predictions_occurence;

}

void FaceAnalyser::AddNextFrame(const cv::Mat& frame, const cv::Mat_<float>& detected_landmarks, bool success, double timestamp_seconds, bool online)
{

	frames_tracking++;

	// Extract shape parameters from the detected landmarks
	cv::Vec6f params_global;
	cv::Mat_<float> params_local;

	// First align the face if tracking was successfull
	if(success)
	{

		pdm.CalcParams(params_global, params_local, detected_landmarks);

		// The aligned face requirement for AUs
		AlignFaceMask(aligned_face_for_au, frame, detected_landmarks, params_global, pdm, triangulation, true, align_scale_au, align_width_au, align_height_au);

		// If the aligned face for AU matches the output requested one, just reuse it, else compute it
		if (align_scale_out == align_scale_au && align_width_out == align_width_au && align_height_out == align_height_au && align_mask)
		{
			aligned_face_for_output = aligned_face_for_au.clone();
		}
		else
		{
			if (align_mask)
			{
				AlignFaceMask(aligned_face_for_output, frame, detected_landmarks, params_global, pdm, triangulation, true, align_scale_out, align_width_out, align_height_out);
			}
			else
			{
				AlignFace(aligned_face_for_output, frame, detected_landmarks, params_global, pdm, true, align_scale_out, align_width_out, align_height_out);
			}
		}
	}
	else
	{
		aligned_face_for_output = cv::Mat(align_height_out, align_width_out, CV_8UC3);
		aligned_face_for_au = cv::Mat(align_height_au, align_width_au, CV_8UC3);
		aligned_face_for_output.setTo(0);
		aligned_face_for_au.setTo(0);
		params_local = cv::Mat_<float>(pdm.NumberOfModes(), 1, 0.0f);
	}

	if (aligned_face_for_output.channels() == 3 && out_grayscale)
	{
		cvtColor(aligned_face_for_output, aligned_face_for_output, cv::COLOR_BGR2GRAY);
	}

	// Extract HOG descriptor from the frame and convert it to a useable format
	cv::Mat_<double> hog_descriptor;
	Extract_FHOG_descriptor(hog_descriptor, aligned_face_for_au, this->num_hog_rows, this->num_hog_cols);
	
	// Store the descriptor
	hog_desc_frame = hog_descriptor;

	cv::Vec3d curr_orient(params_global[1], params_global[2], params_global[3]);
	int orientation_to_use = GetViewId(this->head_orientations, curr_orient);

	// Only update the running median if predictions are not high
	// That is don't update it when the face is expressive (just retrieve it)
	bool update_median = true;

	// TODO test if this would be useful or not
	//if(!this->AU_predictions_reg.empty())
	//{
	//	vector<pair<string, bool>> dyns = this->GetDynamicAUReg();

	//	for(size_t i = 0; i < this->AU_predictions_reg.size(); ++i)
	//	{
	//		bool stat = false;
	//		for (size_t n = 0; n < dyns.size(); ++n)
	//		{
	//			if (dyns[n].first.compare(AU_predictions_reg[i].first) == 0)
	//			{
	//				stat = !dyns[i].second;
	//			}
	//		}

	//		// If static predictor above 1.5 assume it's not a neutral face
	//		if(this->AU_predictions_reg[i].second > 1.5 && stat)
	//		{
	//			update_median = false;				
	//			break;
	//		}
	//	}
	//}

	update_median = update_median & success;

	if (success)
		frames_tracking_succ++;

	// A small speedup
	if(frames_tracking % 2 == 1)
	{
		UpdateRunningMedian(this->hog_desc_hist[orientation_to_use], this->hog_hist_sum[orientation_to_use], this->hog_desc_median, hog_descriptor, update_median, this->num_bins_hog, this->min_val_hog, this->max_val_hog);
		this->hog_desc_median.setTo(0, this->hog_desc_median < 0);
	}	

	// Geom descriptor and its median
	params_local = params_local.t();
	params_local.convertTo(geom_descriptor_frame, CV_64F);

	if(!success)
	{
		geom_descriptor_frame.setTo(0);
	}

	// Stack with the actual feature point locations (without mean)
	// TODO rem double
	cv::Mat_<double> princ_comp_d;
	pdm.princ_comp.convertTo(princ_comp_d, CV_64F);
	cv::Mat_<double> locs = princ_comp_d * geom_descriptor_frame.t();
	
	cv::hconcat(locs.t(), geom_descriptor_frame.clone(), geom_descriptor_frame);
	
	// A small speedup
	if(frames_tracking % 2 == 1)
	{
		UpdateRunningMedian(this->geom_desc_hist, this->geom_hist_sum, this->geom_descriptor_median, geom_descriptor_frame, update_median, this->num_bins_geom, this->min_val_geom, this->max_val_geom);
	}
	
	// Perform AU prediction	
	AU_predictions_reg = PredictCurrentAUs(orientation_to_use);

	// Add the reg predictions to the historic data
	for (size_t au = 0; au < AU_predictions_reg.size(); ++au)
	{

		// Find the appropriate AU (if not found add it)		
		// Only add if the detection was successful
		if(success)
		{
			AU_predictions_reg_all_hist[AU_predictions_reg[au].first].push_back(AU_predictions_reg[au].second);
		}
		else
		{
			AU_predictions_reg_all_hist[AU_predictions_reg[au].first].push_back(0);

			// Also invalidate AU if not successful
			AU_predictions_reg[au].second = 0;
		}
	}
	
	AU_predictions_class = PredictCurrentAUsClass(orientation_to_use);

	for (size_t au = 0; au < AU_predictions_class.size(); ++au)
	{

		// Find the appropriate AU (if not found add it)		
		// Only add if the detection was successful
		if(success)
		{
			AU_predictions_class_all_hist[AU_predictions_class[au].first].push_back(AU_predictions_class[au].second);
		}
		else
		{
			AU_predictions_class_all_hist[AU_predictions_class[au].first].push_back(0);

			// Also invalidate AU if not successful
			AU_predictions_class[au].second = 0;
		}
	}	

	// A workaround for online predictions to make them a bit more accurate
	std::vector<std::pair<std::string, double>> AU_predictions_reg_corrected;
	if (online)
	{
		AU_predictions_reg_corrected = CorrectOnlineAUs(AU_predictions_reg, orientation_to_use, true, false, success, true);
		AU_predictions_reg = AU_predictions_reg_corrected;
	}

	// Useful for prediction corrections (calibration after the whole video is processed)
	if (success && frames_tracking_succ - 1 < max_init_frames)
	{
		hog_desc_frames_init.push_back(hog_descriptor);
		geom_descriptor_frames_init.push_back(geom_descriptor_frame);
		views.push_back(orientation_to_use);
	}

	this->current_time_seconds = timestamp_seconds;

	view_used = orientation_to_use;
			
	valid_preds.push_back(success);
	timestamps.push_back(timestamp_seconds);

}

void FaceAnalyser::GetGeomDescriptor(cv::Mat_<double>& geom_desc)
{
	geom_desc = this->geom_descriptor_frame.clone();
}

// Perform prediction on initial n frames anew as the current neutral face estimate is better now
void FaceAnalyser::PostprocessPredictions()
{
	if(!postprocessed)
	{
		int success_ind = 0;
		int all_ind = 0;
		int all_frames_size = (int)timestamps.size();
		
		while(all_ind < all_frames_size && success_ind < max_init_frames)
		{
		
			if(valid_preds[all_ind])
			{

				this->hog_desc_frame = hog_desc_frames_init[success_ind];
				this->geom_descriptor_frame = geom_descriptor_frames_init[success_ind];

				// Perform AU prediction	
				auto AU_predictions_reg = PredictCurrentAUs(views[success_ind]);								

				// Modify the predictions to the historic data
				for (size_t au = 0; au < AU_predictions_reg.size(); ++au)
				{
					// Find the appropriate AU (if not found add it)		
					AU_predictions_reg_all_hist[AU_predictions_reg[au].first][all_ind] = AU_predictions_reg[au].second;

				}

				auto AU_predictions_class = PredictCurrentAUsClass(views[success_ind]);

				for (size_t au = 0; au < AU_predictions_class.size(); ++au)
				{
					// Find the appropriate AU (if not found add it)		
					AU_predictions_class_all_hist[AU_predictions_class[au].first][all_ind] = AU_predictions_class[au].second;
				}
		
				success_ind++;
			}
			all_ind++;

		}
		postprocessed = true;
	}
}

void FaceAnalyser::ExtractAllPredictionsOfflineReg(std::vector<std::pair<std::string, std::vector<double>>>& au_predictions, 
	std::vector<double>& confidences, std::vector<bool>& successes, std::vector<double>& timestamps, bool dynamic)
{
	if(dynamic)
	{
		PostprocessPredictions();
	}

	timestamps = this->timestamps;
	au_predictions.clear();
	// First extract the valid AU values and put them in a different format
	std::vector<std::vector<double>> aus_valid;
	std::vector<double> offsets;
	successes = this->valid_preds;
	
	std::vector<std::string> dyn_au_names = AU_SVR_dynamic_appearance_lin_regressors.GetAUNames();

	// Allow these AUs to be person calirated based on expected number of neutral frames (learned from the data)
	for(auto au_iter = AU_predictions_reg_all_hist.begin(); au_iter != AU_predictions_reg_all_hist.end(); ++au_iter)
	{
		std::vector<double> au_good;
		std::string au_name = au_iter->first;
		std::vector<double> au_vals = au_iter->second;
		
		au_predictions.push_back(std::pair<std::string, std::vector<double>>(au_name, au_vals));

		for(size_t frame = 0; frame < au_vals.size(); ++frame)
		{

			if(successes[frame])
			{
				au_good.push_back(au_vals[frame]);
			}

		}

		if(au_good.empty() || !dynamic)
		{
			offsets.push_back(0.0);
		}
		else
		{
			std::sort(au_good.begin(), au_good.end());
			// If it is a dynamic AU regressor we can also do some prediction shifting to make it more accurate
			// The shifting proportion is learned and is callen cutoff

			// Find the current id of the AU and the corresponding cutoff
			int au_id = -1;
			for (size_t a = 0; a < dyn_au_names.size(); ++a)
			{
				if (au_name.compare(dyn_au_names[a]) == 0)
				{
					au_id = (int)a;
				}
			}

			if (au_id != -1 && AU_SVR_dynamic_appearance_lin_regressors.GetCutoffs()[au_id] != -1)
			{
				double cutoff = AU_SVR_dynamic_appearance_lin_regressors.GetCutoffs()[au_id];
				offsets.push_back(au_good.at((int)((double)au_good.size() * cutoff)));
			}
			else
			{
				offsets.push_back(0);
			}
		}
		
		aus_valid.push_back(au_good);
	}
	
	// sort each of the aus and adjust the dynamic ones
	for(size_t au = 0; au < au_predictions.size(); ++au)
	{

		for(size_t frame = 0; frame < au_predictions[au].second.size(); ++frame)
		{

			if(successes[frame])
			{
				double scaling = 1;
				
				au_predictions[au].second[frame] = (au_predictions[au].second[frame] - offsets[au]) * scaling;
				
				if(au_predictions[au].second[frame] < 0.0)
					au_predictions[au].second[frame] = 0;

				if(au_predictions[au].second[frame] > 5)
					au_predictions[au].second[frame] = 5;
				
			}
			else
			{
				au_predictions[au].second[frame] = 0;
			}
		}
	}

	// Perform some prediction smoothing
	for (auto au_iter = au_predictions.begin(); au_iter != au_predictions.end(); ++au_iter)
	{
		std::string au_name = au_iter->first;

		// Perform a moving average of 3 frames
		int window_size = 3;
		std::vector<double> au_vals_tmp = au_iter->second;
		for (size_t i = (window_size - 1) / 2; i < au_iter->second.size() - (window_size - 1) / 2; ++i)
		{
			double sum = 0;
			for (int w = -(window_size - 1) / 2; w <= (window_size - 1) / 2; ++w)
			{
				sum += au_vals_tmp[i + w];
			}
			sum = sum / window_size;

			au_iter->second[i] = sum;
		}

	}

}

void FaceAnalyser::ExtractAllPredictionsOfflineClass(std::vector<std::pair<std::string, std::vector<double>>>& au_predictions, 
	std::vector<double>& confidences, std::vector<bool>& successes, std::vector<double>& timestamps, bool dynamic)
{
	if (dynamic)
	{
		PostprocessPredictions();
	}

	timestamps = this->timestamps;
	au_predictions.clear();

	for(auto au_iter = AU_predictions_class_all_hist.begin(); au_iter != AU_predictions_class_all_hist.end(); ++au_iter)
	{
		std::string au_name = au_iter->first;
		std::vector<double> au_vals = au_iter->second;
		
		// Perform a moving average of 7 frames on classifications
		int window_size = 7;
		std::vector<double> au_vals_tmp = au_vals;
		if((int)au_vals.size() > (window_size - 1) / 2)
		{
			for (size_t i = (window_size - 1)/2; i < au_vals.size() - (window_size - 1) / 2; ++i)
			{
				double sum = 0;
				int div_by = 0;
				for (int w = -(window_size - 1) / 2; w <= (window_size - 1) / 2 && (i+w < au_vals_tmp.size()); ++w)
				{
					sum += au_vals_tmp[i + w];
					div_by++;
				}
				sum = sum / div_by;
				if (sum < 0.5)
					sum = 0;
				else
					sum = 1;

				au_vals[i] = sum;
			}
		}
		au_predictions.push_back(std::pair<std::string, std::vector<double>>(au_name, au_vals));

	}

	successes = this->valid_preds;
}

// Reset the models
void FaceAnalyser::Reset()
{
	frames_tracking = 0;

	this->hog_desc_median.setTo(cv::Scalar(0));
	this->face_image_median.setTo(cv::Scalar(0));

	for( size_t i = 0; i < hog_desc_hist.size(); ++i)
	{
		this->hog_desc_hist[i] = cv::Mat_<int>(hog_desc_hist[i].rows, hog_desc_hist[i].cols, (int)0);
		this->hog_hist_sum[i] = 0;


		this->face_image_hist[i] = cv::Mat_<int>(face_image_hist[i].rows, face_image_hist[i].cols, (int)0);
		this->face_image_hist_sum[i] = 0;

		// 0 callibration predictions
		this->au_prediction_correction_count[i] = 0;
		this->au_prediction_correction_histogram[i] = cv::Mat_<int>(au_prediction_correction_histogram[i].rows, au_prediction_correction_histogram[i].cols, (int)0);
	}

	this->geom_descriptor_median.setTo(cv::Scalar(0));
	this->geom_desc_hist = cv::Mat_<int>(geom_desc_hist.rows, geom_desc_hist.cols, (int)0);
	geom_hist_sum = 0;

	// Reset the predictions
	AU_prediction_track = cv::Mat_<double>(AU_prediction_track.rows, AU_prediction_track.cols, 0.0);

	geom_desc_track = cv::Mat_<double>(geom_desc_track.rows, geom_desc_track.cols, 0.0);

	dyn_scaling = std::vector<std::vector<double>>(dyn_scaling.size(), std::vector<double>(dyn_scaling[0].size(), 5.0));

	AU_predictions_reg.clear();
	AU_predictions_class.clear();
	AU_predictions_combined.clear();
	timestamps.clear();
	AU_predictions_reg_all_hist.clear();
	AU_predictions_class_all_hist.clear();
	valid_preds.clear();

	// Clean up the postprocessing data as well
	hog_desc_frames_init.clear();
	geom_descriptor_frames_init.clear();
	postprocessed = false;
	frames_tracking_succ = 0;
}

void FaceAnalyser::UpdateRunningMedian(cv::Mat_<int>& histogram, int& hist_count, cv::Mat_<double>& median, const cv::Mat_<double>& descriptor, bool update, int num_bins, double min_val, double max_val)
{

	double length = max_val - min_val;
	if(length < 0)
		length = -length;

	// The median update
	if(histogram.empty())
	{
		histogram = cv::Mat_<int>(descriptor.cols, num_bins, (int)0);
		median = descriptor.clone();
	}

	if(update)
	{
		// Find the bins corresponding to the current descriptor
		cv::Mat_<double> converted_descriptor = (descriptor - min_val)*((double)num_bins)/(length);

		// Capping the top and bottom values
		converted_descriptor.setTo(cv::Scalar(num_bins-1), converted_descriptor > num_bins - 1);
		converted_descriptor.setTo(cv::Scalar(0), converted_descriptor < 0);

		for(int i = 0; i < histogram.rows; ++i)
		{
			int index = (int)converted_descriptor.at<double>(i);
			histogram.at<int>(i, index)++;
		}

		// Update the histogram count
		hist_count++;
	}

	if(hist_count == 1)
	{
		median = descriptor.clone();
	}
	else
	{
		// Recompute the median
		int cutoff_point = (hist_count + 1)/2;

		// For each dimension
		for(int i = 0; i < histogram.rows; ++i)
		{
			int cummulative_sum = 0;
			for(int j = 0; j < histogram.cols; ++j)
			{
				cummulative_sum += histogram.at<int>(i, j);
				if(cummulative_sum >= cutoff_point)
				{
					median.at<double>(i) = min_val + ((double)j) * (length/((double)num_bins)) + (0.5*(length)/ ((double)num_bins));
					break;
				}
			}
		}
	}
}


void FaceAnalyser::ExtractMedian(cv::Mat_<int>& histogram, int hist_count, cv::Mat_<double>& median, int num_bins, double min_val, double max_val)
{

	double length = max_val - min_val;
	if(length < 0)
		length = -length;

	// The median update
	if(histogram.empty())
	{
		return;
	}
	else
	{
		if(median.empty())
		{
			median = cv::Mat_<double>(1, histogram.rows, 0.0);
		}

		// Compute the median
		int cutoff_point = (hist_count + 1)/2;

		// For each dimension
		for(int i = 0; i < histogram.rows; ++i)
		{
			int cummulative_sum = 0;
			for(int j = 0; j < histogram.cols; ++j)
			{
				cummulative_sum += histogram.at<int>(i, j);
				if(cummulative_sum > cutoff_point)
				{
					median.at<double>(i) = min_val + j * (max_val/num_bins) + (0.5*(length)/num_bins);
					break;
				}
			}
		}
	}
}
// Apply the current predictors to the currently stored descriptors
std::vector<std::pair<std::string, double>> FaceAnalyser::PredictCurrentAUs(int view)
{

	std::vector<std::pair<std::string, double>> predictions;

	if(!hog_desc_frame.empty())
	{
		std::vector<std::string> svr_lin_stat_aus;
		std::vector<double> svr_lin_stat_preds;

		AU_SVR_static_appearance_lin_regressors.Predict(svr_lin_stat_preds, svr_lin_stat_aus, hog_desc_frame, geom_descriptor_frame);

		for(size_t i = 0; i < svr_lin_stat_preds.size(); ++i)
		{
			predictions.push_back(std::pair<std::string, double>(svr_lin_stat_aus[i], svr_lin_stat_preds[i]));
		}

		std::vector<std::string> svr_lin_dyn_aus;
		std::vector<double> svr_lin_dyn_preds;

		AU_SVR_dynamic_appearance_lin_regressors.Predict(svr_lin_dyn_preds, svr_lin_dyn_aus, hog_desc_frame, geom_descriptor_frame,  this->hog_desc_median, this->geom_descriptor_median);

		for(size_t i = 0; i < svr_lin_dyn_preds.size(); ++i)
		{
			predictions.push_back(std::pair<std::string, double>(svr_lin_dyn_aus[i], svr_lin_dyn_preds[i]));
		}

	}

	return predictions;
}

std::vector<std::pair<std::string, double>> FaceAnalyser::CorrectOnlineAUs(std::vector<std::pair<std::string, double>> predictions_orig, 
	int view, bool dyn_shift, bool dyn_scale, bool update_track, bool clip_values)
{
	// Correction that drags the predicion to 0 (assuming the bottom 10% of predictions are of neutral expresssions)
	std::vector<double> correction(predictions_orig.size(), 0.0);

	std::vector<std::pair<std::string, double>> predictions = predictions_orig;

	if(update_track)
	{
		UpdatePredictionTrack(au_prediction_correction_histogram[view], au_prediction_correction_count[view], correction, predictions, 0.10, 200, -3, 5, 10);
	}

	if(dyn_shift)
	{
		for(size_t i = 0; i < correction.size(); ++i)
		{
			predictions[i].second = predictions[i].second - correction[i];
		}
	}
	if(dyn_scale)
	{
		// Some scaling for effect better visualisation
		// Also makes sense as till the maximum expression is seen, it is hard to tell how expressive a persons face is
		if(dyn_scaling[view].empty())
		{
			dyn_scaling[view] = std::vector<double>(predictions.size(), 5.0);
		}
		
		for(size_t i = 0; i < predictions.size(); ++i)
		{
			// First establish presence (assume it is maximum as we have not seen max) 
			if(predictions[i].second > 1)
			{
				double scaling_curr = 5.0 / predictions[i].second;
				
				if(scaling_curr < dyn_scaling[view][i])
				{
					dyn_scaling[view][i] = scaling_curr;
				}
				predictions[i].second = predictions[i].second * dyn_scaling[view][i];
			}

			if(predictions[i].second > 5)
			{
				predictions[i].second = 5;
			}
		}
	}

	if(clip_values)
	{
		for(size_t i = 0; i < correction.size(); ++i)
		{
			if(predictions[i].second < 0)
				predictions[i].second = 0;
			if(predictions[i].second > 5)
				predictions[i].second = 5;
		}
	}
	return predictions;
}

// Apply the current predictors to the currently stored descriptors (classification)
std::vector<std::pair<std::string, double>> FaceAnalyser::PredictCurrentAUsClass(int view)
{

	std::vector<std::pair<std::string, double>> predictions;

	if(!hog_desc_frame.empty())
	{
		std::vector<std::string> svm_lin_stat_aus;
		std::vector<double> svm_lin_stat_preds;
		
		AU_SVM_static_appearance_lin.Predict(svm_lin_stat_preds, svm_lin_stat_aus, hog_desc_frame, geom_descriptor_frame);

		for(size_t i = 0; i < svm_lin_stat_aus.size(); ++i)
		{
			predictions.push_back(std::pair<std::string, double>(svm_lin_stat_aus[i], svm_lin_stat_preds[i]));
		}

		std::vector<std::string> svm_lin_dyn_aus;
		std::vector<double> svm_lin_dyn_preds;

		AU_SVM_dynamic_appearance_lin.Predict(svm_lin_dyn_preds, svm_lin_dyn_aus, hog_desc_frame, geom_descriptor_frame, this->hog_desc_median, this->geom_descriptor_median);

		for(size_t i = 0; i < svm_lin_dyn_aus.size(); ++i)
		{
			predictions.push_back(std::pair<std::string, double>(svm_lin_dyn_aus[i], svm_lin_dyn_preds[i]));
		}
		
	}

	return predictions;
}

std::vector<std::pair<std::string, double>> FaceAnalyser::GetCurrentAUsClass() const
{
	return AU_predictions_class;
}

std::vector<std::pair<std::string, double>> FaceAnalyser::GetCurrentAUsReg() const
{
	return AU_predictions_reg;
}

std::vector<std::pair<std::string, double>> FaceAnalyser::GetCurrentAUsCombined() const
{
	return AU_predictions_combined;
}

void FaceAnalyser::Read(std::string model_loc)
{
	// Reading in the modules for AU recognition

	//std::cout << "Reading the AU analysis module from: " << model_loc << std::endl;

	std::ifstream locations(model_loc.c_str(), std::ios_base::in);
	if (!locations.is_open())
	{
		std::cout << "Couldn't open the model file, aborting" << std::endl;
		return;
	}
	std::string line;

	// The other module locations should be defined as relative paths from the main model
	fs::path root = fs::path(model_loc).parent_path();

	// The main file contains the references to other files
	while (!locations.eof())
	{
		getline(locations, line);

		std::stringstream lineStream(line);

		std::string module;
		std::string location;

		// figure out which module is to be read from which file
		lineStream >> module;

		lineStream >> location;

		// remove carriage return at the end for compatibility with unix systems
		if (location.size() > 0 && location.at(location.size() - 1) == '\r')
		{
			location = location.substr(0, location.size() - 1);
		}

		// append to root
		location = (root / location).string();
		if (module.compare("AUPredictor") == 0)
		{
			// The AU predictors
			//std::cout << "Reading the AU predictors from: " << location;
			ReadAU(location);
			//std::cout << "... Done" << std::endl;
		}
		else if (module.compare("PDM") == 0)
		{
			//std::cout << "Reading the PDM from: " << location;
			pdm = LandmarkDetector::PDM();
			pdm.Read(location);
			//std::cout << "... Done" << std::endl;
		}
		else if (module.compare("Triangulation") == 0)
		{
			//std::cout << "Reading the triangulation from:" << location;
			// The triangulation used for masking out the non-face parts of aligned image
			std::ifstream triangulation_file(location);
			ReadMat(triangulation_file, triangulation);
			//std::cout << "... Done" << std::endl;
		}
	}

}

// Split the string into tokens
static void split(const std::string& str, std::vector<std::string>& out, char delim = ' ')
{
	std::stringstream ss(str);
	std::string token;
	while (std::getline(ss, token, delim)) {
		out.push_back(token);
	}
}

// Trim the end of a string (in place)
static void rtrim(std::string &s) {
	s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
		return !std::isspace(ch);
	}).base(), s.end());
}

// Reading in AU prediction modules
void FaceAnalyser::ReadAU(std::string au_model_location)
{

	// Open the list of the regressors in the file
	std::ifstream locations(au_model_location.c_str(), std::ios::in);

	if(!locations.is_open())
	{
		std::cout << "Couldn't open the AU prediction files at: " << au_model_location.c_str() << " aborting" << std::endl;
		std::cout.flush();
		return;
	}

	std::string line;
	
	// The other module locations should be defined as relative paths from the main model
	fs::path root = fs::path(au_model_location).parent_path();		
	
	// The main file contains the references to other files
	while (!locations.eof())
	{ 
		
		getline(locations, line);

		std::stringstream lineStream(line);

		std::string name;
		std::string location;

		// figure out which module is to be read from which file
		lineStream >> location;

		// Parse comma separated names that this regressor produces
		name = lineStream.str();
		int index = (int)name.find_first_of(' ');

		if(index >= 0)
		{
			name = name.substr(index+1);
			// remove carriage return at the end for compatibility with unix systems
			name.erase(name.find_last_not_of(" \n\r\t") + 1);
		}
		std::vector<std::string> au_names;
		split(name, au_names, ',');

		// append the lovstion to root location (boost syntax)
		location = (root / location).string();
				
		ReadRegressor(location, au_names);
	}
  
}

void FaceAnalyser::UpdatePredictionTrack(cv::Mat_<int>& prediction_corr_histogram, int& prediction_correction_count, 
	std::vector<double>& correction, const std::vector<std::pair<std::string, double>>& predictions, double ratio, int num_bins, 
	double min_val, double max_val, int min_frames)
{
	double length = max_val - min_val;
	if(length < 0)
		length = -length;

	correction.resize(predictions.size(), 0);

	// The median update
	if(prediction_corr_histogram.empty())
	{
		prediction_corr_histogram = cv::Mat_<int>((int)predictions.size(), num_bins, (int)0);
	}
	
	for(int i = 0; i < prediction_corr_histogram.rows; ++i)
	{
		// Find the bins corresponding to the current descriptor
		int index = (int)((predictions[i].second - min_val)*((double)num_bins)/(length));
		if(index < 0)
		{
			index = 0;
		}
		else if(index > num_bins - 1)
		{
			index = num_bins - 1;
		}
		prediction_corr_histogram.at<int>(i, index)++;
	}

	// Update the histogram count
	prediction_correction_count++;

	if(prediction_correction_count >= min_frames)
	{
		// Recompute the correction
		int cutoff_point = (int)(ratio * prediction_correction_count);

		// For each dimension
		for(int i = 0; i < prediction_corr_histogram.rows; ++i)
		{
			int cummulative_sum = 0;
			for(int j = 0; j < prediction_corr_histogram.cols; ++j)
			{
				cummulative_sum += prediction_corr_histogram.at<int>(i, j);
				if(cummulative_sum > cutoff_point)
				{
					double corr = min_val + j * (length/num_bins);
					correction[i] = corr;
					break;
				}
			}
		}
	}
}

void FaceAnalyser::GetSampleHist(cv::Mat_<int>& prediction_corr_histogram, int prediction_correction_count, std::vector<double>& sample,
	double ratio, int num_bins, double min_val, double max_val)
{

	double length = max_val - min_val;
	if(length < 0)
		length = -length;

	sample.resize(prediction_corr_histogram.rows, 0);

	// Recompute the correction
	int cutoff_point = (int)(ratio * prediction_correction_count);

	// For each dimension
	for(int i = 0; i < prediction_corr_histogram.rows; ++i)
	{
		int cummulative_sum = 0;
		for(int j = 0; j < prediction_corr_histogram.cols; ++j)
		{
			cummulative_sum += prediction_corr_histogram.at<int>(i, j);
			if(cummulative_sum > cutoff_point)
			{
				double corr = min_val + j * (length/num_bins);
				sample[i] = corr;
				break;
			}
		}
	}

}

void FaceAnalyser::ReadRegressor(std::string fname, const std::vector<std::string>& au_names)
{
	std::ifstream regressor_stream(fname.c_str(), std::ios::in | std::ios::binary);

	if (regressor_stream.is_open())
	{
		// First read the input type
		int regressor_type;
		regressor_stream.read((char*)&regressor_type, 4);

		if (regressor_type == SVR_appearance_static_linear)
		{
			AU_SVR_static_appearance_lin_regressors.Read(regressor_stream, au_names);
		}
		else if (regressor_type == SVR_appearance_dynamic_linear)
		{
			AU_SVR_dynamic_appearance_lin_regressors.Read(regressor_stream, au_names);
		}
		else if (regressor_type == SVM_linear_stat)
		{
			AU_SVM_static_appearance_lin.Read(regressor_stream, au_names);
		}
		else if (regressor_type == SVM_linear_dyn)
		{
			AU_SVM_dynamic_appearance_lin.Read(regressor_stream, au_names);
		}
	}
}

double FaceAnalyser::GetCurrentTimeSeconds() {
	return current_time_seconds;
}

// Allows for post processing of the AU signal
void FaceAnalyser::PostprocessOutputFile(std::string output_file)
{

	std::vector<double> certainties;
	std::vector<bool> successes;
	std::vector<double> timestamps;
	std::vector<std::pair<std::string, std::vector<double>>> predictions_reg;
	std::vector<std::pair<std::string, std::vector<double>>> predictions_class;

	// Construct the new values to overwrite the output file with
	ExtractAllPredictionsOfflineReg(predictions_reg, certainties, successes, timestamps, dynamic);
	ExtractAllPredictionsOfflineClass(predictions_class, certainties, successes, timestamps, dynamic);

	int num_class = (int)predictions_class.size();
	int num_reg = (int)predictions_reg.size();

	// Extract the indices of writing out first
	std::vector<std::string> au_reg_names = GetAURegNames();
	std::sort(au_reg_names.begin(), au_reg_names.end());
	std::vector<int> inds_reg;

	// write out ar the correct index
	for (std::string au_name : au_reg_names)
	{
		for (int i = 0; i < num_reg; ++i)
		{
			if (au_name.compare(predictions_reg[i].first) == 0)
			{
				inds_reg.push_back(i);
				break;
			}
		}
	}

	std::vector<std::string> au_class_names = GetAUClassNames();
	std::sort(au_class_names.begin(), au_class_names.end());
	std::vector<int> inds_class;

	// write out ar the correct index
	for (std::string au_name : au_class_names)
	{
		for (int i = 0; i < num_class; ++i)
		{
			if (au_name.compare(predictions_class[i].first) == 0)
			{
				inds_class.push_back(i);
				break;
			}
		}
	}
	// Read all of the output file in
	std::vector<std::string> output_file_contents;

	std::ifstream infile(output_file);
	std::string line;

	while (std::getline(infile, line))
		output_file_contents.push_back(line);

	infile.close();

	// Read the header and find all _r and _c parts in a file and use their indices
	std::vector<std::string> tokens;
	split(output_file_contents[0], tokens, ',');

	int begin_ind = -1;

	for (size_t i = 0; i < tokens.size(); ++i)
	{
		if (tokens[i].find("AU") != std::string::npos && begin_ind == -1)
		{
			begin_ind = (int)i;
			break;
		}
	}
	int end_ind = begin_ind + num_class + num_reg;

	// Now overwrite the whole file
	std::ofstream outfile(output_file, std::ios_base::out);
	// Write the header
	outfile << std::setprecision(2);
	outfile << std::fixed;
	outfile << std::noshowpoint;

	outfile << output_file_contents[0].c_str() << std::endl;

	// Write the contents
	for (int i = 1; i < (int)output_file_contents.size(); ++i)
	{
		std::vector<std::string> tokens;
		split(output_file_contents[i], tokens, ',');

		rtrim(tokens[0]);
		outfile << tokens[0];

		for (int t = 1; t < (int)tokens.size(); ++t)
		{
			if (t >= begin_ind && t < end_ind)
			{
				if (t - begin_ind < num_reg)
				{
					outfile << ", " << predictions_reg[inds_reg[t - begin_ind]].second[i - 1];
				}
				else
				{
					outfile << ", " << predictions_class[inds_class[t - begin_ind - num_reg]].second[i - 1];
				}
			}
			else
			{
				rtrim(tokens[t]);
				outfile << ", " << tokens[t];
			}
		}
		outfile << std::endl;
	}


}
