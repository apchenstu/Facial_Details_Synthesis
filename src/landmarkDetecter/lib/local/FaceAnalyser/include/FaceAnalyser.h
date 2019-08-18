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

#ifndef FACEANALYSER_H
#define FACEANALYSER_H

// STL includes
#include <string>
#include <vector>
#include <map>

// OpenCV includes
#include <opencv2/core/core.hpp>

// Local includes
#include "SVR_dynamic_lin_regressors.h"
#include "SVR_static_lin_regressors.h"
#include "SVM_static_lin.h"
#include "SVM_dynamic_lin.h"
#include "PDM.h"
#include "FaceAnalyserParameters.h"

namespace FaceAnalysis
{

class FaceAnalyser{

public:


	enum RegressorType{ SVR_appearance_static_linear = 0, SVR_appearance_dynamic_linear = 1, SVR_dynamic_geom_linear = 2, SVR_combined_linear = 3, SVM_linear_stat = 4, SVM_linear_dyn = 5, SVR_linear_static_seg = 6, SVR_linear_dynamic_seg =7};

	// Constructor for FaceAnalyser using the parameters structure
	FaceAnalyser(const FaceAnalysis::FaceAnalyserParameters& face_analyser_params);

	void AddNextFrame(const cv::Mat& frame, const cv::Mat_<float>& detected_landmarks, bool success, double timestamp_seconds, bool online = false);

	double GetCurrentTimeSeconds();
	
	// Grab the current predictions about AUs from the face analyser
	std::vector<std::pair<std::string, double>> GetCurrentAUsClass() const; // AU presence
	std::vector<std::pair<std::string, double>> GetCurrentAUsReg() const;   // AU intensity
	std::vector<std::pair<std::string, double>> GetCurrentAUsCombined() const; // Both presense and intensity

	// A standalone call for predicting AUs and computing face texture features from a static image
	void PredictStaticAUsAndComputeFeatures(const cv::Mat& frame, const cv::Mat_<float>& detected_landmarks);

	void Reset();

	void GetLatestHOG(cv::Mat_<double>& hog_descriptor, int& num_rows, int& num_cols);
	void GetLatestAlignedFace(cv::Mat& image);
	
	void GetLatestNeutralHOG(cv::Mat_<double>& hog_descriptor, int& num_rows, int& num_cols);
	
	cv::Mat_<int> GetTriangulation();
	
	void GetGeomDescriptor(cv::Mat_<double>& geom_desc);

	// Grab the names of AUs being predicted
	std::vector<std::string> GetAUClassNames() const; // Presence
	std::vector<std::string> GetAURegNames() const; // Intensity

	// Identify if models are static or dynamic (useful for correction and shifting)
	std::vector<bool> GetDynamicAUClass() const; // Presence
	std::vector<std::pair<std::string, bool>> GetDynamicAUReg() const; // Intensity


	void ExtractAllPredictionsOfflineReg(std::vector<std::pair<std::string, std::vector<double>>>& au_predictions, 
		std::vector<double>& confidences, std::vector<bool>& successes, std::vector<double>& timestamps, bool dynamic);
	void ExtractAllPredictionsOfflineClass(std::vector<std::pair<std::string, std::vector<double>>>& au_predictions,
		std::vector<double>& confidences, std::vector<bool>& successes, std::vector<double>& timestamps, bool dynamic);

	// Helper function for post-processing AU output files
	void PostprocessOutputFile(std::string output_file);

private:

	// Point distribution model coddesponding to the current Face Analyser
	LandmarkDetector::PDM pdm;

	// Where the predictions are kept
	std::vector<std::pair<std::string, double>> AU_predictions_reg;
	std::vector<std::pair<std::string, double>> AU_predictions_class;

	std::vector<std::pair<std::string, double>> AU_predictions_combined;

	// Keeping track of AU predictions over time (useful for post-processing)
	std::vector<double> timestamps;
	std::map<std::string, std::vector<double>> AU_predictions_reg_all_hist;
	std::map<std::string, std::vector<double>> AU_predictions_class_all_hist;
	std::vector<bool> valid_preds;

	int frames_tracking;

	// Is the AU model dynamic
	bool dynamic;

	// Cache of intermediate images
	cv::Mat aligned_face_for_au;
	cv::Mat aligned_face_for_output;
	bool out_grayscale;

	// Private members to be used for predictions
	// The HOG descriptor of the last frame
	cv::Mat_<double> hog_desc_frame;
	int num_hog_rows;
	int num_hog_cols;

	// Keep a running median of the hog descriptors and a aligned images
	cv::Mat_<double> hog_desc_median;
	cv::Mat_<double> face_image_median;

	// Use histograms for quick (but approximate) median computation
	// Use the same for
	std::vector<cv::Mat_<int> > hog_desc_hist;

	// This is not being used at the moment as it is a bit slow
	std::vector<cv::Mat_<int> > face_image_hist;
	std::vector<int> face_image_hist_sum;

	std::vector<cv::Vec3d> head_orientations;

	int num_bins_hog;
	double min_val_hog;
	double max_val_hog;
	std::vector<int> hog_hist_sum;
	int view_used;

	// The geometry descriptor (rigid followed by non-rigid shape parameters from CLNF)
	cv::Mat_<double> geom_descriptor_frame;
	cv::Mat_<double> geom_descriptor_median;
	
	int geom_hist_sum;
	cv::Mat_<int> geom_desc_hist;
	int num_bins_geom;
	double min_val_geom;
	double max_val_geom;
	
	// Using the bounding box of previous analysed frame to determine if a reset is needed
	cv::Rect_<double> face_bounding_box;
	
	// The AU predictions internally
	std::vector<std::pair<std::string, double>> PredictCurrentAUs(int view);
	std::vector<std::pair<std::string, double>> PredictCurrentAUsClass(int view);

	// special step for online (rather than offline AU prediction)
	std::vector<std::pair<std::string, double>> CorrectOnlineAUs(std::vector<std::pair<std::string, double>> predictions_orig, int view, bool dyn_shift = false, bool dyn_scale = false, bool update_track = true, bool clip_values = false);

	void Read(std::string model_loc);

	void ReadAU(std::string au_location);

	void ReadRegressor(std::string fname, const std::vector<std::string>& au_names);

	// A utility function for keeping track of approximate running medians used for AU and emotion inference using a set of histograms (the histograms are evenly spaced from min_val to max_val)
	// Descriptor has to be a row vector
	// TODO this duplicates some other code
	void UpdateRunningMedian(cv::Mat_<int>& histogram, int& hist_sum, cv::Mat_<double>& median, const cv::Mat_<double>& descriptor, bool update, int num_bins, double min_val, double max_val);
	void ExtractMedian(cv::Mat_<int>& histogram, int hist_count, cv::Mat_<double>& median, int num_bins, double min_val, double max_val);
	
	// The linear SVR regressors
	SVR_static_lin_regressors AU_SVR_static_appearance_lin_regressors;
	SVR_dynamic_lin_regressors AU_SVR_dynamic_appearance_lin_regressors;
		
	// The linear SVM classifiers
	SVM_static_lin AU_SVM_static_appearance_lin;
	SVM_dynamic_lin AU_SVM_dynamic_appearance_lin;

	// The AUs predicted by the model are not always 0 calibrated to a person. That is they don't always predict 0 for a neutral expression
	// Keeping track of the predictions we can correct for this, by assuming that at least "ratio" of frames are neutral and subtract that value of prediction, only perform the correction after min_frames
	void UpdatePredictionTrack(cv::Mat_<int>& prediction_corr_histogram, int& prediction_correction_count, 
		std::vector<double>& correction, const std::vector<std::pair<std::string, double>>& predictions, double ratio=0.25, int num_bins = 200, double min_val = -3, double max_val = 5, int min_frames = 10);
	void GetSampleHist(cv::Mat_<int>& prediction_corr_histogram, int prediction_correction_count, 
		std::vector<double>& sample, double ratio, int num_bins = 200, double min_val = 0, double max_val = 5);

	void PostprocessPredictions();

	std::vector<cv::Mat_<int>> au_prediction_correction_histogram;
	std::vector<int> au_prediction_correction_count;

	// Some dynamic scaling (the logic is that before the extreme versions of expression or emotion are shown,
	// it is hard to tell the boundaries, this allows us to scale the model to the most extreme seen)
	// They have to be view specific
	std::vector<std::vector<double>> dyn_scaling;
	
	// Keeping track of predictions for summary stats
	cv::Mat_<double> AU_prediction_track;
	cv::Mat_<double> geom_desc_track;

	double current_time_seconds;

	// Used for face alignment
	cv::Mat_<int> triangulation;
	double align_scale_au;
	int align_width_au;
	int align_height_au;

	bool align_mask;
	double align_scale_out;
	int align_width_out;
	int align_height_out;

	// Useful placeholder for renormalizing the initial frames of shorter videos
	int max_init_frames = 3000;
	std::vector<cv::Mat_<double>> hog_desc_frames_init;
	std::vector<cv::Mat_<double>> geom_descriptor_frames_init;
	std::vector<int> views;
	bool postprocessed = false;
	int frames_tracking_succ = 0;

};
  //===========================================================================
}
#endif // FACEANALYSER_H
