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

#include <LandmarkDetectorModel.h>

// Local includes
#include <LandmarkDetectorUtils.h>
#include <RotationHelpers.h>

using namespace LandmarkDetector;

//=============================================================================
//=============================================================================

// Constructors
// A default constructor
CLNF::CLNF()
{
	FaceModelParameters parameters;

	// A successful read wil set this to true
	loaded_successfully = false;

	this->Read(parameters.model_location);
}

// Constructor from a model file
CLNF::CLNF(std::string fname)
{
	// A successful read wil set this to true
	loaded_successfully = false;

	this->Read(fname);
}

// Copy constructor (makes a deep copy of CLNF)
CLNF::CLNF(const CLNF& other): pdm(other.pdm), params_local(other.params_local.clone()), params_global(other.params_global), detected_landmarks(other.detected_landmarks.clone()),
	landmark_likelihoods(other.landmark_likelihoods.clone()), patch_experts(other.patch_experts), landmark_validator(other.landmark_validator), haar_face_detector_location(other.haar_face_detector_location),
	mtcnn_face_detector_location(other.mtcnn_face_detector_location), hierarchical_mapping(other.hierarchical_mapping), hierarchical_models(other.hierarchical_models), hierarchical_model_names(other.hierarchical_model_names),
	hierarchical_params(other.hierarchical_params), eye_model(other.eye_model), face_detector_MTCNN(other.face_detector_MTCNN), preference_det(other.preference_det), loaded_successfully(other.loaded_successfully)
{
	this->detection_success = other.detection_success;
	this->tracking_initialised = other.tracking_initialised;
	this->detection_certainty = other.detection_certainty;
	this->model_likelihood = other.model_likelihood;
	this->failures_in_a_row = other.failures_in_a_row;

	// Load the CascadeClassifier (as it does not have a proper copy constructor)
	if(!haar_face_detector_location.empty())
	{
		this->face_detector_HAAR.load(haar_face_detector_location);
	}
	// Make sure the matrices are allocated properly
	this->triangulations.resize(other.triangulations.size());
	for(size_t i = 0; i < other.triangulations.size(); ++i)
	{
		// Make sure the matrix is copied.
		this->triangulations[i] = other.triangulations[i].clone();
	}

	// Make sure the matrices are allocated properly
	for(std::map<int, cv::Mat_<float>>::const_iterator it = other.kde_resp_precalc.begin(); it!= other.kde_resp_precalc.end(); it++)
	{
		// Make sure the matrix is copied.
		this->kde_resp_precalc.insert(std::pair<int, cv::Mat_<float>>(it->first, it->second.clone()));
	}

}

// Assignment operator for lvalues (makes a deep copy of CLNF)
CLNF & CLNF::operator= (const CLNF& other)
{
	if (this != &other) // protect against invalid self-assignment
	{
		pdm = PDM(other.pdm);
		params_local = other.params_local.clone();
		params_global = other.params_global;
		detected_landmarks = other.detected_landmarks.clone();
		
		landmark_likelihoods =other.landmark_likelihoods.clone();
		patch_experts = Patch_experts(other.patch_experts);
		landmark_validator = DetectionValidator(other.landmark_validator);
		haar_face_detector_location = other.haar_face_detector_location;
		mtcnn_face_detector_location = other.mtcnn_face_detector_location;

		this->detection_success = other.detection_success;
		this->tracking_initialised = other.tracking_initialised;
		this->detection_certainty = other.detection_certainty;
		this->model_likelihood = other.model_likelihood;
		this->failures_in_a_row = other.failures_in_a_row;

		this->eye_model = other.eye_model;
		
		this->preference_det = other.preference_det;

		// Load the CascadeClassifier (as it does not have a proper copy constructor)
		if(!haar_face_detector_location.empty())
		{
			this->face_detector_HAAR.load(haar_face_detector_location);
		}
		// Make sure the matrices are allocated properly
		this->triangulations.resize(other.triangulations.size());
		for(size_t i = 0; i < other.triangulations.size(); ++i)
		{
			// Make sure the matrix is copied.
			this->triangulations[i] = other.triangulations[i].clone();
		}

		// Make sure the matrices are allocated properly
		for(std::map<int, cv::Mat_<float>>::const_iterator it = other.kde_resp_precalc.begin(); it!= other.kde_resp_precalc.end(); it++)
		{
			// Make sure the matrix is copied.
			this->kde_resp_precalc.insert(std::pair<int, cv::Mat_<float>>(it->first, it->second.clone()));
		}

		// Copy over the hierarchical models
		this->hierarchical_mapping = other.hierarchical_mapping;
		this->hierarchical_models = other.hierarchical_models;
		this->hierarchical_model_names = other.hierarchical_model_names;
		this->hierarchical_params = other.hierarchical_params;

		mtcnn_face_detector_location = other.mtcnn_face_detector_location;
		face_detector_MTCNN = other.face_detector_MTCNN;

		loaded_successfully = other.loaded_successfully;
	}

	return *this;
}

// Move constructor
CLNF::CLNF(const CLNF&& other)
{
	this->detection_success = other.detection_success;
	this->tracking_initialised = other.tracking_initialised;
	this->detection_certainty = other.detection_certainty;
	this->model_likelihood = other.model_likelihood;
	this->failures_in_a_row = other.failures_in_a_row;

	pdm = other.pdm;
	params_local = other.params_local;
	params_global = other.params_global;
	detected_landmarks = other.detected_landmarks;
	landmark_likelihoods = other.landmark_likelihoods;
	patch_experts = other.patch_experts;
	landmark_validator = other.landmark_validator;
	haar_face_detector_location = other.haar_face_detector_location;
	mtcnn_face_detector_location = other.mtcnn_face_detector_location;

	face_detector_HAAR = other.face_detector_HAAR;

	triangulations = other.triangulations;
	kde_resp_precalc = other.kde_resp_precalc;

	face_detector_MTCNN = other.face_detector_MTCNN;

	// Copy over the hierarchical models
	this->hierarchical_mapping = other.hierarchical_mapping;
	this->hierarchical_models = other.hierarchical_models;
	this->hierarchical_model_names = other.hierarchical_model_names;
	this->hierarchical_params = other.hierarchical_params;

	this->eye_model = other.eye_model;

	this->preference_det = other.preference_det;

	this->loaded_successfully = other.loaded_successfully;

}

// Assignment operator for rvalues
CLNF & CLNF::operator= (const CLNF&& other)
{
	this->detection_success = other.detection_success;
	this->tracking_initialised = other.tracking_initialised;
	this->detection_certainty = other.detection_certainty;
	this->model_likelihood = other.model_likelihood;
	this->failures_in_a_row = other.failures_in_a_row;

	pdm = other.pdm;
	params_local = other.params_local;
	params_global = other.params_global;
	detected_landmarks = other.detected_landmarks;
	landmark_likelihoods = other.landmark_likelihoods;
	patch_experts = other.patch_experts;
	landmark_validator = other.landmark_validator;
	haar_face_detector_location = other.haar_face_detector_location;
	mtcnn_face_detector_location = other.mtcnn_face_detector_location;

	face_detector_HAAR = other.face_detector_HAAR;

	triangulations = other.triangulations;
	kde_resp_precalc = other.kde_resp_precalc;

	face_detector_MTCNN = other.face_detector_MTCNN;

	// Copy over the hierarchical models
	this->hierarchical_mapping = other.hierarchical_mapping;
	this->hierarchical_models = other.hierarchical_models;
	this->hierarchical_model_names = other.hierarchical_model_names;
	this->hierarchical_params = other.hierarchical_params;

	this->eye_model = other.eye_model;

	this->preference_det = other.preference_det;

	this->loaded_successfully = other.loaded_successfully;

	return *this;
}


bool CLNF::Read_CLNF(std::string clnf_location)
{
	// Location of modules
	std::ifstream locations(clnf_location.c_str(), std::ios_base::in);

	if(!locations.is_open())
	{
		std::cout << "Couldn't open the CLNF model file aborting" << std::endl;
		std::cout.flush();
		return false;
	}

	std::string line;
	
	std::vector<std::string> intensity_expert_locations;
	std::vector<std::string> ccnf_expert_locations;
	std::vector<std::string> cen_expert_locations;
	std::string early_term_loc;

	// The other module locations should be defined as relative paths from the main model
	fs::path root = fs::path(clnf_location).parent_path();

	// The main file contains the references to other files
	while (!locations.eof())
	{ 
		
		getline(locations, line);

		std::stringstream lineStream(line);

		std::string module;
		std::string location;

		// figure out which module is to be read from which file
		lineStream >> module;
		
		getline(lineStream, location);

		if(location.size() > 0)
			location.erase(location.begin()); // remove the first space
				
		// remove carriage return at the end for compatibility with unix systems
		if(location.size() > 0 && location.at(location.size()-1) == '\r')
		{
			location = location.substr(0, location.size()-1);
		}

		// append the lovstion to root location (boost syntax)
		location = (root / location).string();
				
		if (module.compare("PDM") == 0) 
		{            
			//std::cout << "Reading the PDM module from: " << location << "....";
			bool read_success = pdm.Read(location);

			if (!read_success)
			{
				return false;
			}

			//std::cout << "Done" << std::endl;
		}
		else if (module.compare("Triangulations") == 0) 
		{       
			//std::cout << "Reading the Triangulations module from: " << location << "....";
			std::ifstream triangulationFile(location.c_str(), std::ios_base::in);

			if(!triangulationFile.is_open())
			{
				return false;
			}

			LandmarkDetector::SkipComments(triangulationFile);

			int numViews;
			triangulationFile >> numViews;

			// read in the triangulations
			triangulations.resize(numViews);

			for(int i = 0; i < numViews; ++i)
			{
				LandmarkDetector::SkipComments(triangulationFile);
				LandmarkDetector::ReadMat(triangulationFile, triangulations[i]);
			}
			//std::cout << "Done" << std::endl;
		}
		else if(module.compare("PatchesIntensity") == 0)
		{
			intensity_expert_locations.push_back(location);
		}
		else if(module.compare("PatchesCCNF") == 0)
		{
			ccnf_expert_locations.push_back(location);
		}
		else if (module.compare("PatchesCEN") == 0)
		{
			cen_expert_locations.push_back(location);
		}
		else if (module.compare("EarlyTermination") == 0)
		{
			early_term_loc = location;
		}
	} 
  
	// Initialise the patch experts
	bool read_success = patch_experts.Read(intensity_expert_locations, ccnf_expert_locations, cen_expert_locations, early_term_loc);

	if(!read_success)
	{
		return false;
	}

	return true;
	
}

void CLNF::Read(std::string main_location)
{

	//std::cout << "Reading the landmark detector/tracker from: " << main_location << std::endl;
	
	std::ifstream locations(main_location.c_str(), std::ios_base::in);
	if(!locations.is_open())
	{
		std::cout << "Couldn't open the model file, aborting" << std::endl;
		loaded_successfully = false;
		return;
	}
	std::string line;
	
	// The other module locations should be defined as relative paths from the main model
	fs::path root = fs::path(main_location).parent_path();	

	// Assume no eye model, unless read-in
	eye_model = false;

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
		if(location.size() > 0 && location.at(location.size()-1) == '\r')
		{
			location = location.substr(0, location.size()-1);
		}


		// append to root
		location = (root / location).string();
		if (module.compare("LandmarkDetector") == 0) 
		{ 
			//std::cout << "Reading the landmark detector module from: " << location << std::endl;

			// The CLNF module includes the PDM and the patch experts
			bool read_success = Read_CLNF(location);

			if(!read_success)
			{
				loaded_successfully = false;
				return;
			}
		}
		else if(module.compare("LandmarkDetector_part") == 0)
		{
			std::string part_name;
			lineStream >> part_name;
			//std::cout << "Reading part based module...." << part_name << std::endl;

			std::vector<std::pair<int, int>> mappings;
			while(!lineStream.eof())
			{
				int ind_in_main;
				lineStream >> ind_in_main;
				
				int ind_in_part;
				lineStream >> ind_in_part;
				mappings.push_back(std::pair<int, int>(ind_in_main, ind_in_part));
			}
		
			this->hierarchical_mapping.push_back(mappings);

			CLNF part_model(location);

			if (!part_model.loaded_successfully)
			{
				loaded_successfully = false;
				return;
			}

			this->hierarchical_models.push_back(part_model);

			this->hierarchical_model_names.push_back(part_name);

			// Making sure we look based on model directory
			std::string root_loc = fs::path(main_location).parent_path().string();
			std::vector<std::string> sub_arguments{ root_loc };
			
			FaceModelParameters params(sub_arguments);
			
			params.validate_detections = false;
			params.refine_hierarchical = false;
			params.refine_parameters = false;

			if(part_name.compare("left_eye") == 0 || part_name.compare("right_eye") == 0)
			{
				
				std::vector<int> windows_large;
				windows_large.push_back(5);
				windows_large.push_back(3);

				std::vector<int> windows_small;
				windows_small.push_back(5);
				windows_small.push_back(3);

				params.window_sizes_init = windows_large;
				params.window_sizes_small = windows_small;
				params.window_sizes_current = windows_large;

				params.reg_factor = 0.1;
				params.sigma = 2;
			}
			else if(part_name.compare("left_eye_28") == 0 || part_name.compare("right_eye_28") == 0)
			{
				std::vector<int> windows_large;
				windows_large.push_back(3);
				windows_large.push_back(5);
				windows_large.push_back(9);

				std::vector<int> windows_small;
				windows_small.push_back(3);
				windows_small.push_back(5);
				windows_small.push_back(9);

				params.window_sizes_init = windows_large;
				params.window_sizes_small = windows_small;
				params.window_sizes_current = windows_large;

				params.reg_factor = 0.5;
				params.sigma = 1.0;

				eye_model = true;

			}
			else if(part_name.compare("mouth") == 0)
			{
				std::vector<int> windows_large;
				windows_large.push_back(7);
				windows_large.push_back(7);

				std::vector<int> windows_small;
				windows_small.push_back(7);
				windows_small.push_back(7);

				params.window_sizes_init = windows_large;
				params.window_sizes_small = windows_small;
				params.window_sizes_current = windows_large;

				params.reg_factor = 1.0;
				params.sigma = 2.0;
			}
			else if(part_name.compare("brow") == 0)
			{
				std::vector<int> windows_large;
				windows_large.push_back(11);
				windows_large.push_back(9);

				std::vector<int> windows_small;
				windows_small.push_back(11);
				windows_small.push_back(9);

				params.window_sizes_init = windows_large;
				params.window_sizes_small = windows_small;
				params.window_sizes_current = windows_large;

				params.reg_factor = 10.0;
				params.sigma = 3.5;
			}
			else if(part_name.compare("inner") == 0)
			{
				std::vector<int> windows_large;
				windows_large.push_back(9);

				std::vector<int> windows_small;
				windows_small.push_back(9);

				params.window_sizes_init = windows_large;
				params.window_sizes_small = windows_small;
				params.window_sizes_current = windows_large;

				params.reg_factor = 2.5;
				params.sigma = 1.75;
				params.weight_factor = 2.5;
			}

			this->hierarchical_params.push_back(params);

			//std::cout << "Done" << std::endl;
		}
		else if (module.compare("DetectionValidator") == 0)
		{            
			//std::cout << "Reading the landmark validation module....";
			landmark_validator.Read(location);
			//std::cout << "Done" << std::endl;
		}
	}
 
	detected_landmarks.create(2 * pdm.NumberOfPoints(), 1);
	detected_landmarks.setTo(0);

	detection_success = false;
	tracking_initialised = false;
	model_likelihood = -10; // very low
	detection_certainty = 0; // very uncertain

	// Initialising default values for the rest of the variables

	// local parameters (shape)
	params_local.create(pdm.NumberOfModes(), 1);
	params_local.setTo(0.0);

	// global parameters (pose) [scale, euler_x, euler_y, euler_z, tx, ty]
	params_global = cv::Vec6f(1, 0, 0, 0, 0, 0);

	failures_in_a_row = -1;

	preference_det.x = -1;
	preference_det.y = -1;

	loaded_successfully = true;

}

// Resetting the model (for a new video, or complet reinitialisation
void CLNF::Reset()
{
	detected_landmarks.setTo(0);

	detection_success = false;
	tracking_initialised = false;
	model_likelihood = -10;  // very low
	detection_certainty = 0; // very uncertain

	// local parameters (shape)
	params_local.setTo(0.0);

	// global parameters (pose) [scale, euler_x, euler_y, euler_z, tx, ty]
	params_global = cv::Vec6f(1, 0, 0, 0, 0, 0);

	failures_in_a_row = -1;
	face_template = cv::Mat_<uchar>();
}

// Resetting the model, choosing the face nearest (x,y)
void CLNF::Reset(double x, double y)
{

	// First reset the model overall
	this->Reset();

	// Now in the following frame when face detection takes place this is the point at which it will be preffered
	this->preference_det.x = x;
	this->preference_det.y = y;

}

// The main internal landmark detection call (should not be used externally?)
bool CLNF::DetectLandmarks(const cv::Mat_<uchar> &image, FaceModelParameters& params)
{

	// TODO this could be moved out
	cv::Mat_<float> gray_image_flt;
	image.convertTo(gray_image_flt, CV_32F);

	// Fits from the current estimate of local and global parameters in the model
	bool fit_success = Fit(gray_image_flt, params.window_sizes_current, params);

	// Store the landmarks converged on in detected_landmarks
	pdm.CalcShape2D(detected_landmarks, params_local, params_global);	

	if(params.refine_hierarchical && hierarchical_models.size() > 0)
	{
		bool parts_used = false;		

		// Do the hierarchical models in parallel
		parallel_for_(cv::Range(0, hierarchical_models.size()), [&](const cv::Range& range) {
			for (int part_model = range.start; part_model < range.end; part_model++)
			{
				
				int n_part_points = hierarchical_models[part_model].pdm.NumberOfPoints();

				std::vector<std::pair<int, int>> mappings = this->hierarchical_mapping[part_model];

				cv::Mat_<float> part_model_locs(n_part_points * 2, 1, 0.0f);

				// Extract the corresponding landmarks
				for (size_t mapping_ind = 0; mapping_ind < mappings.size(); ++mapping_ind)
				{
					part_model_locs.at<float>(mappings[mapping_ind].second) = detected_landmarks.at<float>(mappings[mapping_ind].first);
					part_model_locs.at<float>(mappings[mapping_ind].second + n_part_points) = detected_landmarks.at<float>(mappings[mapping_ind].first + this->pdm.NumberOfPoints());
				}

				// Fit the part based model PDM
				hierarchical_models[part_model].pdm.CalcParams(hierarchical_models[part_model].params_global, hierarchical_models[part_model].params_local, part_model_locs);

				// Only do this if we don't need to upsample
				if (params_global[0] > 0.9 * hierarchical_models[part_model].patch_experts.patch_scaling[0])
				{
					parts_used = true;

					this->hierarchical_params[part_model].window_sizes_current = this->hierarchical_params[part_model].window_sizes_init;

					// Do the actual landmark detection
					hierarchical_models[part_model].DetectLandmarks(image, hierarchical_params[part_model]);

				}
				else
				{
					hierarchical_models[part_model].pdm.CalcShape2D(hierarchical_models[part_model].detected_landmarks, hierarchical_models[part_model].params_local, hierarchical_models[part_model].params_global);
				}
		
			}
		});

		// Recompute main model based on the fit part models
		if(parts_used)
		{

			for (size_t part_model = 0; part_model < hierarchical_models.size(); ++part_model)
			{
				std::vector<std::pair<int, int>> mappings = this->hierarchical_mapping[part_model];

				// Reincorporate the models into main tracker
				for (size_t mapping_ind = 0; mapping_ind < mappings.size(); ++mapping_ind)
				{
					detected_landmarks.at<float>(mappings[mapping_ind].first) = hierarchical_models[part_model].detected_landmarks.at<float>(mappings[mapping_ind].second);
					detected_landmarks.at<float>(mappings[mapping_ind].first + pdm.NumberOfPoints()) = hierarchical_models[part_model].detected_landmarks.at<float>(mappings[mapping_ind].second + hierarchical_models[part_model].pdm.NumberOfPoints());
				}
			}

			pdm.CalcParams(params_global, params_local, detected_landmarks);		
			pdm.CalcShape2D(detected_landmarks, params_local, params_global);
		}

	}

	// Check detection correctness
	if(params.validate_detections && fit_success)
	{

		cv::Vec3d orientation(params_global[1], params_global[2], params_global[3]);

		detection_certainty = landmark_validator.Check(orientation, image, detected_landmarks);

		detection_success = detection_certainty > params.validation_boundary;

	}
	else
	{
		detection_success = fit_success;
		if(fit_success)
		{
			detection_certainty = 1;
		}
		else
		{
			detection_certainty = 0;
		}

	}

	return detection_success;
}

//=============================================================================
bool CLNF::Fit(const cv::Mat_<float>& im, const std::vector<int>& window_sizes, const FaceModelParameters& parameters)
{
	// Making sure it is a single channel image
	assert(im.channels() == 1);	
	
	// Placeholder for the landmarks
	cv::Mat_<float> current_shape(2 * pdm.NumberOfPoints() , 1, 0.0f);

	int n = pdm.NumberOfPoints(); 
		
	int num_scales = patch_experts.patch_scaling.size();

	// Storing the patch expert response maps
	std::vector<cv::Mat_<float> > patch_expert_responses(n);

	// Converting from image space to patch expert space (normalised for rotation and scale)
	cv::Matx22f sim_ref_to_img;
	cv::Matx22f sim_img_to_ref;

	FaceModelParameters tmp_parameters = parameters;

	// Active scale is there in case we need to upsample too much
	int active_scale = 0;

	// Optimise the model across a number of areas of interest (usually in descending window size and ascending scale size)
	for(int scale = 0; scale < num_scales; scale++)
	{
		// Control the number of iterations through window size
		if (window_sizes[scale] == 0)
			continue;

		int window_size = window_sizes[scale];

		// The patch expert response computation
		patch_experts.Response(patch_expert_responses, sim_ref_to_img, sim_img_to_ref, im, pdm, params_global, params_local, window_size, scale);

		if(parameters.refine_parameters == true)
		{
			int scale_max = scale >= 2 ? 2 : scale;

			// Adapt the parameters based on scale (wan't to reduce regularisation as scale increases, but increase sigma and Tikhonov)
			tmp_parameters.reg_factor = parameters.reg_factor - 15 * log(patch_experts.patch_scaling[scale_max]/0.25)/log(2);
			
			if(tmp_parameters.reg_factor <= 0)
				tmp_parameters.reg_factor = 0.001;

			tmp_parameters.sigma = parameters.sigma + 0.25 * log(patch_experts.patch_scaling[scale_max]/0.25)/log(2);
			tmp_parameters.weight_factor = parameters.weight_factor + 2 * parameters.weight_factor *  log(patch_experts.patch_scaling[scale_max]/0.25)/log(2);
		}

		// Get the current landmark locations
		pdm.CalcShape2D(current_shape, params_local, params_global);

		// Get the view used by patch experts
		int view_id = patch_experts.GetViewIdx(params_global, scale);
		this->view_used = view_id;

		// the actual optimisation step
		this->NU_RLMS(params_global, params_local, patch_expert_responses, cv::Vec6f(params_global), params_local.clone(), current_shape, sim_img_to_ref, sim_ref_to_img, window_size, view_id, true, scale, this->landmark_likelihoods, tmp_parameters, false);

		// non-rigid optimisation

		// If we are terminating next iteration, make sure to record the model likelihood
		if(scale == num_scales - 1 || window_sizes[scale + 1] == 0 || params_global[0] < 0.30)
		{			
			this->model_likelihood = this->NU_RLMS(params_global, params_local, patch_expert_responses, cv::Vec6f(params_global), params_local.clone(), current_shape, sim_img_to_ref, sim_ref_to_img, window_size, view_id, false, scale, this->landmark_likelihoods, tmp_parameters, true);
		}
		else
		{
			this->NU_RLMS(params_global, params_local, patch_expert_responses, cv::Vec6f(params_global), params_local.clone(), current_shape, sim_img_to_ref, sim_ref_to_img, window_size, view_id, false, scale, this->landmark_likelihoods, tmp_parameters, false);
		}

		// Can't track very small images reliably (less than ~30px across)
		if (params_global[0] < 0.25)
		{
			std::cout << "Face too small for landmark detection" << std::endl;
			return false;
		}

		// Making sure we do not upsample too much
		if (active_scale < num_scales - 1 && 0.9 * patch_experts.patch_scaling[active_scale + 1] < params_global[0])
			active_scale = active_scale + 1;

	}

	return true;
}

void CLNF::NonVectorisedMeanShift_precalc_kde(cv::Mat_<float>& out_mean_shifts, const std::vector<cv::Mat_<float> >& patch_expert_responses,
	const cv::Mat_<float> &dxs, const cv::Mat_<float> &dys, int resp_size, float a, int scale, int view_id, 
	std::map<int, cv::Mat_<float> >& kde_resp_precalc)
{
	
	int n = dxs.rows;
	
	cv::Mat_<float> kde_resp;
	float step_size = 0.1;

	// if this has not been precomputer, precompute it, otherwise use it
	if(kde_resp_precalc.find(resp_size) == kde_resp_precalc.end())
	{		
		kde_resp = cv::Mat_<float>((int)((resp_size / step_size)*(resp_size/step_size)), resp_size * resp_size);
		cv::MatIterator_<float> kde_it = kde_resp.begin();

		for(int x = 0; x < resp_size/step_size; x++)
		{
			float dx = x * step_size;
			for(int y = 0; y < resp_size/step_size; y++)
			{
				float dy = y * step_size;

				int ii,jj;
				float v,vx,vy;
			
				for(ii = 0; ii < resp_size; ii++)
				{
					vx = (dy-ii)*(dy-ii);
					for(jj = 0; jj < resp_size; jj++)
					{
						vy = (dx-jj)*(dx-jj);

						// the KDE evaluation of that point
						v = exp(a*(vx+vy));
						
						*kde_it++ = v;
					}
				}
			}
		}

		kde_resp_precalc[resp_size] = kde_resp.clone();
	}
	else
	{
		// use the precomputed version
		kde_resp = kde_resp_precalc.find(resp_size)->second;
	}

	// for every point (patch) calculating mean-shift
	for(int i = 0; i < n; i++)
	{
		if(patch_experts.visibilities[scale][view_id].at<int>(i,0) == 0)
		{
			out_mean_shifts.at<float>(i,0) = 0;
			out_mean_shifts.at<float>(i+n,0) = 0;
			continue;
		}

		// indices of dx, dy
		float dx = dxs.at<float>(i);
		float dy = dys.at<float>(i);

		// Ensure that we are within bounds (important for precalculation)
		if(dx < 0)
			dx = 0;
		if(dy < 0)
			dy = 0;
		if(dx > resp_size - step_size)
			dx = resp_size - step_size;
		if(dy > resp_size - step_size)
			dy = resp_size - step_size;
		
		// Pick the row from precalculated kde that approximates the current dx, dy best		
		int closest_col = (int)(dy /step_size + 0.5); // Plus 0.5 is there, as C++ rounds down with int cast
		int closest_row = (int)(dx /step_size + 0.5); // Plus 0.5 is there, as C++ rounds down with int cast
		
		int idx = closest_row * ((int)(resp_size/step_size + 0.5)) + closest_col; // Plus 0.5 is there, as C++ rounds down with int cast

		cv::MatIterator_<float> kde_it = kde_resp.begin() + kde_resp.cols*idx;
		
		float mx=0.0;
		float my=0.0;
		float sum=0.0;

		// Iterate over the patch responses here
		cv::MatConstIterator_<float> p = patch_expert_responses[i].begin();
			
		// TODO maybe do through MatMuls instead?
		for(int ii = 0; ii < resp_size; ii++)
		{
			for(int jj = 0; jj < resp_size; jj++)
			{

				// the KDE evaluation of that point multiplied by the probability at the current, xi, yi
				float v = (*p++) * (*kde_it++);

				sum += v;

				// mean shift in x and y
				mx += v*jj;
				my += v*ii; 

			}
		}
		
		float msx = (mx/sum - dx);
		float msy = (my/sum - dy);

		out_mean_shifts.at<float>(i,0) = msx;
		out_mean_shifts.at<float>(i+n,0) = msy;

	}

}

void CLNF::GetWeightMatrix(cv::Mat_<float>& WeightMatrix, int scale, int view_id, const FaceModelParameters& parameters)
{
	int n = pdm.NumberOfPoints();  

	// Is the weight matrix needed at all
	if(parameters.weight_factor > 0)
	{
		WeightMatrix = cv::Mat_<float>::zeros(n*2, n*2);

		for (int p=0; p < n; p++)
		{
			if (!patch_experts.cen_expert_intensity.empty())
			{

				// for the x dimension
				WeightMatrix.at<float>(p, p) = WeightMatrix.at<float>(p, p) + patch_experts.cen_expert_intensity[scale][view_id][p].confidence;

				// for they y dimension
				WeightMatrix.at<float>(p + n, p + n) = WeightMatrix.at<float>(p, p);

			}
			else if(!patch_experts.ccnf_expert_intensity.empty())
			{

				// for the x dimension
				WeightMatrix.at<float>(p,p) = WeightMatrix.at<float>(p,p)  + patch_experts.ccnf_expert_intensity[scale][view_id][p].patch_confidence;
				
				// for they y dimension
				WeightMatrix.at<float>(p+n,p+n) = WeightMatrix.at<float>(p,p);

			}
			else
			{
				// Across the modalities add the confidences
				for(size_t pc=0; pc < patch_experts.svr_expert_intensity[scale][view_id][p].svr_patch_experts.size(); pc++)
				{
					// for the x dimension
					WeightMatrix.at<float>(p,p) = WeightMatrix.at<float>(p,p)  + patch_experts.svr_expert_intensity[scale][view_id][p].svr_patch_experts.at(pc).confidence;
				}	
				// for the y dimension
				WeightMatrix.at<float>(p+n,p+n) = WeightMatrix.at<float>(p,p);
			}
		}
		WeightMatrix = parameters.weight_factor * WeightMatrix;
	}
	else
	{
		WeightMatrix = cv::Mat_<float>::eye(n*2, n*2);
	}

}

//=============================================================================
float CLNF::NU_RLMS(cv::Vec6f& final_global, cv::Mat_<float>& final_local, const std::vector<cv::Mat_<float> >& patch_expert_responses, const cv::Vec6f& initial_global, const cv::Mat_<float>& initial_local,
		          const cv::Mat_<float>& base_shape, const cv::Matx22f& sim_img_to_ref, const cv::Matx22f& sim_ref_to_img, int resp_size, int view_id, bool rigid, int scale, cv::Mat_<float>& landmark_lhoods,
				  const FaceModelParameters& parameters, bool compute_lhood)
{		

	int n = pdm.NumberOfPoints();  
	
	// Mean, eigenvalues, eigenvectors
	cv::Mat_<float> M = this->pdm.mean_shape;
	cv::Mat_<float> E = this->pdm.eigen_values;
	//Mat_<float> V = this->pdm.princ_comp;

	int m = pdm.NumberOfModes();
	
	cv::Vec6f current_global(initial_global);

	cv::Mat_<float> current_local = initial_local.clone();

	cv::Mat_<float> current_shape;
	cv::Mat_<float> previous_shape;

	// Pre-calculate the regularisation term
	cv::Mat_<float> regTerm;

	if(rigid)
	{
		regTerm = cv::Mat_<float>::zeros(6,6);
	}
	else
	{
		cv::Mat_<float> regularisations = cv::Mat_<float>::zeros(1, 6 + m);

		// Setting the regularisation to the inverse of eigenvalues
		cv::Mat(parameters.reg_factor / E).copyTo(regularisations(cv::Rect(6, 0, m, 1)));
		regTerm = cv::Mat::diag(regularisations.t());
	}	

	cv::Mat_<float> WeightMatrix;
	GetWeightMatrix(WeightMatrix, scale, view_id, parameters);

	cv::Mat_<float> dxs, dys;
	
	// The preallocated memory for the mean shifts
	cv::Mat_<float> mean_shifts(2 * pdm.NumberOfPoints(), 1, 0.0);

	// Number of iterations
	for(int iter = 0; iter < parameters.num_optimisation_iteration; iter++)
	{
		// get the current estimates of x
		pdm.CalcShape2D(current_shape, current_local, current_global);
		
		if(iter > 0)
		{
			// if the shape hasn't changed terminate
			if(norm(current_shape, previous_shape) < 0.01)
			{				
				break;
			}
		}

		current_shape.copyTo(previous_shape);
		
		// Jacobian, and transposed weighted jacobian
		cv::Mat_<float> J, J_w_t;

		// calculate the appropriate Jacobians in 2D, even though the actual behaviour is in 3D, using small angle approximation and oriented shape
		if(rigid)
		{
			pdm.ComputeRigidJacobian(current_local, current_global, J, WeightMatrix, J_w_t);
		}
		else
		{
			pdm.ComputeJacobian(current_local, current_global, J, WeightMatrix, J_w_t);
		}
		
		// useful for mean shift calculation
		float a = -0.5/(parameters.sigma * parameters.sigma);

		cv::Mat_<float> current_shape_2D = current_shape.reshape(1, 2).t();
		cv::Mat_<float> base_shape_2D = base_shape.reshape(1, 2).t();

		cv::Mat_<float> offsets;
		cv::Mat((current_shape_2D - base_shape_2D) * cv::Mat(sim_img_to_ref).t()).convertTo(offsets, CV_32F);
		
		dxs = offsets.col(0) + (resp_size-1)/2;
		dys = offsets.col(1) + (resp_size-1)/2;
		
		NonVectorisedMeanShift_precalc_kde(mean_shifts, patch_expert_responses, dxs, dys, resp_size, a, scale, view_id, kde_resp_precalc);

		// Now transform the mean shifts to the the image reference frame, as opposed to one of ref shape (object space)
		cv::Mat_<float> mean_shifts_2D = (mean_shifts.reshape(1, 2)).t();
		
		mean_shifts_2D = mean_shifts_2D * cv::Mat(sim_ref_to_img).t();
		mean_shifts = cv::Mat(mean_shifts_2D.t()).reshape(1, n*2);

		// remove non-visible observations
		for(int i = 0; i < n; ++i)
		{
			// if patch unavailable for current index
			if(patch_experts.visibilities[scale][view_id].at<int>(i,0) == 0)
			{				
				cv::Mat Jx = J.row(i);
				Jx = cvScalar(0);
				cv::Mat Jy = J.row(i+n);
				Jy = cvScalar(0);

				Jx = J_w_t.col(i);
				Jx = cvScalar(0);
				Jy = J_w_t.col(i + n);
				Jy = cvScalar(0);

				mean_shifts.at<float>(i,0) = 0.0f;
				mean_shifts.at<float>(i+n,0) = 0.0f;
			}
		}

		// projection of the meanshifts onto the jacobians (using the weighted Jacobian, see Baltrusaitis 2013)
		cv::Mat_<float> J_w_t_m = J_w_t * mean_shifts;

		// Add the regularisation term
		if(!rigid)
		{
			J_w_t_m(cv::Rect(0,6,1, m)) = J_w_t_m(cv::Rect(0,6,1, m)) - regTerm(cv::Rect(6,6, m, m)) * current_local;
		}

		cv::Mat_<float> Hessian = regTerm.clone();

		// Perform matrix multiplication in OpenBLAS (fortran call)
		float alpha1 = 1.0;
		float beta1 = 1.0;
		char N[2]; N[0] = 'N';
		sgemm_(N, N, &J.cols, &J_w_t.rows, &J_w_t.cols, &alpha1, (float*)J.data, &J.cols, (float*)J_w_t.data, &J_w_t.cols, &beta1, (float*)Hessian.data, &J.cols);

		// Above is a fast (but ugly) version of 
		// cv::Mat_<float> Hessian = J_w_t * J + regTerm;

		// Solve for the parameter update (from Baltrusaitis 2013 based on eq (36) Saragih 2011)
		cv::Mat_<float> param_update;
		cv::solve(Hessian, J_w_t_m, param_update, cv::DECOMP_CHOLESKY);
		
		// update the reference
		pdm.UpdateModelParameters(param_update, current_local, current_global);		
		
		// clamp to the local parameters for valid expressions
		pdm.Clamp(current_local, current_global, parameters);

	}

	// compute the log likelihood
	float loglhood = 0;
	
	if(compute_lhood)
	{
		landmark_lhoods = cv::Mat_<float>(n, 1, -1e8);
	
		for(int i = 0; i < n; i++)
		{

			if(patch_experts.visibilities[scale][view_id].at<int>(i,0) == 0 )
			{
				continue;
			}
			float dx = dxs.at<float>(i);
			float dy = dys.at<float>(i);

			int ii,jj;
			float v,vx,vy,sum=0.0;

			// Iterate over the patch responses here
			cv::MatConstIterator_<float> p = patch_expert_responses[i].begin();
			
			for(ii = 0; ii < resp_size; ii++)
			{
				vx = (dy-ii)*(dy-ii);
				for(jj = 0; jj < resp_size; jj++)
				{
					vy = (dx-jj)*(dx-jj);

					// the probability at the current, xi, yi
					v = *p++;

					// the KDE evaluation of that point
					v *= exp(-0.5*(vx+vy)/(parameters.sigma * parameters.sigma));

					sum += v;
				}
			}
			landmark_lhoods.at<float>(i,0) = sum;

			// the offset is there for numerical stability
			loglhood += log(sum + 1e-8);

		}	
		loglhood = loglhood/sum(patch_experts.visibilities[scale][view_id])[0];
	}

	final_global = current_global;
	final_local = current_local;

	return loglhood;

}

// Getting a 3D shape model from the current detected landmarks (in camera space)
cv::Mat_<float> CLNF::GetShape(float fx, float fy, float cx, float cy) const
{
	int n = this->detected_landmarks.rows/2;

	cv::Mat_<float> shape3d(n*3, 1);
	this->pdm.CalcShape3D(shape3d, this->params_local);

	// Need to rotate the shape to get the actual 3D representation
	
	// get the rotation matrix from the euler angles
	cv::Matx33f R = Utilities::Euler2RotationMatrix(cv::Vec3f((float)params_global[1], (float)params_global[2], (float)params_global[3]));

	shape3d = shape3d.reshape(1, 3);

	shape3d = shape3d.t() * cv::Mat(R).t();
	
	// from the weak perspective model can determine the average depth of the object
	float Zavg = fx / (float)params_global[0];	

	cv::Mat_<float> outShape(n, 3, 0.0f);

	// this is described in the paper in section 3.4 (equation 10) (of the CLM-Z paper)
	for(int i = 0; i < n; i++)
	{
		float Z = Zavg + shape3d.at<float>(i,2);

		float X = Z * ((this->detected_landmarks.at<float>(i) - cx)/fx);
		float Y = Z * ((this->detected_landmarks.at<float>(i + n) - cy)/fy);

		outShape.at<float>(i,0) = X;
		outShape.at<float>(i,1) = Y;
		outShape.at<float>(i,2) = Z;

	}

	// The format is 3 rows - n cols
	return outShape.t();
	
}

cv::Mat_<int> CLNF::GetVisibilities() const
{
	// Get the view of the largest scale
	int scale = patch_experts.visibilities.size() - 1;
	int view_id = patch_experts.GetViewIdx(params_global, scale);

	cv::Mat_<int> visibilities_to_ret = this->patch_experts.visibilities[scale][view_id].clone();
	return visibilities_to_ret;
}

// A utility bounding box function
cv::Rect_<float> CLNF::GetBoundingBox() const
{
	float min_x, max_x;
	float min_y, max_y;
	ExtractBoundingBox(this->detected_landmarks, min_x, max_x, min_y, max_y);

	cv::Rect_<float> model_rect(min_x, min_y, max_x - min_x, max_y - min_y);
	return model_rect;
}
