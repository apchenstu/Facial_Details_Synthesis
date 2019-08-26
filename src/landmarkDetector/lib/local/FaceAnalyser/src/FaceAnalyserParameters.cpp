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

#include "FaceAnalyserParameters.h"

#ifndef CONFIG_DIR
#define CONFIG_DIR "~"
#endif

using namespace FaceAnalysis;

FaceAnalyserParameters::FaceAnalyserParameters():root()
{
	// initialise the default values
	init();
}

FaceAnalyserParameters::FaceAnalyserParameters(std::string root_dir)
{
	this->root = root_dir;
	init();
	
}
FaceAnalyserParameters::FaceAnalyserParameters(std::vector<std::string> &arguments):root()
{

	// First element is reserved for the executable location (useful for finding relative model locs)
	this->root = fs::path(arguments[0]).parent_path();

	// initialise the default values
	init();

	bool* valid = new bool[arguments.size()];
	valid[0] = true;

	bool scale_set = false;
	bool size_set = false;

	for (size_t i = 1; i < arguments.size(); ++i)
	{
		valid[i] = true;

		if (arguments[i].compare("-au_static") == 0)
		{
			dynamic = false;
			valid[i] = false;
		}
		else if (arguments[i].compare("-g") == 0)
		{
			grayscale = true;
			valid[i] = false;
		}
		else if (arguments[i].compare("-nomask") == 0)
		{
			sim_align_face_mask = false;
			valid[i] = false;
		}
		else if (arguments[i].compare("-simscale") == 0)
		{
			sim_scale_out = stod(arguments[i + 1]);
			valid[i] = false;
			valid[i + 1] = false;
			scale_set = true;
			i++;
		}
		else if (arguments[i].compare("-simsize") == 0)
		{
			sim_size_out = stoi(arguments[i + 1]);
			valid[i] = false;
			valid[i + 1] = false;
			size_set = true;
			i++;
		}
	}

	for (int i = (int)arguments.size() - 1; i >= 0; --i)
	{
		if (!valid[i])
		{
			arguments.erase(arguments.begin() + i);
		}
	}

	if (dynamic)
	{
		this->model_location = "AU_predictors/main_dynamic_svms.txt";
	}
	else
	{
		this->model_location = "AU_predictors/main_static_svms.txt";
	}

	// If we set the size but not the scale, adapt the scale to the right size
	if (!scale_set && size_set) sim_scale_out = sim_size_out * (0.7 / 112.0);

	// Make sure model_location is valid
	// First check working directory, then the executable's directory, then the config path set by the build process.
	fs::path config_path = fs::path(CONFIG_DIR);
	fs::path model_path = fs::path(this->model_location);
	if (fs::exists(model_path))
	{
		this->model_location = model_path.string();
	}
	else if (fs::exists(root/model_path))
	{
		this->model_location = (root/model_path).string();
	}
	else if (fs::exists(config_path/model_path))
	{
		this->model_location = (config_path/model_path).string();
	}
	else
	{
		std::cout << "Could not find the face analysis module to load" << std::endl;
	}
}

void FaceAnalyserParameters::init()
{
	// Initialize default parameter values
	this->dynamic = true;
	this->grayscale = false;
	this->sim_scale_out = 0.7;
	this->sim_size_out = 112;
	this->sim_align_face_mask = true;

	this->model_location = "AU_predictors/main_dynamic_svms.txt";

	// Make sure model_location is valid
	// First check working directory, then the executable's directory, then the config path set by the build process.
	fs::path config_path = fs::path(CONFIG_DIR);
	fs::path model_path = fs::path(this->model_location);
	if (fs::exists(model_path))
	{
		this->model_location = model_path.string();
	}
	else if (fs::exists(root / model_path))
	{
		this->model_location = (root / model_path).string();
	}
	else if (fs::exists(config_path / model_path))
	{
		this->model_location = (config_path / model_path).string();
	}
	else
	{
		std::cout << "Could not find the face analysis module to load" << std::endl;
	}

	orientation_bins = std::vector<cv::Vec3d>();

}

// Use getters and setters for these as they might need to reload models and make sure the scale and size ratio makes sense
void FaceAnalyserParameters::setAlignedOutput(int output_size, double scale, bool masked)
{
	this->sim_size_out = output_size;
	// If we set the size but not the scale, adapt the scale to the right size
	if (scale == -1)
	{
		this->sim_scale_out = sim_size_out * (0.7 / 112.0);
	}
	else
	{
		this->sim_scale_out = scale;
	}

	this->sim_align_face_mask = masked;

}
// This will also change the model location
void FaceAnalyserParameters::OptimizeForVideos()
{
	// Set the post-processing to true and load a dynamic model
	dynamic = true;

	this->model_location = "AU_predictors/main_dynamic_svms.txt";

	// Make sure model_location is valid
	// First check working directory, then the executable's directory, then the config path set by the build process.
	fs::path config_path = fs::path(CONFIG_DIR);
	fs::path model_path = fs::path(this->model_location);
	if (fs::exists(model_path))
	{
		this->model_location = model_path.string();
	}
	else if (fs::exists(root / model_path))
	{
		this->model_location = (root / model_path).string();
	}
	else if (fs::exists(config_path / model_path))
	{
		this->model_location = (config_path / model_path).string();
	}
	else
	{
		std::cout << "Could not find the face analysis module to load" << std::endl;
	}

}

void FaceAnalyserParameters::OptimizeForImages()
{
	// Set the post-processing to true and load a dynamic model
	dynamic = false;

	this->model_location = "AU_predictors/main_static_svms.txt";

	// Make sure model_location is valid
	// First check working directory, then the executable's directory, then the config path set by the build process.
	fs::path config_path = fs::path(CONFIG_DIR);
	fs::path model_path = fs::path(this->model_location);
	if (fs::exists(model_path))
	{
		this->model_location = model_path.string();
	}
	else if (fs::exists(root / model_path))
	{
		this->model_location = (root / model_path).string();
	}
	else if (fs::exists(config_path / model_path))
	{
		this->model_location = (config_path / model_path).string();
	}
	else
	{
		std::cout << "Could not find the AU detection model to load" << std::endl;
	}
}

