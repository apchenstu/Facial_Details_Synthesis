///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Tadas Baltrusaitis, all rights reserved.
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
#include "stdafx_ut.h"

#include "RecorderOpenFaceParameters.h"

using namespace Utilities;

RecorderOpenFaceParameters::RecorderOpenFaceParameters(std::vector<std::string> &arguments, bool sequence, bool from_webcam, float fx, float fy, float cx, float cy, double fps_vid_out)
{

	this->is_sequence = sequence;
	this->is_from_webcam = from_webcam;
	this->fx = fx;
	this->fy = fy;
	this->cx = cx;
	this->cy = cy;

	if(fps_vid_out > 0)
	{
		this->fps_vid_out = fps_vid_out;
	}
	else
	{
		this->fps_vid_out = 30; // If an illegal value for fps provided, default to 30
	}
	// Default output code
	this->output_codec = "DIVX";

	this->image_format_aligned = "bmp";
	this->image_format_visualization = "jpg";

	bool output_set = false;

	this->output_2D_landmarks = false;
	this->output_3D_landmarks = false;
	this->output_model_params = false;
	this->output_pose = false;
	this->output_AUs = false;
	this->output_gaze = false;
	this->output_hog = false;
	this->output_tracked = false;
	this->output_aligned_faces = false;

	this->record_aligned_bad = true;

	for (size_t i = 0; i < arguments.size(); ++i)
	{
		if (arguments[i].compare("-format_aligned") == 0)
		{
			this->image_format_aligned = arguments[i+1];
			i++;
		}
		if (arguments[i].compare("-format_vis_image") == 0)
		{
			this->image_format_visualization = arguments[i + 1];
			i++;
		}
		if (arguments[i].compare("-nobadaligned") == 0)
		{
			this->record_aligned_bad = false;
		}
		if (arguments[i].compare("-simalign") == 0)
		{
			this->output_aligned_faces = true;
			output_set = true;
		}
		else if (arguments[i].compare("-hogalign") == 0)
		{
			this->output_hog = true;
			output_set = true;
		}
		else if (arguments[i].compare("-2Dfp") == 0)
		{
			this->output_2D_landmarks = true;
			output_set = true;
		}
		else if (arguments[i].compare("-3Dfp") == 0)
		{
			this->output_3D_landmarks = true;
			output_set = true;
		}
		else if (arguments[i].compare("-pdmparams") == 0)
		{
			this->output_model_params = true;
			output_set = true;
		}
		else if (arguments[i].compare("-pose") == 0)
		{
			this->output_pose = true;
			output_set = true;
		}
		else if (arguments[i].compare("-aus") == 0)
		{
			this->output_AUs = true;
			output_set = true;
		}
		else if (arguments[i].compare("-gaze") == 0)
		{
			this->output_gaze = true;
			output_set = true;
		}
		else if (arguments[i].compare("-tracked") == 0)
		{
			this->output_tracked = true;
			output_set = true;
		}
	}

	// Output everything if nothing has been set

	if (!output_set)
	{
		this->output_2D_landmarks = true;
		this->output_3D_landmarks = true;
		this->output_model_params = true;
		this->output_pose = true;
		this->output_AUs = true;
		this->output_gaze = true;
		this->output_hog = true;
		this->output_tracked = true;
		this->output_aligned_faces = true;
	}

}

RecorderOpenFaceParameters::RecorderOpenFaceParameters(bool sequence, bool is_from_webcam, bool output_2D_landmarks, bool output_3D_landmarks,
	bool output_model_params, bool output_pose, bool output_AUs, bool output_gaze, bool output_hog, bool output_tracked,
	bool output_aligned_faces, bool record_bad, float fx, float fy, float cx, float cy, double fps_vid_out)
{
	this->is_sequence = sequence;
	this->is_from_webcam = is_from_webcam;
	this->fx = fx;
	this->fy = fy;
	this->cx = cx;
	this->cy = cy;

	if (fps_vid_out > 0)
	{
		this->fps_vid_out = fps_vid_out;
	}
	else
	{
		this->fps_vid_out = 30; // If an illegal value for fps provided, default to 30
	}
	// Default output code
	this->output_codec = "DIVX";

	this->image_format_aligned = "bmp";
	this->image_format_visualization = "jpg";

	this->output_2D_landmarks = output_2D_landmarks;
	this->output_3D_landmarks = output_3D_landmarks;
	this->output_model_params = output_model_params;
	this->output_pose = output_pose;
	this->output_AUs = output_AUs;
	this->output_gaze = output_gaze;
	this->output_hog = output_hog;
	this->output_tracked = output_tracked;
	this->output_aligned_faces = output_aligned_faces;
}